"""N5: the adapter's capture and steering surface, MEASURED against a library-independent reference.

Interpretune's `tests/core/test_forward_convergence_parity.py` covers the FORWARD-level claim: that
interp-engine, TransformerBridge and NNsight observe the same HuggingFace forward. It draws the boundary
explicitly and leaves this half here, because composition-level parity needs the component's own loading
path rather than just `interp_engine` on `sys.path`.

So this module reaches the adapter the way a consumer does: it STAGES the component tree with
`build_component_tree`, enforces the manifest's `requires`, executes the entrypoint by path under a
synthetic module name, and pulls the module class out of a `CompositionRegistry`. That is the loader's
own mechanism, minus the download.

**Staged rather than pulled from the Hub, deliberately.** Pulling would test a PUBLISHED revision, which
is a different and older artifact than the tree under test, and would add a network dependency and a
private-repo token to a unit test. Staging tests what is about to be published.

**The reference is the HuggingFace model, never a participant.** A participant agreeing with itself is
not parity. Both the capture and the intervention are compared against tensors taken straight off the HF
module by hook, with neither interp-engine nor TransformerLens involved in producing them.

**`resid_pre` is the compared point, and the choice is load-bearing.** It means the same tensor in every
implementation. The tempting neighbours do not: this adapter's whole reason for existing is that
TransformerLens' block-level `hook_mlp_in`/`hook_attn_in` are PRE-norm while the engine's `mlp_in`/
`attn_in` are POST-norm. Comparing those would produce a real divergence that is not the one being
measured, and the natural response to an unexplained divergence is a looser tolerance, which would bury
the defect under a number chosen to hide it.

WHAT THIS MEASURED, and it is the point of N5 rather than a footnote. The numbers agree EXACTLY: capture
and intervention both match the HF reference to 0.0, not to a tolerance. What differed was never
numeric. It was that interp-engine steers EVERY prompt position by default while interpretune's contract
is the last token, so the two backends were computing different operations while agreeing about the
arithmetic of each. Three defects in this component were found by running this rather than reasoning
about it, all of them on a path that unit tests had covered only in its refusals:

  * `_Seam.forward` called `engine_model.forward(...)`, which has never existed on `EagerModel`, against
    1.3.6 or 1.5.1 alike. `fwd_w_intervention` therefore could not complete a single call.
  * `InterventionSpec` is a `NamedTuple`, so `isinstance(specs, (list, tuple))` unpacked a lone spec into
    its own fields.
  * the position-semantics divergence above.

None of the three is subtle once executed, and none was reachable by asserting signatures.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile

import pytest

MODEL_ID = "gpt2"
LAYER = 5
PROMPT = "The capital of France is"

# Tight because both paths run the SAME HuggingFace forward: the engine wraps the model in place rather
# than converting weights, so any difference would come from the observation mechanism. Measured at 0.0;
# the tolerance is here so a future regression reports a NUMBER rather than a boolean.
RTOL = ATOL = 1e-5

torch = pytest.importorskip("torch", reason="torch is a declared dependency")
pytest.importorskip("transformers", reason="transformers arrives with interpretune")


@pytest.fixture(scope="module")
def staged_module():
    """The entrypoint, executed from a staged component tree the way the hub loader executes it."""
    import yaml
    from interpretune.hub.components import enforce_component_requires
    from interpretune.hub.publish import build_component_tree

    out = pathlib.Path(tempfile.mkdtemp()) / "staged"
    build_component_tree(pathlib.Path("src/interp_engine_adapter"), out)
    manifest = yaml.safe_load((out / "it_component.yaml").read_text())

    # The requires gate is part of the loading path and belongs in the test: a component whose declared
    # engine range excludes the installed one must not quietly be exercised anyway.
    enforce_component_requires(manifest, source="staged")

    name = "it_hub_adapters.staged.parity"
    spec = importlib.util.spec_from_file_location(name, out / manifest["adapters"]["entrypoint"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, manifest


@pytest.fixture(scope="module")
def composition_class(staged_module):
    """The `(core, interp_engine)` module class, taken from a registry rather than imported.

    Importing `InterpEngineModule` directly would test the same code and a different claim. Going through
    registration is what makes this composition-level.
    """
    from interpretune.adapters.registration import CompositionRegistry, register_dynamic_adapter
    from interpretune.hub import adapters as hub_adapters
    from interpretune.hub.adapters import _partition_declared_compositions

    module, manifest = staged_module
    satisfiable, _ = _partition_declared_compositions(manifest, source="staged")
    register_dynamic_adapter("interp_engine", source="speediedan/it-interp-engine-adapter")

    token = hub_adapters._SUPPORTED_COMPOSITIONS.set(tuple(satisfiable))
    try:
        registry = CompositionRegistry()
        module.InterpEngineAdapter.register_adapter_ctx(registry)
    finally:
        hub_adapters._SUPPORTED_COMPOSITIONS.reset(token)

    key = next(
        k
        for k in registry.available_compositions()
        if sorted(getattr(a, "value", str(a)) for a in k[1:]) == ["core", "interp_engine"]
    )
    return registry.get(key)[0]


@pytest.fixture(scope="module")
def prompt_ids():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)(PROMPT, return_tensors="pt")["input_ids"]


@pytest.fixture(scope="module")
def hf_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="module")
def steer_vector():
    torch.manual_seed(0)
    return torch.randn(768)


@pytest.fixture(scope="module")
def hf_resid_pre(hf_model, prompt_ids):
    """Library-independent ground truth: the block's input, straight off the HF module."""
    captured = {}
    handle = hf_model.transformer.h[LAYER].register_forward_pre_hook(
        lambda _m, args: captured.__setitem__("resid_pre", args[0].detach().clone())
    )
    try:
        with torch.no_grad():
            hf_model(prompt_ids)
    finally:
        handle.remove()
    assert "resid_pre" in captured, "the pre-hook never fired; the reference would otherwise be vacuous"
    return captured["resid_pre"][0]


@pytest.fixture(scope="module")
def hf_steered_logits(hf_model, prompt_ids, steer_vector):
    """The reference intervention: add the vector at the LAST position of the block's input.

    Written against the HF module directly so the comparison cannot be circular. Last position because
    that is interpretune's intervention contract, which its bundled backends implement through
    `apply_intervention_to_last_token`.
    """

    def hook(_module, args):
        hidden = args[0].clone()
        hidden[:, -1, :] = hidden[:, -1, :] + steer_vector
        return (hidden,) + args[1:]

    handle = hf_model.transformer.h[LAYER].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            return hf_model(prompt_ids).logits
    finally:
        handle.remove()


@pytest.fixture(scope="module")
def hosted(staged_module, composition_class, hf_model, prompt_ids):
    """The composition class, minimally hosted around a real model.

    A full interpretune session would bring configuration this measurement does not use and cannot
    verify. What is under test is the adapter's capture and steering surface, so the host supplies
    exactly what `post_auto_model_init` reads and nothing else.
    """
    module, _ = staged_module

    class _Host(composition_class):
        def __init__(self):
            torch.nn.Module.__init__(self)
            object.__setattr__(self, "_hosted_model", hf_model)
            self.tokenizer = None
            self.it_cfg = type(
                "_Cfg", (), {"model_name_or_path": MODEL_ID, "interp_engine_cfg": module.InterpEngineConfig()}
            )()

        @property
        def model(self):
            return self._hosted_model

        def post_auto_model_init(self, *args, **kwargs):
            return None

    host = _Host()
    module.InterpEngineCaptureMixin.post_auto_model_init(host)
    return host


class TestReferenceIsLive:
    """Positive control.

    Every comparison below asserts closeness to an HF reference, and a degenerate reference would let all of them pass
    while measuring nothing.
    """

    def test_reference_activation_is_non_degenerate(self, hf_resid_pre, prompt_ids):
        assert hf_resid_pre.shape[0] == prompt_ids.shape[1], "one position per input token"
        assert torch.isfinite(hf_resid_pre).all(), "reference carries non-finite values"
        assert hf_resid_pre.abs().max() > 0, "reference is all zeros; comparisons against it prove nothing"

    def test_the_intervention_reference_actually_differs_from_baseline(self, hf_model, prompt_ids, hf_steered_logits):
        """If the steered reference equalled the baseline, the parity assertion would be trivially true."""
        with torch.no_grad():
            baseline = hf_model(prompt_ids).logits
        assert not torch.allclose(baseline, hf_steered_logits), "the reference intervention changed nothing"


class TestCaptureParity:
    def test_capture_matches_the_hf_reference(self, hosted, prompt_ids, hf_resid_pre):
        """The engine wraps the HF model in place, so this should be exact rather than close."""
        cache = hosted.engine_capture(prompt_ids, [f"blocks.{LAYER}.hook_resid_pre"])
        got = cache["resid_pre", LAYER]
        got = got[0] if got.ndim == 3 else got
        assert got.shape == hf_resid_pre.shape
        torch.testing.assert_close(got, hf_resid_pre, rtol=RTOL, atol=ATOL)


class TestInterventionParity:
    """The measurement N5 exists for, through the registered composition's model backend."""

    def _steered(self, hosted, prompt_ids, steer_vector):
        from interpretune.analysis.backends import InterventionSpec

        spec = InterventionSpec(intervention_tensor=steer_vector, mode="add", scale_factor=1.0)
        # A BARE spec, not a list. `InterventionSpec` is a NamedTuple, so this is the shape that used to
        # be unpacked into its own fields; passing it deliberately keeps that regression covered.
        # No coercion here any more, and its absence is the assertion. This used to normalize with
        # `o.logits if hasattr(o, "logits") else o`, which quietly accepted either an output object or a
        # tensor -- and so hid that the seam was returning the object where interpretune's protocol
        # declares `torch.Tensor`. The workaround was written before the mismatch was noticed.
        return hosted._model_backend.fwd_w_intervention(
            hosted.engine_model, {"input_ids": prompt_ids}, {f"blocks.{LAYER}.hook_resid_pre": spec}
        )

    def test_steered_logits_match_the_hf_reference(self, hosted, prompt_ids, steer_vector, hf_steered_logits):
        _, post = self._steered(hosted, prompt_ids, steer_vector)
        torch.testing.assert_close(post, hf_steered_logits, rtol=RTOL, atol=ATOL)

    def test_the_intervention_applies_at_the_last_position_only(self, hosted, prompt_ids, steer_vector):
        """The divergence this module found, pinned so it cannot come back.

        interp-engine steers EVERY prompt position by default; interpretune's contract is the last token.
        Nothing raises when they disagree: the shapes match, the forward completes, and only the meaning
        differs. Asserting the position COUNT catches that where comparing logits alone would not, since
        a whole-prompt intervention still produces plausible logits.
        """
        pre, post = self._steered(hosted, prompt_ids, steer_vector)
        changed = (post - pre).abs().max(dim=-1).values[0] > 1e-6
        assert int(changed.sum()) == 1, (
            f"{int(changed.sum())} of {changed.numel()} positions changed; interpretune's intervention "
            "contract is the last token. Steering every position completes and returns a plausible "
            "result, which is why this asserts the count rather than only the values."
        )
        assert bool(changed[-1]), "the changed position was not the last one"

    def test_the_baseline_is_an_unsteered_forward(self, hosted, prompt_ids, steer_vector, hf_model):
        """`fwd_w_intervention` returns (baseline, steered).

        A baseline that was itself steered would make the pair agree for the wrong reason.
        """
        pre, _ = self._steered(hosted, prompt_ids, steer_vector)
        with torch.no_grad():
            unsteered = hf_model(prompt_ids).logits
        torch.testing.assert_close(pre, unsteered, rtol=RTOL, atol=ATOL)


class TestTheProtocolsDeclaredShapesAreReturned:
    """`fwd`, `fwd_w_cache` and `fwd_w_intervention` return TENSORS, as the protocol declares.

    `fwd_w_cache` used to read `getattr(cache, "logits", None)`. The engine's `Cache` exposes `get`,
    `output` and `tensors`, with the logits at `cache.output.logits`, so that attribute never existed
    and the default fired on every call: `answer_logits` downstream was silently `None`, forever, in
    every version.

    A `getattr` default is a worse failure than a wrong attribute name. A wrong name raises the first
    time it runs; a default returns a plausible value indefinitely. So these assert the TYPE rather
    than only that a call succeeds, because succeeding was never the problem.
    """

    def test_fwd_returns_a_tensor(self, hosted, prompt_ids):
        out = hosted._model_backend.fwd(hosted.engine_model, {"input_ids": prompt_ids})
        assert isinstance(out, torch.Tensor), f"protocol declares torch.Tensor, got {type(out).__name__}"

    def test_fwd_w_cache_returns_a_tensor_and_not_none(self, hosted, prompt_ids):
        logits, cache = hosted._model_backend.fwd_w_cache(
            hosted.engine_model, {"input_ids": prompt_ids}, [f"blocks.{LAYER}.hook_resid_pre"]
        )
        assert logits is not None, "the regression: logits were None on every call"
        assert isinstance(logits, torch.Tensor), f"protocol declares torch.Tensor, got {type(logits).__name__}"
        assert cache is not None

    def test_fwd_w_intervention_returns_tensors(self, hosted, prompt_ids, steer_vector):
        from interpretune.analysis.backends import InterventionSpec

        spec = InterventionSpec(intervention_tensor=steer_vector, mode="add", scale_factor=1.0)
        pre, post = hosted._model_backend.fwd_w_intervention(
            hosted.engine_model, {"input_ids": prompt_ids}, {f"blocks.{LAYER}.hook_resid_pre": spec}
        )
        for name, value in (("baseline", pre), ("steered", post)):
            assert isinstance(value, torch.Tensor), f"{name} should be a tensor, got {type(value).__name__}"

    def test_the_cache_logits_agree_with_a_plain_forward(self, hosted, prompt_ids, hf_model):
        """Not merely a tensor: the RIGHT tensor. A shape-correct wrong tensor would pass the checks
        above, which is the failure mode this component exists to be careful about."""
        logits, _ = hosted._model_backend.fwd_w_cache(
            hosted.engine_model, {"input_ids": prompt_ids}, [f"blocks.{LAYER}.hook_resid_pre"]
        )
        with torch.no_grad():
            reference = hf_model(prompt_ids).logits
        torch.testing.assert_close(logits, reference, rtol=RTOL, atol=ATOL)
