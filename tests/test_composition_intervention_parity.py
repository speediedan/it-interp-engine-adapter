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


def _hf_steered(hf_model, prompt_ids, edits):
    """Logits from the HF model with `edits` applied by forward pre-hook: `(layer, vector, last_only)`.

    The reference is built here rather than taken from a participant, so a participant agreeing with itself cannot count
    as parity.
    """
    handles = []
    for layer, vector, last_only in edits:

        def make(vector=vector, last_only=last_only):
            def hook(_module, args):
                hidden = args[0].clone()
                if last_only:
                    hidden[:, -1, :] = hidden[:, -1, :] + vector
                else:
                    hidden = hidden + vector
                return (hidden,) + args[1:]

            return hook

        handles.append(hf_model.transformer.h[layer].register_forward_pre_hook(make()))
    try:
        with torch.no_grad():
            return hf_model(prompt_ids).logits
    finally:
        for handle in handles:
            handle.remove()


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

    # FILTERED BY COMPONENT KEY, not by adapters alone. Once the datamodule slot existed, two keys
    # matched this adapter set and `next()` could return the datamodule composition, so the "module
    # class" under test became `ITDataModule` and every host built on it failed inside `nn.Module`.
    key = next(
        k
        for k in registry.available_compositions()
        if k[0] == "module" and sorted(getattr(a, "value", str(a)) for a in k[1:]) == ["core", "interp_engine"]
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

    def _steered(self, hosted, prompt_ids, steer_vector, scope=None):
        from interpretune.analysis.backends import InterventionSpec

        extra = {"position_scope": scope} if scope is not None else {}
        spec = InterventionSpec(intervention_tensor=steer_vector, op="add", scale_factor=1.0, **extra)
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

    def test_the_scope_selects_the_position_set(self, hosted, prompt_ids, steer_vector):
        """The position-SET discriminator, which replaced a count assertion.

        A count is a weaker instrument than it looks: "one position changed" is satisfied by the WRONG
        one position. The set says which. That distinction is the whole point here, because both scopes
        produce plausible logits and a value comparison alone cannot separate them.

        Written to lift into interpretune's conformance suite (speediedan/interpretune#450) as the
        position-scope case; the body is deliberately free of anything specific to this component beyond
        the payload builder.
        """
        from interpretune.analysis.backends import PositionScope

        n = prompt_ids.shape[1]
        for scope, expected in (
            (PositionScope.LAST_TOKEN, {n - 1}),
            (PositionScope.ALL_POSITIONS, set(range(n))),
        ):
            pre, post = self._steered(hosted, prompt_ids, steer_vector, scope=scope)
            changed = {i for i, c in enumerate((post - pre).abs().max(dim=-1).values[0] > 1e-6) if bool(c)}
            assert changed == expected, f"{scope.value}: changed {sorted(changed)}, expected {sorted(expected)}"

    def test_the_prompt_can_discriminate_the_two_scopes(self, prompt_ids):
        """A PRECONDITION on the fixture, not a comparison of results, and the difference matters.

        The two exact-set cases above already catch a scope-ignoring backend: an always-last one fails
        `all_positions`, an always-all one fails `last_token`. What they cannot catch is a prompt of ONE
        token, where both scopes legitimately yield `{0}` and the sets become identical. A scope-ignoring
        backend passes both cases there, and the suite reports green while measuring nothing.

        Asserting "the two sets differ" would be the wrong guard for the same reason: on a one-token
        prompt a CORRECT backend produces identical sets, so that assertion fails on correct behaviour.
        The discriminating power lives in the test DATA, so that is what gets asserted.

        Measured on gpt2: at n=5 the sets are {4} and {0..4}; at n=1 both are {0}.

        COUNTS THE SHAPE, WHICH IS ONLY SAFE BECAUSE THIS FIXTURE IS UNPADDED. `prompt_ids` here is one
        tokenized string with no attention mask, so shape equals real positions. A padded batch would
        make this assertion too weak in exactly the way it exists to prevent: two columns of which one
        is padding still leaves one real position. Interpretune's conformance suite counts REAL
        positions per batch from the mask for that reason, and supersedes this when item 4 of
        speediedan/it-interp-engine-adapter#14 adopts it.
        """
        assert prompt_ids.shape[1] >= 2, (
            f"prompt is {prompt_ids.shape[1]} token(s); the two position scopes are indistinguishable "
            "below 2, so the exact-set cases would pass a backend that ignores scope entirely."
        )

    def test_each_scope_matches_a_hand_built_reference(self, hosted, prompt_ids, steer_vector, hf_model):
        """The set says WHICH positions; this says the values there are right.

        A backend could edit the correct positions with the wrong arithmetic and satisfy the discriminator alone.
        """
        from interpretune.analysis.backends import PositionScope

        for scope, last_only in ((PositionScope.LAST_TOKEN, True), (PositionScope.ALL_POSITIONS, False)):
            reference = _hf_steered(hf_model, prompt_ids, [(LAYER, steer_vector, last_only)])
            _, post = self._steered(hosted, prompt_ids, steer_vector, scope=scope)
            torch.testing.assert_close(post, reference, rtol=RTOL, atol=ATOL)

    def test_a_mixed_scope_payload_is_expressed_rather_than_refused(self, hosted, prompt_ids, hf_model):
        """Measured, not assumed. `position_mask` is per CALL and `SteerSpec` carries no position, so the obvious
        reading is that one payload cannot hold two scopes. `steer` contexts NEST, which gives a mask per group, so
        the payload is grouped by scope instead of refused.

        Refusing would have been the mirror image of the defect this component exists to prevent: declining
        an operation the backend performs exactly, rather than substituting a wrong one, but still an
        answer that misrepresents what it can do.

        This composes because every declared op is additive and addition commutes, so group order cannot
        matter. `test_the_declared_ops_equal_the_mapped_ops` is what keeps that true.
        """
        from interpretune.analysis.backends import InterventionSpec, PositionScope

        torch.manual_seed(1)
        v_all, v_last = torch.randn(768), torch.randn(768)
        payload = {
            "blocks.3.hook_resid_pre": InterventionSpec(
                intervention_tensor=v_all, op="add", scale_factor=1.0, position_scope=PositionScope.ALL_POSITIONS
            ),
            "blocks.7.hook_resid_pre": InterventionSpec(
                intervention_tensor=v_last, op="add", scale_factor=1.0, position_scope=PositionScope.LAST_TOKEN
            ),
        }
        _, post = hosted._model_backend.fwd_w_intervention(hosted.engine_model, {"input_ids": prompt_ids}, payload)
        reference = _hf_steered(hf_model, prompt_ids, [(3, v_all, False), (7, v_last, True)])
        torch.testing.assert_close(post, reference, rtol=RTOL, atol=ATOL)

    def test_the_baseline_is_an_unsteered_forward(self, hosted, prompt_ids, steer_vector, hf_model):
        """`fwd_w_intervention` returns (baseline, steered).

        A baseline that was itself steered would make the pair agree for the wrong reason.
        """
        pre, _ = self._steered(hosted, prompt_ids, steer_vector)
        with torch.no_grad():
            unsteered = hf_model(prompt_ids).logits
        torch.testing.assert_close(pre, unsteered, rtol=RTOL, atol=ATOL)


class TestTheDeclarationAgreesWithTheImplementation:
    """`intervention_support` says what the backend claims; the refusal table says what it does.

    Drift between them is undetectable from either side alone, and it is what the conformance suite exists to catch
    across backends. This is the one-line version for this component.
    """

    def test_the_declared_ops_equal_the_mapped_ops(self, staged_module):
        module, _ = staged_module
        backend = module.InterpEngineModelBackend()
        declared = set(backend.intervention_support.ops)
        mapped = set(module._OP_TO_STEER_METHOD)
        assert declared == mapped, (
            f"declared ops {sorted(declared)} != mapped {sorted(mapped)}. A declaration wider than the "
            "table promises an op the seam refuses; narrower refuses one it can perform."
        )

    def test_both_scopes_are_declared_because_the_engine_expresses_both(self, staged_module):
        from interpretune.analysis.backends import PositionScope

        module, _ = staged_module
        scopes = module.InterpEngineModelBackend().intervention_support.position_scopes
        assert scopes == {PositionScope.LAST_TOKEN, PositionScope.ALL_POSITIONS}


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

        spec = InterventionSpec(intervention_tensor=steer_vector, op="add", scale_factor=1.0)
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
