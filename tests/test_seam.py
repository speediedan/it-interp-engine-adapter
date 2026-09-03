"""The seam's own tests. This is where this adapter's correctness lives.

Interpretune's core suite owns the RAILS (that the `adapters` kind loads, gates and registers, tested
against a fixture) and ONE optional contract test (that the published component still satisfies them).
Everything about what this adapter actually does belongs here, so that an interp-engine minor release
moving a signature turns THIS repo red rather than a framework nobody expects to break.
"""

from __future__ import annotations

import pytest

# INTERNALS COME FROM `.adapter`, NOT FROM THE PACKAGE, and the distinction is load-bearing rather than
# stylistic. `adapter.py` is the file the hub loader executes by path out of the snapshot; the package
# `__init__` exists only for the pip path and the loader never runs it. Importing the module under test
# directly means these tests exercise the same object a hub consumer gets. `TestPipInstallSurface` below
# covers the other path deliberately, so a break in either is attributable.
from interp_engine_adapter.adapter import (
    ADAPTER_NAME,
    InterpEngineAdapter,
    InterpEngineConfig,
    InterpEngineModule,
    UnmappableHookError,
    _Seam,
)


class TestConfig:
    def test_eager_is_the_default(self):
        assert InterpEngineConfig().backend == "eager"

    def test_vllm_is_refused_with_a_reason(self):
        """Out of scope deliberately, so the refusal explains rather than merely rejects."""
        with pytest.raises(ValueError, match="only 'eager'"):
            InterpEngineConfig(backend="vllm")


class TestHookNameParsing:
    @pytest.mark.parametrize(
        ("hook_name", "expected"),
        [
            ("blocks.5.mlp.hook_in", (5, "mlp.hook_in")),
            ("blocks.12.ln2.hook_out", (12, "ln2.hook_out")),
            ("blocks.0.hook_resid_post", (0, "hook_resid_post")),
            ("unembed.hook_in", (None, "unembed.hook_in")),
        ],
    )
    def test_layer_and_base_are_split(self, hook_name, expected):
        assert _Seam._split(hook_name) == expected


class TestSublayerInputRefusal:
    """The behaviour this adapter exists to get right; see NOTE [hook names lie] in the entrypoint."""

    @pytest.mark.parametrize(
        ("refused", "suggested"),
        [("blocks.5.hook_mlp_in", "mlp.hook_in"), ("blocks.5.hook_attn_in", "attn.hook_in")],
    )
    def test_legacy_block_level_inputs_are_refused(self, refused, suggested):
        with pytest.raises(UnmappableHookError) as excinfo:
            _Seam.point_for(refused, model=None)
        message = str(excinfo.value)
        assert suggested in message, "the refusal must name what to ask for instead"
        assert "hook_resid_mid" in message, "and name the hook for the tensor it actually is"

    def test_refusal_does_not_need_a_model(self):
        """It must fire before any engine call, so a caller learns at translation time."""
        with pytest.raises(UnmappableHookError):
            _Seam.point_for("blocks.0.hook_mlp_in", model=None)

    @pytest.mark.parametrize("hook_name", ["blocks.5.mlp.hook_in", "blocks.5.ln2.hook_out"])
    def test_both_spellings_of_the_sublayer_input_agree(self, hook_name):
        """The norm's output IS the MLP's argument: measured identical, cos 1.000000."""
        point = _Seam.point_for(hook_name, model=None)
        assert point.name == "mlp_in"
        assert point.layer == 5


#: circuit-tracer is OPTIONAL and CI does not install it. Tests needing the real pairing class skip
#: there and run locally; the tests that matter most for CI are the ABSENT-path ones, which is the
#: environment CI actually has. Stated here rather than left to a reader, because a suite that silently
#: skips its own subject is the failure mode this repo exists to avoid.
_HAS_CT = True
try:  # pragma: no cover - environment probe
    import circuit_tracer  # noqa: F401
    from interpretune.adapters.circuit_tracer.adapter import CircuitTracerAdapter  # noqa: F401
except Exception:  # pragma: no cover
    _HAS_CT = False

requires_ct = pytest.mark.skipif(not _HAS_CT, reason="circuit-tracer not installed (optional dependency)")


def _names(registry):
    """Composition keys as readable `a+b` strings, order-insensitive like the registry itself."""
    return sorted(
        "+".join(sorted(a.name if hasattr(a, "name") else str(a) for a in key[1:]))
        for key in registry.available_compositions()
    )


def _fresh_registry():
    from interpretune.adapters.registration import CompositionRegistry, register_dynamic_adapter

    register_dynamic_adapter(ADAPTER_NAME, source="speediedan/it-interp-engine-adapter")
    return CompositionRegistry()


class TestRegistration:
    def test_registers_the_unconditional_compositions(self):
        registry = _fresh_registry()
        InterpEngineAdapter.register_adapter_ctx(registry)

        names = _names(registry)
        assert "core+interp_engine" in names
        assert "interp_engine+lightning" in names
        for key in registry.available_compositions():
            if "circuit_tracer" not in str(key):
                assert registry.get(key) == (InterpEngineModule,)


class TestCompositionKeysMatchTheRails:
    """The key this component builds must equal the key the rails build, or every composition is skipped.

    Not a formality. The rails compare a component's registrations against the satisfiable set they
    computed from the manifest, and a mismatched key looks exactly like "this composition is unavailable
    here" -- the documented, correct outcome for something genuinely unsatisfiable. So a key bug does not
    raise; it silently registers nothing and the diagnostic points at registration rather than at the key.
    """

    def test_key_matches_the_rails_key_for_every_declared_composition(self):
        from interpretune.hub.adapters import _composition_key

        for combination, _needs_ct, _desc in InterpEngineAdapter._COMPOSITIONS:
            mine = InterpEngineAdapter._composition_key(combination)
            theirs = _composition_key({"component": "module", "adapters": list(mine[1:])})
            assert mine == theirs, f"key mismatch for {combination}: {mine} vs {theirs}"

    def test_the_key_uses_adapter_name_not_str(self):
        """`str(Adapter.core)` is 'Adapter.core'.

        A key built with `str()` never matches, and the failure is silent. This asserts the actual trap rather than only
        the happy path.
        """
        from interpretune.protocol import Adapter

        assert str(Adapter.core) != Adapter.core.name, "premise changed; this guard may no longer be needed"
        key = InterpEngineAdapter._composition_key((Adapter.core, ADAPTER_NAME))
        assert key == ("module", "core", "interp_engine")
        assert not any(seg.startswith("Adapter.") for seg in key)


class TestConditionalCompositionRegistration:
    """One published component, whatever compositions the environment supports.

    The rails hand an executing entrypoint the SATISFIABLE set; a direct import gets `None`, meaning no
    manifest governed this invocation, and the component answers the availability question itself.
    """

    def _register_with_supported(self, supported):
        from interpretune.hub import adapters as hub_adapters

        token = hub_adapters._SUPPORTED_COMPOSITIONS.set(supported)
        try:
            registry = _fresh_registry()
            InterpEngineAdapter.register_adapter_ctx(registry)
            return _names(registry)
        finally:
            hub_adapters._SUPPORTED_COMPOSITIONS.reset(token)

    def test_rails_without_circuit_tracer_get_only_the_core_compositions(self):
        names = self._register_with_supported(
            (("module", "core", "interp_engine"), ("module", "interp_engine", "lightning"))
        )
        assert names == ["core+interp_engine", "interp_engine+lightning"]

    @requires_ct
    def test_rails_with_circuit_tracer_get_all_three(self):
        names = self._register_with_supported(
            (
                ("module", "core", "interp_engine"),
                ("module", "interp_engine", "lightning"),
                ("module", "circuit_tracer", "core", "interp_engine"),
            )
        )
        assert names == [
            "circuit_tracer+core+interp_engine",
            "core+interp_engine",
            "interp_engine+lightning",
        ]

    def test_direct_import_without_circuit_tracer_registers_the_core_compositions(self, monkeypatch):
        """The pip path in an environment that never installed the optional package.

        THE case the maintainer's requirement is about: the core compositions must work with circuit-tracer absent.
        """
        import interp_engine_adapter.adapter as mod

        monkeypatch.setattr(mod, "_CT_PAIRING", None)
        monkeypatch.setattr(mod, "_circuit_tracer_pairing", lambda: None)
        registry = _fresh_registry()
        InterpEngineAdapter.register_adapter_ctx(registry)
        assert _names(registry) == ["core+interp_engine", "interp_engine+lightning"]

    def test_a_satisfiable_composition_that_cannot_build_raises_rather_than_registering_nothing(self, monkeypatch):
        """Disagreement between the rails and this component is reported HERE, naming the cause.

        Skipping would leave the rails' own invariant to report it as a registration mismatch, which points at the wrong
        thing.
        """
        import interp_engine_adapter.adapter as mod
        from interpretune.hub import adapters as hub_adapters

        monkeypatch.setattr(mod, "_CT_PAIRING", None)
        monkeypatch.setattr(mod, "_circuit_tracer_pairing", lambda: None)
        token = hub_adapters._SUPPORTED_COMPOSITIONS.set((("module", "circuit_tracer", "core", "interp_engine"),))
        try:
            with pytest.raises(UnmappableHookError, match="reported satisfiable"):
                InterpEngineAdapter.register_adapter_ctx(_fresh_registry())
        finally:
            hub_adapters._SUPPORTED_COMPOSITIONS.reset(token)


class TestManifestDeclaresTheConditionality:
    """The manifest must STATE that the third composition is conditional, not merely omit the condition.

    An unconditional declaration advertises three compositions to every reader while an environment without circuit-
    tracer yields two, with nothing saying so. That is the declarative twin of the silent registration loss the rails'
    skip reporting exists to remove.
    """

    def _manifest(self):
        from pathlib import Path

        import yaml

        import interp_engine_adapter.adapter as mod

        return yaml.safe_load((Path(mod.__file__).parent / "it_component.yaml").read_text())

    def test_the_circuit_tracer_composition_declares_requires(self):
        entries = self._manifest()["adapters"]["compositions"]
        ct = [e for e in entries if "circuit_tracer" in e["adapters"]]
        assert len(ct) == 1, "expected exactly one circuit-tracer composition"
        assert ct[0].get("requires", {}).get("pip") == ["circuit-tracer"]

    def test_the_core_compositions_declare_no_requires(self):
        """They must not, or an environment without circuit-tracer would get nothing at all."""
        entries = self._manifest()["adapters"]["compositions"]
        for entry in entries:
            if "circuit_tracer" not in entry["adapters"]:
                assert not entry.get("requires"), f"{entry['adapters']} should be unconditional"

    def test_every_declared_composition_is_one_the_code_can_register(self):
        """Manifest and code compared entry for entry, in the direction that catches drift: a manifest advertising
        something the code cannot produce is a promise to a reader that nothing will keep."""
        declared = {("module", *sorted(e["adapters"])) for e in self._manifest()["adapters"]["compositions"]}
        known = {InterpEngineAdapter._composition_key(c) for c, _, _ in InterpEngineAdapter._COMPOSITIONS}
        assert declared == known, f"manifest and code disagree: {declared ^ known}"


@requires_ct
class TestCircuitTracerPairing:
    """The composed class, and the refusal that must survive composition."""

    def test_the_mro_puts_circuit_tracer_ahead_of_this_adapters_base(self):
        """Mirrors the bundled pairings: circuit-tracer's layers win over the partner backend's base,
        and the pairing mixin wins over both."""
        import interp_engine_adapter.adapter as mod

        pairing = mod._circuit_tracer_pairing()
        names = [c.__name__ for c in pairing.__mro__]
        assert names[1] == "CircuitTracerInterpEngineModuleMixin"
        assert names.index("BaseCircuitTracerModule") < names.index("BaseInterpEngineModule")
        assert names.index("CircuitTracerAdapter") < names.index("BaseInterpEngineModule")

    def test_the_pairing_is_cached_not_rebuilt(self):
        import interp_engine_adapter.adapter as mod

        assert mod._circuit_tracer_pairing() is mod._circuit_tracer_pairing()

    def _guarded_model(self):
        import interp_engine_adapter.adapter as mod

        mixin = mod._circuit_tracer_pairing().__mro__[1]

        class _FakeReplacementModel:
            def _resolve_feature_hook(self, hook):
                return f"resolved:{hook}"

        class _Host:
            _replacement_model = _FakeReplacementModel()
            _guard_feature_hook_resolution = mixin._guard_feature_hook_resolution

        host = _Host()
        host._guard_feature_hook_resolution()
        return host._replacement_model

    @pytest.mark.parametrize("hook", ["hook_mlp_in", "hook_attn_in"])
    def test_the_refusal_survives_composition(self, hook):
        """THE point of the pairing.

        circuit-tracer resolves transcoder hook names through interp-engine's mapper, which resolves these PRE-norm
        names to the POST-norm sublayer inputs with no refusal. A three-way composition that silently regained that
        would be worse than not shipping one, because the refusal in the other two is what tells a user this adapter is
        careful about it.
        """
        with pytest.raises(UnmappableHookError, match="BEFORE the block norm"):
            self._guarded_model()._resolve_feature_hook(hook)

    @pytest.mark.parametrize("hook", ["hook_resid_post", "ln2.hook_normalized"])
    def test_unambiguous_hooks_still_reach_circuit_tracer(self, hook):
        """The guard is narrow on purpose.

        `ln2.hook_normalized` in particular must pass through:
        circuit-tracer handles it correctly via its own norm lookup, and refusing it here would break a
        case that upstream already gets right.
        """
        assert self._guarded_model()._resolve_feature_hook(hook) == f"resolved:{hook}"

    def test_guarding_twice_does_not_double_wrap(self):
        model = self._guarded_model()
        first = model._resolve_feature_hook
        assert getattr(first, "_it_seam_guarded", False)


class TestModelBackendIsAttachedNotOverridden:
    """The composition contract: attach a model backend only when nothing else has.

    This is the half of composition that fails SILENTLY. If two adapters in one MRO each assign
    `_model_backend` unconditionally, the composition still constructs, still runs, and simply loses one
    adapter's capabilities -- decided by MRO order, reported nowhere. Interpretune's circuit-tracer
    adapter states the contract and honours it; these tests are what keep this side of a composition
    honouring it too, because nothing upstream tests this adapter's behaviour.
    """

    def _module_with(self, existing):
        """A stand-in carrying just the surface `post_auto_model_init` touches.

        Deliberately not an `InterpEngineModule`: constructing one pulls a real interpretune session, and
        what is under test is the attach DECISION, which reads `__dict__` and calls the seam. A narrow
        double keeps the failure attributable to that decision rather than to session setup.
        """
        from interp_engine_adapter.adapter import InterpEngineCaptureMixin

        class _Rest:
            """Stands in for the rest of the MRO, which the mixin's `super()` call reaches."""

            def post_auto_model_init(self, *args, **kwargs):
                return "inner"

        class _Composed(InterpEngineCaptureMixin, _Rest):
            def __init__(self):
                self.it_cfg = None
                self.model = object()
                self.tokenizer = None
                if existing is not None:
                    self._model_backend = existing

        return _Composed()

    def test_attaches_when_nothing_has(self, monkeypatch):
        import interp_engine_adapter.adapter as mod

        monkeypatch.setattr(mod._Seam, "wrap_model", staticmethod(lambda *a, **k: "engine"))
        m = self._module_with(None)
        m.post_auto_model_init()
        assert isinstance(m._model_backend, mod.InterpEngineModelBackend)

    def test_does_not_override_a_backend_another_adapter_attached(self, monkeypatch):
        """The regression that matters.

        An unconditional assignment passes every single-adapter test and only misbehaves in a composition, which is
        where nobody is looking.
        """
        import interp_engine_adapter.adapter as mod

        monkeypatch.setattr(mod._Seam, "wrap_model", staticmethod(lambda *a, **k: "engine"))
        sentinel = object()
        m = self._module_with(sentinel)
        m.post_auto_model_init()
        assert m._model_backend is sentinel, "clobbered a backend another adapter in the MRO attached"

    def test_the_engine_is_still_wrapped_when_the_backend_is_left_alone(self, monkeypatch):
        """Not overriding the BACKEND must not skip the WRAP.

        They are separate responsibilities, and guarding both under one condition would disable capture in exactly the
        composed case.
        """
        import interp_engine_adapter.adapter as mod

        monkeypatch.setattr(mod._Seam, "wrap_model", staticmethod(lambda *a, **k: "engine"))
        m = self._module_with(object())
        m.post_auto_model_init()
        assert m.engine_model == "engine"


class TestModelBackend:
    """The backend claims exactly what it implements, and refuses what it does not."""

    def test_claims_intervention_and_nothing_else(self):
        from interpretune.analysis.backends.capabilities import BackendCapability
        from interp_engine_adapter.adapter import InterpEngineModelBackend

        backend = InterpEngineModelBackend()
        assert backend.supports(BackendCapability.INTERVENTION)
        for absent in (BackendCapability.LATENT_MODELS, BackendCapability.GRADIENTS, BackendCapability.BATCHED_HOOKS):
            assert not backend.supports(absent)

    def test_satisfies_core_and_intervention_protocols_only(self):
        from interpretune.analysis.backends.protocols import (
            ModelBackend,
            ModelBackendCore,
            SupportsGradients,
            SupportsIntervention,
            SupportsLatentModels,
        )
        from interp_engine_adapter.adapter import InterpEngineModelBackend

        backend = InterpEngineModelBackend()
        assert isinstance(backend, ModelBackendCore)
        assert isinstance(backend, SupportsIntervention)
        # The point of the decomposition: a partial backend is NOT the full protocol, and structural
        # checks say so rather than the mismatch surfacing as an AttributeError inside an op.
        assert not isinstance(backend, SupportsLatentModels)
        assert not isinstance(backend, SupportsGradients)
        assert not isinstance(backend, ModelBackend)

    def test_latent_handles_are_refused_not_ignored(self):
        from interp_engine_adapter.adapter import InterpEngineModelBackend, UnmappableHookError

        backend = InterpEngineModelBackend()
        with pytest.raises(UnmappableHookError, match="does not support latent-model handles"):
            backend.fwd_w_intervention(None, {"input_ids": None}, {}, latent_model_handles=[object()])

    def test_a_callable_names_filter_is_refused_rather_than_under_capturing(self):
        from interp_engine_adapter.adapter import InterpEngineModelBackend, UnmappableHookError

        backend = InterpEngineModelBackend()
        with pytest.raises(UnmappableHookError, match="callable names_filters"):
            backend.fwd_w_cache(None, {"input_ids": None}, lambda name: True)


class TestInterventionTargetsAreModelAware:
    """Steering must not resolve a hook name without the model.

    `tlens_hook_to_point` ACCEPTS a null model and returns the string-only reading rather than
    raising, and that reading differs on sandwich-norm architectures for exactly the block-level
    `hook_mlp_out`/`hook_attn_out`. Those are not in the alias table, so a model-blind path would
    resolve them to the wrong point and steering would WRITE there, silently.
    """

    def test_a_null_model_is_refused(self):
        from interp_engine_adapter.adapter import UnmappableHookError, _intervention_targets

        with pytest.raises(UnmappableHookError, match="a model is required"):
            _intervention_targets({"blocks.5.hook_mlp_out": object()}, None)

    def test_the_model_blind_reading_really_does_differ(self):
        """Positive control for the refusal above, and the reason it exists.

        If the engine ever started raising on a null model, or the alias table grew to cover these names, the refusal
        would be guarding nothing -- and this test would say so by failing.
        """
        from interp_engine import tlens_hook_to_point

        # These resolve WITHOUT a model, which is precisely the hazard.
        for name, expected in (("blocks.5.hook_mlp_out", "mlp_out"), ("blocks.5.hook_attn_out", "attn_out")):
            address = tlens_hook_to_point(name, None)
            assert address.name == expected and address.layer == 5
            assert name.split(".", 2)[2] not in _seam_alias_table(), (
                f"{name} is now alias-resolved; revisit whether the model-aware requirement still binds"
            )


def _seam_alias_table() -> dict:
    from interp_engine_adapter.adapter import _Seam

    return _Seam._SUBLAYER_INPUT_ALIASES


class TestInterventionModeIsHonoured:
    """`InterventionSpec.mode` must reach the engine, or be refused -- never silently dropped.

    The first draft of this backend built every SteerSpec with the default method, so `mode` was
    ignored entirely. Interpretune's DEFAULT mode is `replace`, and the J-space steering path uses
    `patch`; both would have been applied as ADDITIVE steering, completing normally and returning
    plausible logits for an intervention nobody requested.
    """

    def test_add_maps_to_additive(self):
        from interp_engine_adapter.adapter import _steer_method_for

        assert _steer_method_for("add") == "additive"

    @pytest.mark.parametrize("mode", ["replace", "patch", "project"])
    def test_modes_that_need_the_current_activation_are_refused(self, mode):
        from interp_engine_adapter.adapter import UnmappableHookError, _steer_method_for

        with pytest.raises(UnmappableHookError, match=f"intervention mode {mode!r} cannot be expressed"):
            _steer_method_for(mode)

    def test_every_mapped_method_is_one_the_engine_actually_offers(self):
        """Positive control: the mapping table must name real engine methods, not aspirational ones."""
        from interp_engine import STEER_METHODS

        from interp_engine_adapter.adapter import _MODE_TO_STEER_METHOD

        for mode, method in _MODE_TO_STEER_METHOD.items():
            assert method in STEER_METHODS, f"{mode!r} maps to {method!r}, absent from {STEER_METHODS}"

    def test_the_refused_modes_are_exactly_the_ones_interpretune_defines(self):
        """If interpretune adds a mode, this test fails until someone decides how it maps.

        The failure mode being guarded is a NEW mode defaulting into silence: without this, a mode
        added upstream would simply not appear in the table and would raise only when someone used it.
        """
        from interpretune.analysis.backends.interventions import InterventionSpec

        documented = {"replace", "add", "patch", "project"}
        # `_field_defaults`, not `InterventionSpec.mode`: this is a NamedTuple, so the attribute is the
        # field DESCRIPTOR and comparing it to a string is always False -- an assertion that would have
        # failed for a reason unrelated to what it checks.
        assert InterventionSpec._field_defaults["mode"] == "replace", (
            "the default intervention mode changed; revisit which modes this backend refuses"
        )
        unmapped = documented - set(_mode_table())
        assert unmapped == {"replace", "patch", "project"}, (
            f"interpretune's intervention modes changed: {unmapped}. Decide the mapping or the refusal."
        )


def _mode_table() -> dict:
    from interp_engine_adapter.adapter import _MODE_TO_STEER_METHOD

    return _MODE_TO_STEER_METHOD


class TestTheProjectOrthogonalBoundaryIsGuardedByCode:
    """`project` must never be mapped to `orthogonal`, and prose alone was not guarding that.

    The existing control (every mapped method exists in `interp_engine.STEER_METHODS`) does NOT protect
    this boundary: `orthogonal` IS in STEER_METHODS, so a future `project -> orthogonal` entry would
    pass it. The names resemble each other and the operations discard OPPOSITE halves of the vector
    space, which makes it the substitution most likely to be made and the most damaging to make.
    """

    @staticmethod
    def _fixture():
        import torch

        torch.manual_seed(0)
        h = torch.randn(1, 4, 8)
        v = torch.randn(8)
        return h, v, v / v.norm()

    def test_orthogonal_can_never_produce_project_onto(self):
        """No coeff makes `orthogonal` equal `h -> Ph`, because it always retains `(I-P)h`."""
        import torch
        from interp_engine.steer import OrthogonalProjector

        h, v, vhat = self._fixture()
        project_onto = (h * vhat).sum(-1, keepdim=True) * vhat
        for coeff in (-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 10.0):
            got = h + OrthogonalProjector(v).delta(h, coeff)
            assert not torch.allclose(got, project_onto, atol=1e-6), (
                f"orthogonal(coeff={coeff}) produced project-onto; the complementary-projection "
                "argument for refusing `project` no longer holds and the mapping must be revisited"
            )

    def test_orthogonal_at_zero_is_reject_not_project(self):
        """The nearest miss, pinned: coeff=0 gives the EXACT COMPLEMENT of what `project` wants.

        This is the jlens `reject` primitive, and having it one argument away from `project` is why
        the refusal needs a test rather than a comment.
        """
        import torch
        from interp_engine.steer import OrthogonalProjector

        h, v, vhat = self._fixture()
        parallel = (h * vhat).sum(-1, keepdim=True) * vhat
        assert torch.allclose(h + OrthogonalProjector(v).delta(h, 0.0), h - parallel, atol=1e-6)

    def test_project_is_still_refused(self):
        from interp_engine_adapter.adapter import UnmappableHookError, _steer_method_for

        with pytest.raises(UnmappableHookError):
            _steer_method_for("project")


class TestSpanOperandsAreRefused:
    """A `(k, d_model)` operand must be refused HERE, because the engine does not reliably refuse it.

    interp-engine's projector is rank-one. With `orthogonal`/`projection_cap` a span operand raises, but
    with a tensor-size message that names neither the span nor the projector. With `additive` it does
    not raise at all when `k` equals the sequence length: it broadcasts positionally and returns a
    plausible result that is not a span operation.
    """

    def test_a_span_operand_is_refused_with_the_reason(self):
        import torch

        from interp_engine_adapter.adapter import UnmappableHookError, _require_expressible_operand

        with pytest.raises(UnmappableHookError, match="rank-one"):
            _require_expressible_operand(torch.randn(2, 8), "blocks.5.mlp.hook_in")

    def test_a_single_direction_passes_through(self):
        import torch

        from interp_engine_adapter.adapter import _require_expressible_operand

        v = torch.randn(8)
        assert _require_expressible_operand(v, "blocks.5.mlp.hook_in") is v

    def test_the_silent_window_is_data_dependent_not_narrow(self):
        """Positive control, and the reason a guard beats a test here.

        Whether a span operand is accepted silently depends on the PROMPT LENGTH at the call, not on
        the code: silent at `k == seq_len`, loud otherwise. A test that fixes one prompt length can
        only establish that THAT length is not the bad one. This asserts the shape of the window
        rather than one point in it, so if interp-engine ever validates span operands properly the
        guard's justification is revisited rather than left asserting a hazard that is gone.
        """
        import torch
        from interp_engine import SteerSpec
        from interp_engine.steer import steer_delta

        d = 8

        def broadcast_compatible(k: int, seq: int) -> bool:
            return k == 1 or seq == 1 or k == seq

        for seq in range(1, 7):
            for k in range(1, 7):
                h = torch.randn(1, seq, d)
                V = torch.randn(k, d)
                try:
                    h + steer_delta(SteerSpec(vector=V, layer=0, coeff=1.0, method="additive"), h, V)
                    silent = True
                except Exception:
                    silent = False
                assert silent == broadcast_compatible(k, seq), (
                    f"the silent set is no longer torch's broadcast-compatible set at k={k}, seq={seq}; "
                    "the guard's justification should be revisited rather than left asserting a rule "
                    "that has changed"
                )

    def test_the_worst_case_replaces_the_sequence_axis(self):
        """seq_len == 1 does not merely corrupt values, it changes the prompt's length.

        A 1-token prompt steered with a k-row span yields a k-POSITION activation, which flows downstream as though the
        prompt had k tokens. seq_len == 1 is every step of incremental decoding, so the case most likely to be hit in
        generation is the one that fails worst.
        """
        import torch
        from interp_engine import SteerSpec
        from interp_engine.steer import steer_delta

        h = torch.randn(1, 1, 8)
        V = torch.randn(3, 8)
        out = h + steer_delta(SteerSpec(vector=V, layer=0, coeff=1.0, method="additive"), h, V)
        assert out.shape[1] == 3 and h.shape[1] == 1, "the sequence-axis replacement no longer occurs"


class TestAsyncCaptureSeam:
    """The notebook path exists and passes its arguments in the engine's order."""

    def test_capture_async_is_a_coroutine_function(self):
        import inspect

        from interp_engine_adapter.adapter import _Seam

        assert inspect.iscoroutinefunction(_Seam.capture_async)

    def test_argument_order_matches_the_engine(self):
        """Pin the engine's parameter order, because a future reorder would be silent at our call site.

        The sync and async entry points agree today: `run_with_cache(model, tokens, points)` and
        `model.capture(prompt_token_ids, points)` line up position for position once `self` is the
        model, the differing parameter NAME notwithstanding. The hazard is not a present disagreement,
        it is that both arguments are positional and interchangeable to the interpreter, so if a future
        engine release reordered them our call would keep working and capture at addresses derived from
        token ids. Pinned here so that surfaces as a test failure rather than as a wrong tensor.
        """
        import inspect

        from interp_engine import EagerModel

        params = list(inspect.signature(EagerModel.capture).parameters)
        assert params[:3] == ["self", "prompt_token_ids", "points"], params


class TestPipInstallSurface:
    """The pip path, which is the ONLY path `pytest` can see. The hub path is verified separately.

    A hub consumer never imports this package: the loader executes `adapter.py` by path under a
    revision-scoped synthetic name. A pip consumer does, and re-exporting from `__init__` is what keeps
    `from interp_engine_adapter import InterpEngineAdapter` working across the package restructure. That
    is a real consumer contract, so it gets a test rather than an assumption.
    """

    def test_the_public_surface_is_reachable_from_the_package(self):
        import interp_engine_adapter as pkg

        for name in ("ADAPTER_NAME", "InterpEngineAdapter", "InterpEngineConfig", "InterpEngineModule"):
            assert hasattr(pkg, name), f"{name} is not re-exported for pip consumers"

    def test_the_package_re_export_is_the_same_object_as_the_module(self):
        """Not merely present: the SAME class. Two definitions would make an isinstance check fail in a
        way that reads as a type error rather than as a packaging fault."""
        import interp_engine_adapter as pkg
        import interp_engine_adapter.adapter as mod

        assert pkg.InterpEngineAdapter is mod.InterpEngineAdapter

    def test_the_manifest_ships_as_package_data(self):
        """`it_component.yaml` must resolve BESIDE the installed module, not merely in the source tree.

        Shipping it via `data-files` instead lands it at the install root, where components overwrite each other -- so
        this asserts the location, not just the existence.
        """
        from pathlib import Path

        import interp_engine_adapter.adapter as mod

        manifest = Path(mod.__file__).parent / "it_component.yaml"
        assert manifest.is_file(), f"manifest not installed beside the module: {manifest}"
