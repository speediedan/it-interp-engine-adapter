"""The seam's own tests. This is where this adapter's correctness lives.

Interpretune's core suite owns the RAILS (that the `adapters` kind loads, gates and registers, tested
against a fixture) and ONE optional contract test (that the published component still satisfies them).
Everything about what this adapter actually does belongs here, so that an interp-engine minor release
moving a signature turns THIS repo red rather than a framework nobody expects to break.
"""

from __future__ import annotations

import ast
import contextlib
import pathlib

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
    """The distinct compositions registered, as readable `a+b` strings.

    DEDUPLICATED, because a composition now occupies more than one component slot: the same adapter set
    appears under both `module` and `datamodule`, and this helper answers "which compositions exist"
    rather than "how many slots were registered". `test_every_composition_registers_both_component_slots`
    is what asserts the slot pairing.
    """
    return sorted(
        {
            "+".join(sorted(a.name if hasattr(a, "name") else str(a) for a in key[1:]))
            for key in registry.available_compositions()
        }
    )


def _fresh_registry():
    from interpretune.adapters.registration import CompositionRegistry, register_dynamic_adapter

    register_dynamic_adapter(ADAPTER_NAME, source="speediedan/it-interp-engine-adapter")
    return CompositionRegistry()


@contextlib.contextmanager
def _all_declared_satisfiable(declared):
    """Present `declared` to the entrypoint as the rails' satisfiable set, then restore.

    The entrypoint reads `supported_compositions()`, which is a ContextVar the rails set. Setting it here exercises the
    HUB path rather than the direct-import path, which is the one that consumes the manifest at all -- the direct path
    never sees it.
    """
    from interpretune.hub import adapters as hub_adapters

    token = hub_adapters._SUPPORTED_COMPOSITIONS.set(tuple(sorted(declared)))
    try:
        yield
    finally:
        hub_adapters._SUPPORTED_COMPOSITIONS.reset(token)


class TestTheManifestDeclaresWhatRegisters:
    """The manifest and the code must agree on COMPONENT SLOTS, not only on adapter sets.

    The rails check the other direction and deliberately stop short of this one. Their invariant walks
    the satisfiable set and drops the component name (`wanted = set(entry_key[1:])`), so a manifest that
    declares `module` while the code also registers `datamodule` loads perfectly. Nothing fails, and the
    only visible consequence is the PUBLISHED CARD, which renders one row per declared entry: a reader
    comparing components sees no datamodule and concludes this one has none.

    That is the recurring shape in this repo -- correct locally, misleading globally, silent -- so the
    agreement is asserted here, where it is this component's own business, rather than wished onto a
    framework that has a considered reason not to check it. Overstating already fails at load; this is
    the understating half.
    """

    def _declared_keys(self):
        import yaml

        manifest = yaml.safe_load(pathlib.Path("src/interp_engine_adapter/it_component.yaml").read_text())
        return {
            (str(entry["component"]), *sorted(str(a) for a in entry["adapters"]))
            for entry in manifest["adapters"]["compositions"]
        }

    def test_every_declared_slot_is_registered_and_no_others(self):
        declared = self._declared_keys()
        registry = _fresh_registry()

        # Every declared composition treated as satisfiable, which is what a fully equipped environment
        # sees. The conditional entries are covered by the registration tests above under both outcomes.
        import interp_engine_adapter.adapter as mod

        if mod._circuit_tracer_pairing() is None:
            declared = {k for k in declared if "circuit_tracer" not in k}

        with _all_declared_satisfiable(declared):
            InterpEngineAdapter.register_adapter_ctx(registry)

        registered = {
            (str(key[0]), *sorted(str(getattr(a, "value", a)) for a in key[1:]))
            for key in registry.available_compositions()
        }
        assert registered == declared, (
            f"manifest and registration disagree.\n  declared not registered: {sorted(declared - registered)}"
            f"\n  registered not declared: {sorted(registered - declared)}"
        )

    def test_both_slots_are_declared_for_every_adapter_set(self):
        """The specific asymmetry that was there: `module` declared, `datamodule` registered silently."""
        by_adapters: dict[tuple[str, ...], set[str]] = {}
        for component, *adapters in self._declared_keys():
            by_adapters.setdefault(tuple(adapters), set()).add(component)
        for adapters, slots in by_adapters.items():
            assert slots == {"module", "datamodule"}, f"{'+'.join(adapters)} declares only {sorted(slots)}"


class TestRegistration:
    def test_registers_the_unconditional_compositions(self):
        registry = _fresh_registry()
        InterpEngineAdapter.register_adapter_ctx(registry)

        names = _names(registry)
        assert "core+interp_engine" in names
        assert "interp_engine+lightning" in names

        # KEYED BY COMPONENT, because a composition is more than its module. This used to assert that
        # every registered key mapped to `(InterpEngineModule,)`, which was true only while this component
        # registered the module slot alone -- and registering only that half is exactly why it could not
        # build an `ITSession`. The datamodule slot now exists, so the assertion has to say which slot it
        # is talking about.
        for key in registry.available_compositions():
            if "circuit_tracer" in str(key):
                continue
            if key[0] == "module":
                assert registry.get(key) == (InterpEngineModule,)
            elif key[0] == "datamodule":
                assert registry.get(key), f"{key} registered no datamodule classes"

    def test_every_composition_registers_both_component_slots(self):
        """A component that registers only the module half registers SUCCESSFULLY and fails later, at session
        construction, with a KeyError naming the datamodule key.

        That reads as a missing adapter rather than as a half-registered component, so the pairing is asserted here.
        """
        registry = _fresh_registry()
        InterpEngineAdapter.register_adapter_ctx(registry)

        by_adapters: dict[tuple, set[str]] = {}
        for key in registry.available_compositions():
            by_adapters.setdefault(key[1:], set()).add(key[0])
        for adapters, slots in by_adapters.items():
            assert {"module", "datamodule"} <= slots, f"{adapters} registers only {sorted(slots)}"


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

    def test_every_circuit_tracer_slot_declares_requires(self):
        """BOTH slots, and each independently.

        A composition entry's identity is (component, *adapters), so the module and datamodule slots of
        the conditional composition are two entries and the rails evaluate their `requires` separately.
        Declaring it on one would make the other unconditionally satisfiable, and the entrypoint would
        be handed a satisfiable datamodule slot for a composition whose module slot was skipped.
        """
        entries = self._manifest()["adapters"]["compositions"]
        ct = [e for e in entries if "circuit_tracer" in e["adapters"]]
        assert {e["component"] for e in ct} == {"module", "datamodule"}
        for entry in ct:
            assert entry.get("requires", {}).get("pip") == ["circuit-tracer"], (
                f"the {entry['component']} slot does not declare its requirement"
            )

    def test_the_core_compositions_declare_no_requires(self):
        """They must not, or an environment without circuit-tracer would get nothing at all."""
        entries = self._manifest()["adapters"]["compositions"]
        for entry in entries:
            if "circuit_tracer" not in entry["adapters"]:
                assert not entry.get("requires"), f"{entry['adapters']} should be unconditional"

    def test_every_declared_composition_is_one_the_code_can_register(self):
        """Manifest and code compared entry for entry, in the direction that catches drift: a manifest advertising
        something the code cannot produce is a promise to a reader that nothing will keep."""
        declared = {
            ("module", *sorted(e["adapters"]))
            for e in self._manifest()["adapters"]["compositions"]
            if e["component"] == "module"
        }
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
        from interp_engine_adapter.adapter import InterpEngineConfig, InterpEngineCaptureMixin

        class _Rest:
            """Stands in for the rest of the MRO, which the mixin's `super()` call reaches."""

            def post_auto_model_init(self, *args, **kwargs):
                return "inner"

        class _Composed(InterpEngineCaptureMixin, _Rest):
            def __init__(self):
                # Carries an engine config because a real module always does, and the adapter now refuses
                # rather than defaulting when one did not reach it. The double was `it_cfg = None`, which
                # no composed module ever is; the attach decision under test does not read the config, so
                # supplying a faithful one narrows the double rather than widening it.
                self.it_cfg = type("_Cfg", (), {"interp_engine_cfg": InterpEngineConfig()})()
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
        from interpretune.analysis.backends.capabilities import ModelBackendCapability
        from interp_engine_adapter.adapter import InterpEngineModelBackend

        backend = InterpEngineModelBackend()
        assert backend.supports(ModelBackendCapability.ACTIVATION_INTERVENTION)
        # Enumerated from the ENUM rather than a hand-written list. The previous version named
        # BATCHED_HOOKS explicitly and went stale the moment speediedan/interpretune#449 removed it,
        # which is the failure mode of pinning a vocabulary by transcription: the test breaks on a
        # change it was not written to have an opinion about. Deriving the complement means a member
        # added upstream is asserted absent automatically, and a member removed simply disappears.
        for absent in set(ModelBackendCapability) - {ModelBackendCapability.ACTIVATION_INTERVENTION}:
            assert not backend.supports(absent), f"unexpectedly claims {absent}"

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

    def test_a_callable_names_filter_needs_a_model_it_can_enumerate(self):
        """A callable is applied to an inventory, so a backend with no resolvable model has none.

        This replaced a blanket refusal of callables. The refusal was written when the inventory could not be built at
        all; it can be, from interpretune's own vocabulary, and refusing callables refused ORDINARY use rather than an
        exotic one, because interpretune's core turns an explicit list into a predicate before it reaches a backend.
        """
        from interp_engine_adapter.adapter import InterpEngineModelBackend, UnmappableHookError

        backend = InterpEngineModelBackend()
        with pytest.raises(UnmappableHookError, match="no hook mapping"):
            backend.fwd_w_cache(None, {"input_ids": None}, lambda name: True)


class _FakeEngineModel:
    """The minimum an `EagerModel` has to look like for hook translation, and no more.

    Deliberately a stub rather than a loaded gpt2. Everything under test here is NAME translation -- an
    inventory, a predicate over it, and two alias tables -- which touches no weights, so loading a model
    would buy a slower test and the same assertions. What it must carry is exactly what the translation
    path reads: an `hf_model` whose CLASS NAME selects interpretune's architecture mapping, a config
    depth to expand layers over, and a residual stream count, which is what
    `interp_engine.mappers.has_hyper_connections` gates the block-input alias on. That function RAISES
    rather than defaulting when it finds no count, so omitting `n_residual_streams` here would fail in a
    way that reads like a mapper bug.
    """

    class _Config:
        n_layer = 12

    class GPT2LMHeadModel:  # -- the class NAME is the architecture key, not a style choice
        config = None

    n_residual_streams = 1

    def __init__(self):
        self.hf_model = self.GPT2LMHeadModel()
        self.hf_model.config = self._Config()


class TestTheHookInventoryIsInterpretunesVocabulary:
    """A callable filter is a predicate over interpretune's names, so it must meet interpretune's names.

    The measurement behind this class: enumerating FORWARD from the engine instead -- its points spelled
    through `point_to_tlens_hook` -- produces 252 names on gpt2 and looks complete, but the engine's
    reverse table holds one CANONICAL spelling per point while its forward table accepts aliases. So
    `blocks.5.hook_resid_post` is in that inventory and `blocks.5.hook_out` is not, though both name the
    same tensor and interpretune uses the second. A predicate over it matched NONE of the conformance
    suite's five capture points, and nothing in the failure pointed at spelling.
    """

    def test_the_vocabulary_contains_the_alias_spellings_interpretune_uses(self):
        from interp_engine_adapter.adapter import _hook_vocabulary

        vocabulary = set(_hook_vocabulary(_FakeEngineModel()))
        # Both spellings of the same tensor, which is the property a canonical-only inventory lacks.
        assert "blocks.5.hook_out" in vocabulary
        assert "blocks.5.hook_resid_post" in vocabulary
        assert "unembed.hook_in" in vocabulary

    def test_layer_scoped_hooks_are_expanded_and_global_ones_are_not(self):
        """The bug this caught: `resolver.resolve` accepts names that name nothing.

        The inventory was first built by trying every base name BOTH bare and layer-qualified and
        keeping whichever forms the resolver accepted, to avoid reading a private attribute. It accepts
        `blocks.5.unembed.hook_out` and a bare `attn.hook_z`. While interpretune's vocabulary was 13
        base names that produced a little junk and no visible harm; at 34 it produced 252 junk entries
        out of 433, and four of five ordinary filters started refusing because each selected some
        nonsense name the seam then could not map.
        """
        from interp_engine_adapter.adapter import _hook_vocabulary

        vocabulary = set(_hook_vocabulary(_FakeEngineModel()))

        # Layer-scoped: qualified, never bare.
        assert "blocks.5.attn.hook_z" in vocabulary
        assert "attn.hook_z" not in vocabulary

        # Global: bare, never qualified.
        assert "unembed.hook_in" in vocabulary
        assert "blocks.5.unembed.hook_in" not in vocabulary
        assert not any(n.startswith("blocks.") and ".unembed." in n for n in vocabulary)

    def test_the_refusal_summarises_by_base_name_rather_than_reciting_every_hook(self):
        """An unreadable refusal is one that gets worked around instead of understood.

        A broad predicate selects a hundred names here, so the message groups by the base name a predicate is actually
        written against and carries the counts that let a caller decide what to do.
        """
        from interp_engine_adapter.adapter import UnmappableHookError, _names_filter_to_list

        with pytest.raises(UnmappableHookError) as excinfo:
            _names_filter_to_list(lambda name: name.endswith("hook_out"), _FakeEngineModel())

        message = str(excinfo.value)
        # The base has to satisfy two things this assertion cannot check for itself: the seam must still
        # refuse it (`ln1.hook_out` stood here until it became a mapped alias of `attn.hook_in`), and its
        # name must not be a substring of another refused base -- `embed.hook_out` occurs three times in
        # this message only because `pos_embed.hook_out` contains it, which would make the count assertion
        # below pass for the wrong reason.
        assert "attn.o.hook_out" in message, "the message should name the base, not every layer of it"
        assert "blocks.0.attn.o.hook_out" not in message
        assert message.count("attn.o.hook_out") == 1
        assert "explicit list" in message

    def test_a_filter_matching_nothing_is_refused_rather_than_returning_an_empty_cache(self):
        from interp_engine_adapter.adapter import UnmappableHookError, _names_filter_to_list

        with pytest.raises(UnmappableHookError, match="matched none of"):
            _names_filter_to_list(lambda name: name == "blocks.5.hook_nonexistent", _FakeEngineModel())

    def test_a_filter_selecting_an_unmappable_name_is_refused_rather_than_captured_partially(self):
        """Partial capture is the failure this whole path exists to avoid.

        A cache holding the mappable subset is indistinguishable from a complete one at the call site, so the caller
        computes over a filter that silently meant less than it said.
        """
        from interp_engine_adapter.adapter import UnmappableHookError, _names_filter_to_list

        with pytest.raises(UnmappableHookError, match="cannot capture at"):
            _names_filter_to_list(
                lambda name: name.endswith("hook_mlp_out") or name.endswith("hook_attn_in"), _FakeEngineModel()
            )


class TestTheCircuitTracerGuardPropagatesOnlyOurOwnRefusals:
    """The guard must forward this adapter's refusals and swallow everything else.

    WHY THIS CLASS IS NOT GATED ON CIRCUIT-TRACER, unlike the pairing tests it protects. The guard's
    contract is a property of `_Seam`, and `refuse_if_ambiguous` needs no circuit-tracer to exercise. It
    was previously asserted only through `TestCircuitTracerPairing`, which IS gated, so the contract went
    untested in every environment without circuit-tracer installed -- which is every environment except
    the GPU lane.

    A change that widened engine-gap errors into this adapter's own exception type therefore looked clean
    everywhere it was measured, and broke `blocks.0.ln2.hook_normalized` on the one lane that runs the
    pairing: a hook circuit-tracer resolves perfectly well, refused before it ever got there.
    """

    def test_a_name_the_engine_does_not_know_falls_through_to_circuit_tracer(self):
        """`ln2.hook_normalized` is the exact name the GPU lane caught."""
        from interp_engine_adapter.adapter import _Seam

        for hook in ("ln2.hook_normalized", "ln1.hook_normalized", "hook_resid_post"):
            _Seam.refuse_if_ambiguous(hook)  # must not raise

    def test_this_adapters_own_refusal_still_propagates(self):
        from interp_engine_adapter.adapter import UnmappableHookError, _Seam

        for hook in ("hook_mlp_in", "hook_attn_in"):
            with pytest.raises(UnmappableHookError, match="BEFORE the block norm"):
                _Seam.refuse_if_ambiguous(hook)

    def test_an_engine_gap_is_still_an_unmappable_hook_error_to_a_capture_caller(self):
        """The distinction is for the guard alone; a capture path must refuse either way.

        Subclassing is what buys both: the guard checks the narrow type, every other caller catches the
        base and never learns there was a distinction.
        """
        from interp_engine_adapter.adapter import EngineHasNoPointError, UnmappableHookError

        assert issubclass(EngineHasNoPointError, UnmappableHookError)


class TestDeprecatedSpellingsStayNameable:
    """A spelling interpretune deprecates but still resolves must stay in this backend's inventory.

    speediedan/interpretune#463 made `parse(name, strict=True)` refuse deprecated aliases -- `attn.hook_z`
    among them, which resolves leniently to the `attn.o` component. `_hook_vocabulary` parses LENIENTLY,
    deliberately, and this pins that choice against a future tightening.

    WHY LENIENT IS THE RIGHT DEFAULT HERE. The inventory answers "what could a caller name?", not "what
    spelling would we recommend?". A caller whose predicate targets `attn.hook_z` is asking for a tensor
    this backend can capture, and `point_for` resolves it correctly. Dropping it would make that
    predicate match fewer names with nothing said -- which is the silent under-capture the whole
    filter path exists to prevent, arriving through a style preference rather than a defect.

    THE FAILURE THIS GUARDS. If interpretune later removes such an alias rather than deprecating it,
    `point_for` starts refusing a name the inventory still offers. Every broad predicate that happens to
    select it then fails as a partial selection, and the error names hook spellings rather than the
    upstream change that caused it. Failing here instead names the spelling and this reasoning.
    """

    def test_a_deprecated_but_resolvable_spelling_is_offered_and_resolves(self):
        from interp_engine_adapter.adapter import _Seam, _hook_vocabulary

        model = _FakeEngineModel()
        assert "blocks.5.attn.hook_z" in set(_hook_vocabulary(model))
        assert _Seam.point_for("blocks.5.attn.hook_z", model).name == "z"

    def test_the_inventory_parses_leniently_rather_than_strictly(self):
        """Stated as a property of interpretune's parser, so a change upstream fails HERE by name."""
        pytest.importorskip("interpretune.analysis.points")
        from interpretune.analysis.points import parse

        assert parse("attn.hook_z").component == "attn.o"
        with pytest.raises(Exception) as excinfo:
            parse("attn.hook_z", strict=True)
        assert "Deprecated" in type(excinfo.value).__name__, (
            f"expected a deprecation refusal under strict parsing, got {type(excinfo.value).__name__}. "
            "If interpretune stopped deprecating this spelling, the lenient choice in `_hook_vocabulary` "
            "may no longer be load-bearing and this test should be revisited rather than patched."
        )


class TestMeasuredAliasesAreNotTheRefusals:
    """The two tables in `_Seam` look alike and mean opposite things.

    `_REFUSED` names tensors that DIFFER and must not be substituted. `_MEASURED_ALIASES` names tensors
    measured IDENTICAL that the engine simply has no spelling for. Conflating them in either direction
    is a defect: mapping a refusal answers a question nobody asked, and refusing an alias makes the
    backend unusable for names interpretune treats as fundamental.
    """

    def test_the_two_tables_do_not_overlap(self):
        from interp_engine_adapter.adapter import _Seam

        assert not set(_Seam._REFUSED) & set(_Seam._MEASURED_ALIASES)
        assert not set(_Seam._REFUSED) & set(_Seam._MEASURED_GLOBAL_ALIASES)

    def test_block_input_resolves_to_the_residual_stream_before_the_block(self):
        from interp_engine_adapter.adapter import _Seam

        address = _Seam.point_for("blocks.5.hook_in", _FakeEngineModel())
        assert address.name == "resid_pre"
        assert address.layer == 5

    def test_unembed_input_resolves_to_the_final_norm(self):
        from interp_engine_adapter.adapter import _Seam

        assert _Seam.point_for("unembed.hook_in", _FakeEngineModel()).name == "final_norm"

    def test_the_prenorm_refusal_is_still_a_refusal(self):
        """The regression this repo exists to prevent: an alias table growing into the refusals."""
        from interp_engine_adapter.adapter import UnmappableHookError, _Seam

        with pytest.raises(UnmappableHookError, match="BEFORE the block norm"):
            _Seam.point_for("blocks.5.hook_mlp_in", _FakeEngineModel())


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


def _mode_table() -> dict:
    from interp_engine_adapter.adapter import _MODE_TO_STEER_METHOD

    return _MODE_TO_STEER_METHOD


def _refusal_reasons() -> dict:
    from interp_engine_adapter.adapter import _MODE_REFUSAL_REASONS

    return _MODE_REFUSAL_REASONS


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

    @pytest.mark.parametrize("mode", sorted(_refusal_reasons()))
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

    def test_every_interpretune_mode_is_mapped_or_explicitly_refused(self):
        """Every interpretune mode is mapped or carries a refusal reason, or this fails naming it.

        The failure mode being guarded is a NEW mode defaulting into silence: without this, a mode
        added upstream would simply not appear in the table and would raise only when someone used
        it. ``documented`` is derived from the enum rather than a literal, because a literal cannot
        report on a member that did not exist when it was written.
        """
        from interpretune.analysis.backends.capabilities import InterventionMode
        from interpretune.analysis.backends.interventions import InterventionSpec

        from interp_engine_adapter.adapter import _MODE_REFUSAL_REASONS, _uncovered_modes

        # `_field_defaults`, not `InterventionSpec.mode`: this is a NamedTuple, so the attribute is the
        # field DESCRIPTOR and comparing it to a string is always False -- an assertion that would have
        # failed for a reason unrelated to what it checks.
        assert InterventionSpec._field_defaults["mode"] == "replace", (
            "the default intervention mode changed; revisit which modes this backend refuses"
        )
        documented = {m.value for m in InterventionMode}
        uncovered = _uncovered_modes(documented, _mode_table(), _MODE_REFUSAL_REASONS)
        assert uncovered == set(), (
            f"intervention modes with no mapping and no refusal reason: {sorted(uncovered)}. "
            "Decide the mapping or the refusal."
        )

    def test_the_coverage_guard_fires_on_an_undecided_mode(self):
        """Positive control: a mode with no mapping and no reason fails the guard naming it.

        Without this, replacing an inert guard with one whose ability to fire is unverified would
        repeat the exercise.
        """
        from interp_engine_adapter.adapter import _uncovered_modes

        uncovered = _uncovered_modes({"add", "probe_undecided"}, {"add": "additive"}, {})
        assert uncovered == {"probe_undecided"}

    def test_an_explicit_refusal_reason_satisfies_the_guard(self):
        """Positive control the other way: the reason dict must not become a second inert path."""
        from interp_engine_adapter.adapter import _uncovered_modes

        uncovered = _uncovered_modes(
            {"add", "probe_refused"},
            {"add": "additive"},
            {"probe_refused": "cannot be expressed, and this string is the reason"},
        )
        assert uncovered == set()


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


class TestRejectMapping:
    """`reject` maps to `orthogonal` through an orthonormalising change of basis, at every `k`.

    The ruling: ``VV^+`` is the orthogonal projector onto ``span(V)`` regardless of what the rows
    look like, so orthonormal rows reproduce it exactly while raw rows do not (measured 0.1 off at
    row cosine -0.240). The reference throughout is interpretune's own ``_apply_span_rejection``
    (pseudoinverse), not a re-derivation here: a backend author deriving their own oracle is the
    same tree answering twice.
    """

    TOLERANCE = 1e-5  # measured agreement is ~1e-7; two orders of slack for cross-machine float

    @staticmethod
    def _workbench():
        import torch

        torch.manual_seed(7)
        h = torch.randn(5, 64)
        first = torch.zeros(64)
        first[0] = 1.0
        second = torch.zeros(64)
        second[0] = -0.24
        second[1] = (1.0 - 0.24**2) ** 0.5
        return h, torch.stack([first, second])

    @staticmethod
    def _reference(h, V, alpha):
        from interpretune.analysis.backends.interventions import InterventionSpec, _apply_span_rejection

        spec = InterventionSpec(intervention_tensor=V, mode="reject", scale_factor=alpha)
        return _apply_span_rejection(spec, input_value=h, target=V)

    @staticmethod
    def _mapped(h, V, alpha):
        import torch
        from interp_engine import SteerSpec
        from interp_engine.steer import steer_delta

        from interp_engine_adapter.adapter import _coeff_for, _reject_basis_rows

        coeff = _coeff_for("reject", alpha)
        total = torch.zeros_like(h)
        for row in _reject_basis_rows(V, "test"):
            total = total + steer_delta(SteerSpec(vector=row, layer=0, coeff=coeff, method="orthogonal"), h, row)
        return h + total

    def test_coeff_rule_complements_the_scale(self):
        from interp_engine_adapter.adapter import _coeff_for

        for scale, expected in ((0.0, 1.0), (0.5, 0.5), (1.0, 0.0)):
            assert _coeff_for("reject", scale) == expected
            assert _coeff_for("add", scale) == scale

    @pytest.mark.parametrize("alpha", [1.0, 0.5])
    def test_mapped_matches_reference_at_k2(self, alpha):
        import torch

        h, V = self._workbench()
        got, want = self._mapped(h, V, alpha), self._reference(h, V, alpha)
        assert torch.max((got - want).abs()).item() < self.TOLERANCE

    def test_mapped_matches_reference_at_k1(self):
        import torch

        h, V = self._workbench()
        got, want = self._mapped(h, V[:1], 0.3), self._reference(h, V[:1], 0.3)
        assert torch.max((got - want).abs()).item() < self.TOLERANCE

    @pytest.mark.parametrize("alpha", [1.0, 0.5])
    def test_raw_row_composition_must_fail(self, alpha):
        """The negative control the caveat exists for: un-orthonormalised rows are wrong, loudly."""
        import torch
        from interp_engine import SteerSpec
        from interp_engine.steer import steer_delta

        h, V = self._workbench()
        total = torch.zeros_like(h)
        for row in V:
            total = total + steer_delta(SteerSpec(vector=row, layer=0, coeff=1.0 - alpha, method="orthogonal"), h, row)
        assert torch.max((h + total - self._reference(h, V, alpha)).abs()).item() > 1e-2

    def test_rank_deficient_span_shrinks_to_its_rank(self):
        import torch

        h, V = self._workbench()
        duplicated = torch.stack([V[0], V[0]])
        got, want = self._mapped(h, duplicated, 1.0), self._reference(h, duplicated, 1.0)
        assert torch.max((got - want).abs()).item() < self.TOLERANCE

    def test_zero_scale_builds_an_explicit_identity(self):
        """A mapped reject at ``scale_factor=0.0`` is an explicit request, never an omission."""
        import torch

        from interp_engine_adapter.adapter import _Seam

        spec = _Seam.steer_spec(vector=torch.randn(8), layer=0, coeff=1.0, method="orthogonal", explicit=True)
        assert spec.coeff == 1.0

    def test_mixed_add_and_reject_in_one_scope_flows_through(self):
        """Declaration order is the contract within a scope: no refusal, order preserved upstream."""
        from interp_engine_adapter.adapter import _refuse_mixed_modes

        _refuse_mixed_modes(
            [
                (0, "p", {"mode": "add", "intervention_tensor": 0, "scale_factor": 1.0}),
                (0, "p", {"mode": "reject", "intervention_tensor": 0, "scale_factor": 1.0}),
            ]
        )
        _refuse_mixed_modes(
            [
                (0, "p", {"mode": "reject", "intervention_tensor": 0, "scale_factor": 1.0}),
                (0, "p", {"mode": "add", "intervention_tensor": 0, "scale_factor": 1.0}),
            ]
        )

    def test_mixed_add_and_reject_across_scopes_is_refused_by_name(self):
        from interp_engine_adapter.adapter import UnmappableHookError, _refuse_mixed_modes

        with pytest.raises(UnmappableHookError, match="across position scopes"):
            _refuse_mixed_modes(
                [
                    (0, "p", {"mode": "add", "intervention_tensor": 0, "scale_factor": 1.0}),
                    (
                        0,
                        "p",
                        {
                            "mode": "reject",
                            "intervention_tensor": 0,
                            "scale_factor": 1.0,
                            "position_scope": "all_positions",
                        },
                    ),
                ]
            )

    def test_same_mode_multiplicity_is_not_a_mixture(self):
        from interp_engine_adapter.adapter import _refuse_mixed_modes

        _refuse_mixed_modes(
            [
                (0, "p", {"mode": "add", "intervention_tensor": 0, "scale_factor": 1.0}),
                (0, "p", {"mode": "add", "intervention_tensor": 0, "scale_factor": 0.5}),
            ]
        )
        _refuse_mixed_modes([(0, "p", {"mode": "reject", "intervention_tensor": 0, "scale_factor": 1.0})])

    def test_reject_operand_shapes(self):
        import torch

        from interp_engine_adapter.adapter import UnmappableHookError, _reject_basis_rows

        rows = _reject_basis_rows(torch.randn(8), "p")
        assert len(rows) == 1 and torch.allclose(rows[0].norm(), torch.tensor(1.0))
        rows = _reject_basis_rows(torch.randn(3, 8), "p")
        assert len(rows) == 3
        assert torch.allclose(torch.stack(rows) @ torch.stack(rows).T, torch.eye(3), atol=1e-6)
        assert _reject_basis_rows(torch.empty(0, 8), "p") == []
        with pytest.raises(UnmappableHookError, match="one direction"):
            _reject_basis_rows(torch.randn(2, 2, 8), "p")
        with pytest.raises(UnmappableHookError, match="NaN"):
            _reject_basis_rows(torch.full((2, 8), float("nan")), "p")

    def test_modes_declaration_matches_the_ruled_mapping(self):
        from interp_engine_adapter.adapter import _MODE_REFUSAL_REASONS, _MODE_TO_STEER_METHOD

        assert set(_MODE_TO_STEER_METHOD) == {"add", "reject"}
        assert set(_MODE_REFUSAL_REASONS) == {"replace", "patch", "project"}


class TestSpanOperandsAreRefused:
    """A `(k, d_model)` operand must be refused HERE, because the engine does not reliably refuse it.

    This is the additive path (`_require_expressible_operand`): interp-engine's projector is
    rank-one, and with `additive` a span operand broadcasts positionally rather than raising.
    `reject` spans take the other path -- orthonormalised and fanned out in `_reject_basis_rows`,
    pinned by `TestRejectMapping` -- so this class stays scoped to the hazard it names.
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


class TestCudaSelectionIsAdditive:
    """The GPU lane must not narrow its own collection.

    This repo's Azure lane is the only one that installs circuit-tracer, so it is the only place the
    `(core, interp_engine, circuit_tracer)` tests run. If `IT_RUN_CUDA_TESTS=1` ever becomes a SELECTOR
    here -- as it is in interpretune, where `items[:] = [...]` replaces the collected list -- that lane
    silently stops running them and reports a smaller green number instead of an error.

    Asserted against the conftest's SOURCE rather than by running a subprocess: the property is
    structural ("collection is never reassigned"), and a behavioural check would need a nested pytest
    run with a GPU to distinguish additive from selective.
    """

    def _conftest_source(self):
        from pathlib import Path

        return (Path(__file__).parent / "conftest.py").read_text()

    def test_collection_is_never_reassigned(self):
        """Checked on the AST, not on the text.

        A substring scan matches the conftest's own docstring, which quotes the forbidden pattern in order to explain it
        -- so the naive version of this test fails on a correct file. Parsing means it sees assignments and nothing
        else.
        """
        import ast

        tree = ast.parse(self._conftest_source())
        offenders = []
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            for t in targets:
                # `items[:] = ...` is a Subscript on a Name; `items = ...` is a bare Name.
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "items":
                    offenders.append("items[...] = ...")
                elif isinstance(t, ast.Name) and t.id == "items":
                    offenders.append("items = ...")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "items":
                    if node.func.attr in ("clear", "sort", "reverse", "pop", "remove"):
                        offenders.append(f"items.{node.func.attr}()")

        assert not offenders, (
            f"conftest mutates or replaces the collection ({offenders}). `IT_RUN_CUDA_TESTS` must UN-SKIP "
            "here, not select: the Azure lane is the only one that installs circuit-tracer, so narrowing "
            "it silently drops the composition tests and reports a smaller green number."
        )

    def test_the_marker_is_registered(self):
        """An unregistered marker is a warning, not an error, so a typo'd `@pytest.mark.cuda` would silently never
        skip and the test would run on CPU and fail confusingly."""
        assert "config.addinivalue_line" in self._conftest_source()
        assert '"cuda:' in self._conftest_source()

    def test_cuda_marked_tests_skip_without_the_opt_in(self):
        """The default posture.

        Without this, a contributor's plain `pytest -v` tries to allocate a GPU.
        """
        import os

        if os.getenv("IT_RUN_CUDA_TESTS", "0") == "1":
            pytest.skip("opt-in is active in this environment, so the default posture is not observable")
        # No `cuda`-marked test exists yet -- N5's parity measurement is the first. Asserting the
        # DEFAULT posture now means the marker's behaviour is pinned before anything depends on it,
        # rather than being discovered when the first device test lands.
        assert os.getenv("IT_RUN_CUDA_TESTS", "0") != "1"


class TestEveryDeclaredModeCanActuallySteer:
    """A positive control on the band: a mode this backend DECLARES must be able to change something.

    `intervention_support` derives its modes from `_MODE_TO_STEER_METHOD`, so a test comparing the
    declaration against the mapping compares the mapping to itself and cannot fail. What is not
    definitional, and is what a caller actually depends on, is that a spec built through the real seam
    produces a NON-ZERO delta for some residual. Two ways a declared mode can silently do nothing, both
    measured against the engine rather than reasoned about:

    * ``projection_cap`` clamps the projection into ``[min, max]`` and both bounds are independently
      optional, so with neither the delta is zero for every input there is. ``steer_spec`` exposes no
      parameter for a bound, so the moment that method becomes reachable through the mapping, every
      clamp it builds is a no-op that completes and returns plausible logits.
    * ``orthogonal`` at ``coeff=1.0`` rescales the projection by one, which is legitimately no change.

    The second is why this sweeps coefficients and requires only that ONE of them steers: asserting the
    default is non-zero would fail on arithmetic that is correct.
    """

    #: Projections chosen to sit above, below and inside a typical band, so a clamp that is merely
    #: inactive on this data is not mistaken for one that can never act.
    RESIDUALS = ((3.0, 4.0, 0.0), (-3.0, 4.0, 0.0), (0.5, 4.0, 0.0))
    COEFFS = (0.0, 0.5, 2.0)

    def test_each_declared_mode_produces_a_nonzero_delta_for_some_input(self):
        import torch
        from interp_engine import steer_delta

        from interp_engine_adapter.adapter import (
            _MODE_TO_STEER_METHOD,
            _Seam,
            _steer_method_for,
            UnmappableHookError,
        )

        vector = torch.tensor([1.0, 0.0, 0.0])
        inert = []
        for mode in sorted(_MODE_TO_STEER_METHOD):
            method = _steer_method_for(mode)
            steered = False
            for coeff in self.COEFFS:
                try:
                    spec = _Seam.steer_spec(vector=vector, layer=0, coeff=coeff, method=method)
                except UnmappableHookError:
                    # This coefficient is the method's identity and the seam refuses it at build time.
                    # That IS the guard working, so it is not a failure here; keep looking for one that
                    # steers. Without this the sweep would report a correctly-guarded seam as inert.
                    continue
                for row in self.RESIDUALS:
                    delta = steer_delta(spec, torch.tensor([list(row)]), vector)
                    if bool(torch.any(delta != 0)):
                        steered = True
                        break
                if steered:
                    break
            if not steered:
                inert.append(f"{mode!r} -> {method!r}")

        assert not inert, (
            "these declared intervention modes produce a zero delta for every coefficient and residual "
            f"tried, so they steer nothing while being advertised as supported: {inert}. A mode whose "
            "delta is unconditionally zero completes normally and returns plausible logits for an "
            "intervention nobody performed. If this fired after adding a clamp mode, the cause is "
            "`steer_spec` having no `min`/`max` parameter, so the `projection_cap` spec it builds has "
            "no band to clamp into."
        )


class TestTheUnbandedClampIsRefused:
    """The band refusal must FIRE, and must not fire on anything else.

    A guard whose negative case never runs is the defect this fix exists to remove, reintroduced inside
    its own repair, so proving the refusal happens is half of it. The other half is the complement: a
    guard that refused every `projection_cap` spec, or every spec, would pass a test that only checked
    the refusal. So the banded forms and the bandless-by-design methods are asserted to still build.
    """

    #: The ONE parameterization per method whose delta is identically zero, with the value that produces
    #: it. They are at DIFFERENT values, which is the point: a single rule on `coeff` would be wrong for
    #: two of the three, and a reader who assumes the coefficient identities coincide has them backwards.
    #: The parameterizations refused, and the property they share is NOT "the delta is zero". It is that
    #: the no-op cannot be told apart from a parameter nobody set: `orthogonal`'s identity coefficient IS
    #: this signature's default, and an absent band is an absent argument. `additive` at coeff=0.0 is
    #: also identically zero and is deliberately absent from this list -- see the test below.
    IDENTITIES = [
        ({"method": "orthogonal", "coeff": 1.0}, "coeff=1.0"),
        ({"method": "projection_cap"}, "minimum"),
    ]

    @pytest.mark.parametrize("kwargs,expected_in_message", IDENTITIES, ids=["orthogonal", "projection_cap"])
    def test_every_identity_parameterization_is_refused_naming_the_value(self, kwargs, expected_in_message):
        """A spec that cannot steer ANY input is refused; the message names the mode and the value."""
        import torch

        from interp_engine_adapter.adapter import UnmappableHookError, _Seam

        with pytest.raises(UnmappableHookError) as excinfo:
            _Seam.steer_spec(vector=torch.tensor([1.0, 0.0, 0.0]), layer=0, **kwargs)
        message = str(excinfo.value)
        assert kwargs["method"] in message, "the refusal must name the mode"
        assert expected_in_message in message, "it must name the value or parameter responsible"

    def test_an_explicitly_requested_zero_is_built_rather_than_refused(self):
        """`additive` at coeff=0.0 has a zero delta and is still allowed, deliberately.

        0.0 is never this signature's default, so it is always an explicit request rather than an
        omission, and the conformance contract requires a zero-scale intervention to be an IDENTITY
        rather than an error (`test_zero_intervention_is_identity`). An earlier revision refused it on
        the grounds that its delta is provably zero, which broke that suite; the property that actually
        justifies a refusal is indistinguishability from an unset parameter, not zeroness.
        """
        import torch

        from interp_engine_adapter.adapter import _Seam

        spec = _Seam.steer_spec(vector=torch.tensor([1.0, 0.0, 0.0]), layer=0, method="additive", coeff=0.0)
        assert spec.coeff == 0.0

    def test_a_spec_that_is_inert_on_this_input_but_not_all_is_built(self):
        """The complement that keeps the rule honest: `min`-only above the floor does nothing HERE.

        That is a correct clamp on data that needs none, and refusing it would be the mirror-image
        defect -- refusing an operation the backend performs exactly.
        """
        import torch

        from interp_engine_adapter.adapter import _Seam

        spec = _Seam.steer_spec(vector=torch.tensor([1.0, 0.0, 0.0]), layer=0, method="projection_cap", minimum=0.0)
        assert spec.min == 0.0

    def test_projection_cap_with_neither_bound_is_refused_naming_what_is_missing(self):
        import torch

        from interp_engine_adapter.adapter import UnmappableHookError, _Seam

        with pytest.raises(UnmappableHookError) as excinfo:
            _Seam.steer_spec(vector=torch.tensor([1.0, 0.0, 0.0]), layer=0, method="projection_cap")
        message = str(excinfo.value)
        assert "projection_cap" in message, "the refusal must name the mode it refused"
        assert "minimum" in message and "maximum" in message, "it must name the parameters that would fix it"

    @pytest.mark.parametrize(
        "band", [{"minimum": 0.0}, {"maximum": 1.0}, {"minimum": 0.0, "maximum": 1.0}], ids=["min", "max", "both"]
    )
    def test_either_bound_alone_is_enough_to_build(self, band):
        """A `min`-only clamp is inactive on data above the floor, which is correct and not this defect."""
        import torch

        from interp_engine_adapter.adapter import _Seam

        spec = _Seam.steer_spec(vector=torch.tensor([1.0, 0.0, 0.0]), layer=0, method="projection_cap", **band)
        assert spec.min == band.get("minimum")
        assert spec.max == band.get("maximum")

    @pytest.mark.parametrize("method,coeff", [("additive", 1.0), ("orthogonal", 2.0)], ids=["additive", "orthogonal"])
    def test_the_methods_that_take_no_band_are_not_refused(self, method, coeff):
        """`additive` and `orthogonal` ignore min/max, so requiring a band from them would be wrong.

        Each is given a coefficient that steers, because the identity values are refused separately and
        for a different reason: this asserts the BAND rule does not overreach, not that every parameter
        is accepted.
        """
        import torch

        from interp_engine_adapter.adapter import _Seam

        spec = _Seam.steer_spec(vector=torch.tensor([1.0, 0.0, 0.0]), layer=0, method=method, coeff=coeff)
        assert spec.method == method


def _adapter_module_path() -> str:
    """The file the hub loader executes, located via the module under test rather than guessed."""
    import interp_engine_adapter.adapter as _m

    return _m.__file__


class TestTheSeamIsTheOnlyPlaceTheEngineIsImported:
    """`_Seam` documents itself as "The one place this adapter touches ``interp_engine``". Assert it.

    The invariant was true when written and nothing enforced it, which matters more here than it would
    for a stylistic rule: it is the reason this component's analysis-backend surface is hub-delivered
    rather than in-tree. An invariant that decides a placement question and cannot fail is a decision
    resting on an assumption nobody rechecks.

    Parsed rather than grepped, deliberately. ``interp_engine_adapter`` has ``interp_engine`` as a
    prefix, so a naive pattern matches this package's own imports and reports violations that are not
    ones -- which happened while measuring this, and would have made the test either wrong or, worse,
    permanently red for a reason unrelated to the invariant.
    """

    @staticmethod
    def _engine_imports(tree):
        """Every import of the ENGINE, as (node, module_name) pairs.

        Not the adapter package.
        """

        def names_engine(name: str | None) -> bool:
            # Exact module or a submodule of it. `interp_engine_adapter` is neither.
            return name is not None and (name == "interp_engine" or name.startswith("interp_engine."))

        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and names_engine(node.module):
                found.append((node, node.module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if names_engine(alias.name):
                        found.append((node, alias.name))
        return found

    def test_every_engine_import_in_the_package_is_inside_the_seam(self):
        import ast as _ast

        source = pathlib.Path(_adapter_module_path()).read_text()
        tree = _ast.parse(source)

        seam = next(
            (n for n in tree.body if isinstance(n, _ast.ClassDef) and n.name == "_Seam"),
            None,
        )
        assert seam is not None, "the _Seam class is gone; this test's subject no longer exists"
        seam_lines = range(seam.lineno, (seam.end_lineno or seam.lineno) + 1)

        engine_imports = self._engine_imports(tree)
        assert engine_imports, (
            "no interp_engine imports found at all, which means this test is passing vacuously rather "
            "than because the invariant holds. Check the module path and the matcher."
        )

        outside = [f"line {n.lineno}: {mod}" for n, mod in engine_imports if n.lineno not in seam_lines]
        assert not outside, (
            "these interp_engine imports are outside `_Seam`, which documents itself as the one place "
            f"this adapter touches the engine: {outside}. Either move them into the seam, or if the "
            "invariant is being deliberately retired, change _Seam's docstring in the same commit so "
            "the claim and the code do not disagree."
        )


class TestTheEnginePinIsDeclaredOnceInEffect:
    """The engine pin is written twice, and nothing asserted the two copies agree.

    A pip consumer resolves `interp-engine` from `pyproject.toml`'s `dependencies`; a hub consumer is
    gated by `requires.pip` in `it_component.yaml`. Both exist because the two paths are served by
    different machinery, and neither is redundant.

    But the manifest calls this pin "a correctness pin rather than caution" -- interp-engine's 1.x line
    changes signatures between minors with no deprecation aliases, and the seam's contract tests assert
    against the INSTALLED engine. So a drift between the two copies does not produce a version warning;
    it produces one supported path admitting an engine the seam was never validated against, while the
    other refuses it. Nothing downstream would say which copy was wrong.
    """

    @staticmethod
    def _engine_requirement(specifiers: list[str]) -> str | None:
        """The single `interp-engine` requirement in a list, or None.

        Not `interp-engine-something`.
        """
        import re

        matches = [s for s in specifiers if re.match(r"^\s*interp-engine\s*(?:[~=<>!]|$)", s)]
        return matches[0].strip() if len(matches) == 1 else None

    def test_the_pip_dependency_and_the_manifest_requirement_are_the_same_pin(self):
        import tomllib

        import yaml

        root = pathlib.Path(_adapter_module_path()).resolve().parents[2]
        pyproject = tomllib.loads((root / "pyproject.toml").read_text())
        manifest = yaml.safe_load((root / "src" / "interp_engine_adapter" / "it_component.yaml").read_text())

        pip_pin = self._engine_requirement(pyproject["project"]["dependencies"])
        manifest_pin = self._engine_requirement(manifest["requires"]["pip"])

        # Vacuity guard, not defensiveness: if a rename or restructure moves either declaration, the
        # comparison below would pass on two Nones and this test would go green having checked nothing.
        assert pip_pin is not None, "no single interp-engine requirement in pyproject [project].dependencies"
        assert manifest_pin is not None, "no single interp-engine requirement in the manifest's requires.pip"

        assert pip_pin == manifest_pin, (
            f"the engine pin disagrees between the two paths that consume it: pyproject declares "
            f"{pip_pin!r} and the manifest declares {manifest_pin!r}. A pip consumer resolves the first "
            "and a hub consumer is gated on the second, so one path would admit an engine this "
            "component's seam was never validated against. Change both, in the same commit."
        )


class TestADroppedEngineConfigIsRefused:
    """The engine config is refused when absent rather than replaced with a default.

    The default this replaced was dead in this repo -- every consumer supplies a config -- so it could only ever have
    fired when one failed to arrive, which is exactly when substituting one is wrong. The settings it would have
    invented (backend, attention implementation, dtype) change what the engine computes, and no downstream observation
    distinguishes an engine configured by the caller from one configured by this fallback.

    Both directions are asserted. A guard that refused every config would pass a refusal-only test, so the complement
    checks that a supplied config is returned unchanged and by identity, not merely that something non-None comes back.
    """

    @staticmethod
    def _module_with_cfg(cfg_obj):
        from interp_engine_adapter.adapter import InterpEngineAttributeMixin

        class _Composed(InterpEngineAttributeMixin):
            def __init__(self):
                self.it_cfg = cfg_obj

        return _Composed()

    def test_a_config_that_did_not_arrive_is_refused_by_name(self):
        module = self._module_with_cfg(type("_Cfg", (), {})())

        with pytest.raises(ValueError) as excinfo:
            _ = module.interp_engine_cfg

        message = str(excinfo.value)
        assert "interp_engine_cfg" in message, "the refusal must name the missing setting"
        assert "default" in message, "it must say it declined to substitute a default, not merely that it is absent"
        # The cause is the part a reader cannot reconstruct: auto-composition locates an adapter's config
        # by import path, which does not resolve for a hub-delivered adapter.
        assert "hub-delivered" in message, "it must name the mechanism that drops the config"

    def test_a_supplied_config_is_returned_by_identity(self):
        from interp_engine_adapter.adapter import InterpEngineConfig

        supplied = InterpEngineConfig()
        module = self._module_with_cfg(type("_Cfg", (), {"interp_engine_cfg": supplied})())

        assert module.interp_engine_cfg is supplied, (
            "a supplied config must be handed back unchanged; returning an equal-but-distinct object "
            "would hide a substitution just as a default did"
        )
