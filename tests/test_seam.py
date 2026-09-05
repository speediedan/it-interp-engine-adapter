"""The seam's own tests. This is where this adapter's correctness lives.

Interpretune's core suite owns the RAILS (that the `adapters` kind loads, gates and registers, tested
against a fixture) and ONE optional contract test (that the published component still satisfies them).
Everything about what this adapter actually does belongs here, so that an interp-engine minor release
moving a signature turns THIS repo red rather than a framework nobody expects to break.
"""

from __future__ import annotations

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
        # Enumerated from the ENUM rather than a hand-written list. The previous version named
        # BATCHED_HOOKS explicitly and went stale the moment speediedan/interpretune#449 removed it,
        # which is the failure mode of pinning a vocabulary by transcription: the test breaks on a
        # change it was not written to have an opinion about. Deriving the complement means a member
        # added upstream is asserted absent automatically, and a member removed simply disappears.
        for absent in set(BackendCapability) - {BackendCapability.INTERVENTION}:
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
