"""The seam's own tests. This is where this adapter's correctness lives.

Interpretune's core suite owns the RAILS (that the `adapters` kind loads, gates and registers, tested
against a fixture) and ONE optional contract test (that the published component still satisfies them).
Everything about what this adapter actually does belongs here, so that an interp-engine minor release
moving a signature turns THIS repo red rather than a framework nobody expects to break.
"""

from __future__ import annotations

import pytest

from interp_engine_adapter import (
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


class TestRegistration:
    def test_registers_both_declared_compositions(self):
        from interpretune.adapters.registration import CompositionRegistry, register_dynamic_adapter

        register_dynamic_adapter(ADAPTER_NAME, source="speediedan/it-interp-engine-adapter")
        registry = CompositionRegistry()
        InterpEngineAdapter.register_adapter_ctx(registry)

        keys = registry.available_compositions()
        assert len(keys) == 2, f"expected the two declared compositions, got {keys}"
        for key in keys:
            assert registry.get(key) == (InterpEngineModule,)


class TestModelBackend:
    """The backend claims exactly what it implements, and refuses what it does not."""

    def test_claims_intervention_and_nothing_else(self):
        from interpretune.analysis.backends.capabilities import BackendCapability
        from interp_engine_adapter import InterpEngineModelBackend

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
        from interp_engine_adapter import InterpEngineModelBackend

        backend = InterpEngineModelBackend()
        assert isinstance(backend, ModelBackendCore)
        assert isinstance(backend, SupportsIntervention)
        # The point of the decomposition: a partial backend is NOT the full protocol, and structural
        # checks say so rather than the mismatch surfacing as an AttributeError inside an op.
        assert not isinstance(backend, SupportsLatentModels)
        assert not isinstance(backend, SupportsGradients)
        assert not isinstance(backend, ModelBackend)

    def test_latent_handles_are_refused_not_ignored(self):
        from interp_engine_adapter import InterpEngineModelBackend, UnmappableHookError

        backend = InterpEngineModelBackend()
        with pytest.raises(UnmappableHookError, match="does not support latent-model handles"):
            backend.fwd_w_intervention(None, {"input_ids": None}, {}, latent_model_handles=[object()])

    def test_a_callable_names_filter_is_refused_rather_than_under_capturing(self):
        from interp_engine_adapter import InterpEngineModelBackend, UnmappableHookError

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
        from interp_engine_adapter import UnmappableHookError, _intervention_targets

        with pytest.raises(UnmappableHookError, match="a model is required"):
            _intervention_targets({"blocks.5.hook_mlp_out": object()}, None)

    def test_the_model_blind_reading_really_does_differ(self):
        """Positive control for the refusal above, and the reason it exists.

        If the engine ever started raising on a null model, or the alias table grew to cover these
        names, the refusal would be guarding nothing -- and this test would say so by failing.
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
    from interp_engine_adapter import _Seam

    return _Seam._SUBLAYER_INPUT_ALIASES


class TestInterventionModeIsHonoured:
    """`InterventionSpec.mode` must reach the engine, or be refused -- never silently dropped.

    The first draft of this backend built every SteerSpec with the default method, so `mode` was
    ignored entirely. Interpretune's DEFAULT mode is `replace`, and the J-space steering path uses
    `patch`; both would have been applied as ADDITIVE steering, completing normally and returning
    plausible logits for an intervention nobody requested.
    """

    def test_add_maps_to_additive(self):
        from interp_engine_adapter import _steer_method_for

        assert _steer_method_for("add") == "additive"

    @pytest.mark.parametrize("mode", ["replace", "patch", "project"])
    def test_modes_that_need_the_current_activation_are_refused(self, mode):
        from interp_engine_adapter import UnmappableHookError, _steer_method_for

        with pytest.raises(UnmappableHookError, match=f"intervention mode {mode!r} cannot be expressed"):
            _steer_method_for(mode)

    def test_every_mapped_method_is_one_the_engine_actually_offers(self):
        """Positive control: the mapping table must name real engine methods, not aspirational ones."""
        from interp_engine import STEER_METHODS

        from interp_engine_adapter import _MODE_TO_STEER_METHOD

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
    from interp_engine_adapter import _MODE_TO_STEER_METHOD

    return _MODE_TO_STEER_METHOD
