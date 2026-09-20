"""The three-way pairing loads circuit-tracer's replacement model over the engine (#36).

Before this landed the pairing registered and composed, and ``replacement_model`` stayed ``None`` until the
first attribution-driven op failed on it. Neither test here needs a model: the contract is WHICH call the
hook makes and WITH WHAT, plus the registration interpretune's rails validate a backend name against.
"""

from __future__ import annotations

import pytest

from interp_engine_adapter import adapter as adapter_mod

requires_ct = pytest.mark.skipif(
    adapter_mod._circuit_tracer_pairing() is None, reason="circuit-tracer not installed (optional dependency)"
)


@requires_ct
def test_building_the_pairing_registers_the_engine_as_a_circuit_tracer_backend():
    from interpretune.adapters.circuit_tracer.registry import CT_BACKEND_REGISTRY

    assert CT_BACKEND_REGISTRY.get(adapter_mod.ADAPTER_NAME) == "InterpEngineReplacementModel"


@requires_ct
def test_the_pairing_hook_loads_the_replacement_model_over_the_loaded_hf_model_and_arms_the_guard():
    """Positive control for both halves: the load is called with the module's HF model and tokenizer, and the
    seam guard (which reads the loaded model) fires afterwards, where before it returned early on ``None``."""
    pairing = adapter_mod._circuit_tracer_pairing()
    calls: list[dict] = []

    class _Loaded:
        def __init__(self):
            self.resolved: list[str] = []

        def _resolve_feature_hook(self, hook: str):
            self.resolved.append(hook)
            return hook

    class _HF:
        tokenizer = "tok-from-hf"

        def parameters(self):
            yield type("P", (), {"device": "cpu-probe"})()

    class _Probe(pairing):  # type: ignore[misc, valid-type]
        # `circuit_tracer_cfg` is a read-only property on the base module; the probe overrides it.
        circuit_tracer_cfg = property(lambda self: type("Cfg", (), {"backend": adapter_mod.ADAPTER_NAME})())

        def __init__(self):  # no framework init: only the hook under test runs
            self.model = _HF()
            self._engine_model = type("Engine", (), {"tokenizer": "tok-from-engine"})()
            self._replacement_model = None

        def _load_replacement_model(self, pretrained_kwargs=None):
            calls.append(dict(pretrained_kwargs or {}))
            self._replacement_model = _Loaded()

    probe = _Probe()
    probe._load_interp_engine_replacement_model()
    probe._guard_feature_hook_resolution()

    # Device is the HF model's own, read from its parameters, not circuit-tracer's default accelerator.
    assert calls == [{"hf_model": probe.model, "tokenizer": "tok-from-engine", "device": "cpu-probe"}]
    guarded = probe._replacement_model._resolve_feature_hook
    assert getattr(guarded, "_it_seam_guarded", False), "the seam guard did not arm on the loaded model"
    # The guard takes the BARE hook name circuit-tracer resolves (no `blocks.N.` prefix).
    assert guarded("hook_resid_post") == "hook_resid_post"
    with pytest.raises(adapter_mod.UnmappableHookError):
        guarded("hook_mlp_in")  # the block-level name the seam refuses to substitute


@requires_ct
def test_a_foreign_backend_name_is_refused_by_name():
    pairing = adapter_mod._circuit_tracer_pairing()

    class _Probe(pairing):  # type: ignore[misc, valid-type]
        circuit_tracer_cfg = property(lambda self: type("Cfg", (), {"backend": "nnsight"})())

        def __init__(self):
            pass

    with pytest.raises(adapter_mod.UnmappableHookError, match="must be 'interp_engine'"):
        _Probe()._load_interp_engine_replacement_model()
