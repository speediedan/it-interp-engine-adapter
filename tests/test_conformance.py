"""This component under interpretune's distributed conformance contract.

`docs/adapter_conformance_contract.md` in interpretune describes the split: the suite owns the inputs,
the oracles, case selection and the report; a repository owns its session config, its runtime
declarations, its forward family, and how its component is loaded. This file is the whole of this
repository's half.

**Why this replaced hand-written equivalents rather than joining them.** The tests that lived in
`test_composition_intervention_parity.py` measured the same properties -- the position set per scope,
each scope against a HuggingFace reference, the baseline being unsteered -- and they found real defects
doing it. But they were this component's own opinion about what parity means, and the value of the
contract is that the opinion is shared: a case that passes here and fails for a bundled backend is then
a statement about the backends rather than about who wrote which assertions. Keeping both would have
meant two oracles that can disagree, and the local one has no way to win that argument.

**What the target declares, and why each answer is what it is:**

`composition` is the two-adapter core case. The circuit-tracer composition is conditional on an optional
package and is covered by this repo's own registration tests; putting it here would make the suite's
selection depend on an environment the contract does not control.

`forward_family="hf_native"` because interp-engine wraps the HuggingFace model IN PLACE and runs its
forward rather than converting weights. That is what makes the value cases against the HF reference
meaningful: measured at 0.0, not at a tolerance. A backend that reimplemented the forward would have to
declare otherwise and would get structural and causal cases only.

`datamodule_flavour` is left at the default `"bridge"`, and that is a choice rather than an omission.
`_tokens_from_batch` delegates to interpretune's `get_batch_input`, which resolves both `input` and
`input_ids`, so this component is key-agnostic and either flavour would run. `"bridge"` passes an
`attention_mask`, and the suite's position cases count REAL positions from that mask before comparing
any set. The `"nnsight"` flavour would leave it nothing to count with, weakening the precondition that
keeps a scope case from passing on a prompt too short to discriminate.

`load` stages and registers the component the way the hub loader does, which is the point of a
DISTRIBUTED suite: a bundled adapter passes `None` and is simply importable, while a hub component has
to be materialised first. Staged rather than pulled, deliberately -- pulling would test a PUBLISHED
revision, which is a different and older artifact than the tree under test, and would put a network
round trip and a private-repo token inside a unit test.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile

import pytest

pytest.importorskip("interpretune.testing.conformance", reason="needs interpretune with the conformance suite")

from interpretune.testing.conformance import (  # noqa: E402
    ConformanceInputs,
    ConformanceTarget,
    ModelBackendConformance,
)

COMPOSITION = ("core", "interp_engine")


def load_component() -> None:
    """Stage this component and register it, as the hub loader would.

    Registers into the GLOBAL `ADAPTER_REGISTRY` rather than a fresh one, because the session the suite
    builds resolves its composition from there. A local registry would register successfully and leave
    the session unable to find the composition, which reads as a missing adapter rather than as a test
    harness pointing at the wrong registry.
    """
    import yaml
    from interpretune.adapters import ADAPTER_REGISTRY
    from interpretune.adapters.registration import register_dynamic_adapter
    from interpretune.hub import adapters as hub_adapters
    from interpretune.hub.adapters import _partition_declared_compositions
    from interpretune.hub.components import enforce_component_requires
    from interpretune.hub.publish import build_component_tree

    out = pathlib.Path(tempfile.mkdtemp()) / "staged"
    build_component_tree(pathlib.Path("src/interp_engine_adapter"), out)
    manifest = yaml.safe_load((out / "it_component.yaml").read_text())

    # Part of the loading path, so it belongs here: a component whose declared engine range excludes the
    # installed one must not be quietly exercised anyway.
    enforce_component_requires(manifest, source="conformance")

    name = "it_hub_adapters.conformance.staged"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, out / manifest["adapters"]["entrypoint"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]

    satisfiable, _ = _partition_declared_compositions(manifest, source="conformance")
    register_dynamic_adapter("interp_engine", source="speediedan/it-interp-engine-adapter")

    token = hub_adapters._SUPPORTED_COMPOSITIONS.set(tuple(satisfiable))
    try:
        module.InterpEngineAdapter.register_adapter_ctx(ADAPTER_REGISTRY)
    finally:
        hub_adapters._SUPPORTED_COMPOSITIONS.reset(token)


def build_session_cfg(inputs: ConformanceInputs):
    """The seed session, plus this component's config.

    `InterpEngineConfig` is imported from the STAGED module rather than from the installed package, so
    the config class the session gets is the one the staged entrypoint defined. Importing the pip copy
    would work and would silently test a different object than `load_component` registered.
    """
    module = sys.modules["it_hub_adapters.conformance.staged"]

    return inputs.session_cfg(
        COMPOSITION,
        module_cfg_extras={"interp_engine_cfg": module.InterpEngineConfig()},
    )


class TestInterpEngineConformance(ModelBackendConformance):
    target = ConformanceTarget(
        composition=COMPOSITION,
        session_cfg_factory=build_session_cfg,
        forward_family="hf_native",
        load=load_component,
        # ONE ROW PER BATCH, declared rather than worked around. interp-engine's eager tier is
        # single-prompt, and this adapter refuses a multi-row batch instead of looping, because
        # interpretune's batches are left-padded and per-row unpadded forwards would compute something
        # different from the batched forward while looking like an approximation of it. Declaring the
        # limit runs every case unpadded AND turns the refusal itself into a case, which is the same
        # move the position-scope work made: a limitation the vocabulary could not express looked like a
        # defect until it could.
        batch_size=1,
    )
