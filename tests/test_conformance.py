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

`datamodule_flavour` is left at the default, which is now `"hf"` and was `"bridge"` when this file was
written. Still not an omission, and the reason it can be left alone is worth stating because it is the
property that has to hold, not the flavour name: the suite's position cases count REAL positions from an
`attention_mask` before comparing any set, so a flavour that passes no mask silently weakens the
precondition that stops a scope case passing on a prompt too short to discriminate.

`"hf"` passes one (`model_input_names` includes `attention_mask`, left padding, same data-side settings
as `bridge`), so the default is safe here AND better: it builds an adapter-free module config, which is
what a hub adapter should compose over rather than the bridge's. `"nnsight"` would leave the position
cases nothing to count with.

This component is key-agnostic across flavours either way: `_tokens_from_batch` delegates to
interpretune's `get_batch_input`, which resolves both `input` and `input_ids`.

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
    conformance_case,
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
        # `batch_size` is left at the suite's default, and NOT declaring it is the point.
        #
        # This target briefly declared `batch_size=1`, on the strength of an engine message reading
        # "takes one prompt at a time". That message names the MODEL it was given, and it was being
        # given the raw HuggingFace model rather than the engine one; with the engine model it batches.
        # Declaring the limit would have made the suite run every case unpadded, which is the weaker
        # validation, and would have frozen a workaround as a property of the backend.
        #
        # Running padded is what found the defect worth having found: the seam was not passing
        # `attention_mask` to the engine, so a padded row attended to its own pad positions and its
        # last-position logits were wrong by 54.9 while nothing raised. A single-row target cannot see
        # that, because an unpadded batch has no mask to drop.
    )

    # OVERRIDDEN ONLY TO MARK IT, and defined as a new function rather than marking the inherited one:
    # `pytest.mark.xfail(...)(fn)` stores the mark ON the function object, so applying it to
    # `ModelBackendConformance.test_supplied_settings_survive_composition` would mark the base for every
    # other target that inherits it, interpretune's own included.
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "speediedan/interpretune#538: this component's two-way composition works by accident. The "
            "conformance helper sets module_cfg_extras with setattr AFTER the seed config is built, so "
            "`interp_engine_cfg` never reaches the constructor, which is where composition acts, and it "
            "arrives as a stray attribute the adapter then reads back through a default. Not fixable on "
            "this side: one adapter-side workaround builds on the warn-and-proceed branch #538 proposes to "
            "turn into a refusal, and the other would leave `supplied_extras` empty, which makes this case "
            "SKIP rather than pass. strict=True is the point of using a marker at all -- it fails on an "
            "unexpected pass, so the marker cannot outlive the defect quietly once #538 lands."
        ),
    )
    @conformance_case()
    def test_supplied_settings_survive_composition(self, suite):
        super().test_supplied_settings_survive_composition(suite)
