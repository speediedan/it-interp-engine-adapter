# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Arms for this component's memory-profiling delta pairs (adapter#50).

Each composition is a pair: the composition class staged and registered the way the hub loader loads it, hosted
minimally around a loaded Hugging Face model, against the same capture or attribution driven through
interp-engine's (or circuit-tracer's) own API on the same model. interpretune measures nothing about a hub
adapter, so this component measures itself: composing ``interp_engine`` (and, for the second pair,
``circuit_tracer`` over the engine) should cost memory only in interpretune-side objects, which is what the
delta attributes. The harness protocol (recorder, one subprocess per arm, paired deltas) is vendored from
interpretune's `tests/profiling_delta` in `stats.py` and `runner.py` beside this file; see their headers.

The hosting follows `test_composition_intervention_parity.py` on purpose: a full interpretune session would
bring configuration this measurement does not use, and what is under test is the adapter's capture and
attribution surface over an already-loaded model, which is the same claim the parity tests make.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tests.profiling_delta.stats import StepRecorder

ArmFn = Callable[[StepRecorder, int, int], None]

MODEL_ID = "gpt2"
LAYER = 5
PROMPT = "The capital of France is"
HOOK_NAME = f"blocks.{LAYER}.hook_resid_pre"

CT_MODEL_NAME = "google/gemma-3-1b-pt"
CT_TRANSCODER_SET = "mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine"
CT_PROMPT = "Fact: the capital of the state containing Dallas is"


@dataclass
class Composition:
    """One sampled composition: its interpretune arm and the matched native arm."""

    name: str
    description: str
    it_arm: ArmFn
    upstream_arm: ArmFn
    warmup: int | None = None
    window: int | None = None

    def arm(self, which: str) -> ArmFn:
        if which not in ("it", "upstream"):
            raise ValueError(f"unknown arm {which!r}")
        return self.it_arm if which == "it" else self.upstream_arm


def _device() -> torch.device:
    accel = getattr(torch, "accelerator", None)
    return accel.current_accelerator() if accel is not None and accel.is_available() else torch.device("cpu")


def _staged_module():
    """The entrypoint, executed from a staged component tree the way the hub loader executes it."""
    import importlib.util
    import pathlib
    import sys
    import tempfile

    import yaml
    from interpretune.hub.components import enforce_component_requires
    from interpretune.hub.publish import build_component_tree

    out = pathlib.Path(tempfile.mkdtemp()) / "staged"
    build_component_tree(pathlib.Path("src/interp_engine_adapter"), out)
    manifest = yaml.safe_load((out / "it_component.yaml").read_text())
    enforce_component_requires(manifest, source="profiling_delta")
    name = "it_hub_adapters.staged.profiling_delta"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, out / manifest["adapters"]["entrypoint"])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name], manifest


def _composition_class(staged, adapters: list[str]):
    """The module class for `adapters`, taken from a registry rather than imported (as the parity tests do)."""
    from interpretune.adapters.registration import CompositionRegistry, register_dynamic_adapter
    from interpretune.hub import adapters as hub_adapters
    from interpretune.hub.adapters import _partition_declared_compositions

    module, manifest = staged
    satisfiable, _ = _partition_declared_compositions(manifest, source="profiling_delta")
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
        if k[0] == "module" and sorted(getattr(a, "value", str(a)) for a in k[1:]) == sorted(adapters)
    )
    return registry.get(key)[0]


def _hosted(composition_class, hf_model, it_cfg, pre_attrs: dict[str, Any] | None = None):
    """The composition class minimally hosted around a real model, with its init hooks run."""

    class _Host(composition_class):  # type: ignore[valid-type, misc]
        def __init__(self):
            nn.Module.__init__(self)
            object.__setattr__(self, "_hosted_model", hf_model)
            self.tokenizer = None
            self.it_cfg = it_cfg

        @property
        def model(self):
            return self._hosted_model

    host = _Host()
    for name, value in (pre_attrs or {}).items():
        object.__setattr__(host, name, value)
    host.post_auto_model_init()
    return host


def _it_cfg(staged_module, model_id: str, **extra):
    return type(
        "_Cfg",
        (),
        {"model_name_or_path": model_id, "interp_engine_cfg": staged_module.InterpEngineConfig(), **extra},
    )()


def _prompt_ids(tokenizer, prompt: str = PROMPT):
    return tokenizer(prompt, return_tensors="pt")["input_ids"]


def interp_engine_it_arm(recorder: StepRecorder, steps: int, seed: int) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    dev = _device()
    staged = _staged_module()
    composition_class = _composition_class(staged, ["core", "interp_engine"])
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(dev).eval()
    host = _hosted(composition_class, hf_model, _it_cfg(staged[0], MODEL_ID))
    ids = _prompt_ids(AutoTokenizer.from_pretrained(MODEL_ID)).to(dev)
    for _ in range(steps):
        recorder.note_input({"input_ids": ids})
        cache = host.engine_capture(ids, [HOOK_NAME])
        recorder.step_end()
        del cache
    del host, hf_model


def interp_engine_upstream_arm(recorder: StepRecorder, steps: int, seed: int) -> None:
    from interp_engine import EagerModel, run_with_cache, tlens_hook_to_point
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    dev = _device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(dev).eval()
    # interp-engine's own usage: wrap the loaded model and capture through its public point vocabulary
    engine = EagerModel(MODEL_ID, hf_model=hf_model, tokenizer=tokenizer, requires_grad=False, dtype=None)
    point = tlens_hook_to_point(HOOK_NAME, engine)
    ids = _prompt_ids(tokenizer).to(dev)
    for _ in range(steps):
        recorder.note_input({"input_ids": ids})
        cache = run_with_cache(engine, ids, [point], detach=True)
        recorder.step_end()
        del cache
    del engine, hf_model


def interp_engine_ct_it_arm(recorder: StepRecorder, steps: int, seed: int) -> None:
    from interpretune.adapters.circuit_tracer.config import CircuitTracerConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    dev = _device()
    staged = _staged_module()
    composition_class = _composition_class(staged, ["circuit_tracer", "core", "interp_engine"])
    hf_model = AutoModelForCausalLM.from_pretrained(CT_MODEL_NAME, dtype=torch.bfloat16).to(dev).eval()
    ct_cfg = CircuitTracerConfig(
        backend=staged[0].ADAPTER_NAME,
        transcoder_set=CT_TRANSCODER_SET,
        offload="cpu",
        analysis_target_tokens=["▁Dallas", "▁Austin"],
        verbose=False,
    )
    # the minimal host bypasses BaseCircuitTracerModule.__init__, which owns these attributes, and the
    # pairing hook reads circuit_tracer_cfg as a module attribute, so both are set before the hook runs
    host = _hosted(
        composition_class,
        hf_model,
        _it_cfg(staged[0], CT_MODEL_NAME, circuit_tracer_cfg=ct_cfg),
        pre_attrs={"attribution_graphs": [], "_replacement_model": None},
    )
    tokenizer = AutoTokenizer.from_pretrained(CT_MODEL_NAME)
    for _ in range(steps):
        recorder.note_input({"input_ids": _prompt_ids(tokenizer, CT_PROMPT)})
        host.generate_attribution_graph(CT_PROMPT, verbose=False)
        recorder.step_end()
        # the module retains every graph it generates where it keeps them; the native arm discards each
        graphs = host.__dict__.get("attribution_graphs")
        if graphs is not None:
            graphs.clear()
    del host, hf_model


def interp_engine_ct_upstream_arm(recorder: StepRecorder, steps: int, seed: int) -> None:
    from circuit_tracer import ReplacementModel, attribute
    from transformers import AutoTokenizer

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(CT_MODEL_NAME)
    # circuit-tracer's own factory on its default nnsight backend, with the interpretune composition's kwargs
    model = ReplacementModel.from_pretrained(
        CT_MODEL_NAME,
        CT_TRANSCODER_SET,
        backend="nnsight",
        dtype=torch.bfloat16,
        lazy_encoder=True,
        lazy_decoder=True,
        device_map=0,
        device=str(_device()),
    )
    for _ in range(steps):
        recorder.note_input({"input_ids": _prompt_ids(tokenizer, CT_PROMPT)})
        graph = attribute(
            prompt=CT_PROMPT,
            model=model,
            attribution_targets=["▁Dallas", "▁Austin"],
            max_n_logits=10,
            desired_logit_prob=0.95,
            batch_size=256,
            max_feature_nodes=8192,
            offload="cpu",
            verbose=False,
        )
        recorder.step_end()
        del graph
    del model


COMPOSITIONS: dict[str, Composition] = {
    "interp_engine_test": Composition(
        name="interp_engine_test",
        description="the (core, interp_engine) composition capturing one resid_pre point per step on gpt2 versus "
        "interp-engine's run_with_cache on an EagerModel wrapping the same Hugging Face model",
        it_arm=interp_engine_it_arm,
        upstream_arm=interp_engine_upstream_arm,
    ),
    "interp_engine_circuit_tracer_test": Composition(
        name="interp_engine_circuit_tracer_test",
        description="the (core, interp_engine, circuit_tracer) composition attributing one prompt per step on "
        "gemma-3-1b-pt versus a bare circuit-tracer ReplacementModel (nnsight backend) on the same model and "
        "transcoder set",
        it_arm=interp_engine_ct_it_arm,
        upstream_arm=interp_engine_ct_upstream_arm,
        # an attribution is far heavier than a capture: one warmup attribution and a 3-attribution window
        warmup=1,
        window=3,
    ),
}

ARMS = ("it", "upstream")
