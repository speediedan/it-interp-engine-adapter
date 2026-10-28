"""A short phased run through the hub-staged ``(core, interp_engine)`` composition, steered first (#48).

The tuning example's shape (speediedan/interpretune#812) on a model small enough for any runner: a steering vector
declared in configuration (``tuning_interventions``, role ``intervention:steer``) trains alone, then two blocks thaw by
the module's own names. The run asserts what the example relies on:

- phase 0 changes the steering vector and nothing else, bitwise;
- phase 1 thaws exactly the scheduled blocks' tensors beside the vector;
- the loss falls.

Device-agnostic: it runs on the current accelerator when there is one (NVIDIA or AMD alike, through
``torch.accelerator``) and on the CPU otherwise, so the adapter's own pipeline runs it on either vendor.
"""

from __future__ import annotations

from functools import partial

import pytest
import torch

pytest.importorskip("interpretune.tuning.interventions", reason="needs interpretune's interventions seam (#812)")
pytest.importorskip("finetuning_scheduler.tunable_view", reason="finetuning-scheduler predates the view")

from interpretune.testing.conformance import ConformanceInputs  # noqa: E402

from test_conformance import COMPOSITION, load_component  # noqa: E402

SITE = "model.transformer.h.5"
SCHEDULE = {
    0: {"params": ["role:intervention:steer"], "max_transition_epoch": 1},
    1: {"params": [r"model\.transformer\.h\.1[01]\..*"]},
}
STEPS_PER_EPOCH = 4


def _device() -> torch.device:
    accelerator = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None
    return accelerator if accelerator is not None else torch.device("cpu")


@pytest.fixture(scope="module")
def steered_module():
    import sys

    from interpretune.session import ITSession
    from interpretune.tuning.interventions import InterventionCfg

    load_component()
    staged = sys.modules["it_hub_adapters.conformance.staged"]
    cfg = ConformanceInputs().session_cfg(
        COMPOSITION,
        module_cfg_extras={
            "interp_engine_cfg": staged.InterpEngineConfig(),
            "tuning_interventions": [InterventionCfg(name="steer", site=SITE)],
        },
    )
    module = ITSession(cfg).module
    # the whole module, as Lightning moves it: the steering vector lives beside the model (`interventions`), not in it
    module.to(_device())
    return module


def test_a_steered_phased_run_changes_only_what_each_phase_schedules(steered_module):
    from finetuning_scheduler import TuningScheduler

    from interpretune.tuning.views import VIEW_ATTR

    module = steered_module
    model = module.model
    device = next(model.parameters()).device
    assert module.interventions["steer"].vector.device == device
    torch.manual_seed(0)
    ids = torch.randint(0, model.config.vocab_size, (4, 16), device=device)

    fts = TuningScheduler(
        module, SCHEDULE, optimizer=partial(torch.optim.AdamW, lr=1e-3), view=getattr(module, VIEW_ATTR)
    )
    fts.setup()
    names = dict(module.named_parameters())
    block = sorted(n for n in names if n.startswith(("model.transformer.h.10.", "model.transformer.h.11.")))
    snapshot = {n: p.detach().clone() for n, p in names.items()}
    thawed, losses = [], []
    model.train()
    for _ in fts.epochs(2):
        depth = fts.curr_depth
        thawed.append(sorted(n for n, p in names.items() if p.requires_grad))
        for _ in range(STEPS_PER_EPOCH):
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            fts.optimizer.step()
            fts.optimizer.zero_grad()
            losses.append(loss.item())
        changed = {n for n, p in names.items() if not torch.equal(snapshot[n], p.detach())}
        if depth == 0:
            assert changed == {"interventions.steer.vector"}, sorted(changed - {"interventions.steer.vector"})[:5]
        fts.epoch_end({})

    assert thawed[0] == ["interventions.steer.vector"]
    assert thawed[1] == sorted(["interventions.steer.vector", *block])
    assert changed == {"interventions.steer.vector", *block}
    assert losses[-1] < losses[0]
