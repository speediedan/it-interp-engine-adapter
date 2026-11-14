"""The tuning example runs from a run configuration alone: ``hub_adapters`` loads this adapter before the CLI parses
(speediedan/interpretune#872), and ``examples/rte_ft/rte_ft.gpt2.interp_engine.core.yaml`` trains phased on the
``(core, interp_engine)`` composition (#48).

The shipped file names the PUBLISHED adapter revision. Here the entry points at this checkout instead, published
locally into a throwaway components cache beside a copy of the warmed ``speediedan/rte`` seed, so the run executes the
code under review and no other cache is touched. The schedule is the shipped one's shape, cut to two phases and written
locally, because the warm step fetches the seed's configurations and not its extra files.
"""

from __future__ import annotations

import pathlib
import shutil
import sys

import pytest
import yaml

pytest.importorskip("interpretune.hub.adapters", reason="needs interpretune's hub adapter loader")
hub_adapters = pytest.importorskip("interpretune.hub.adapters")
if not hasattr(hub_adapters, "load_run_hub_adapters"):
    pytest.skip("interpretune predates run-configuration hub_adapters (#872)", allow_module_level=True)
pytest.importorskip(
    "finetuning_scheduler.core.tuning_scheduler", reason="finetuning-scheduler predates TuningScheduler"
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "rte_ft" / "rte_ft.gpt2.interp_engine.core.yaml"
REPO = "speediedan/it-interp-engine-adapter"
SCHEDULE = {
    0: {"params": [r"model.transformer.ln_f.*", r"model.transformer.h.(10|11)\..*"], "max_transition_epoch": 1},
    1: {"params": [r"model.transformer.h.[6-9]\..*"], "lr": 3.0e-05},
}


@pytest.fixture(autouse=True)
def restore_adapter_registration():
    """Undo the run's hub load, so later tests can register their own copy of this adapter.

    The run loads this checkout as a locally published revision, which defines a second ``ITInterpEngineConfig`` and
    registers it for ``interp_engine``. interpretune refuses a second config class for one adapter, so leaving it
    registered fails every later test that stages the component (``test_conformance.load_component``).
    """
    from interpretune.adapters import ADAPTER_REGISTRY

    registry = ADAPTER_REGISTRY.registry  # the populated ``CompositionRegistry`` behind the lazy wrapper
    compositions = dict(registry)
    cfg_classes = dict(registry._module_cfg_classes)
    loaded = dict(hub_adapters._LOADED_HUB_ADAPTERS)
    modules = set(sys.modules)
    yield
    registry.clear()
    registry.update(compositions)
    registry._module_cfg_classes.clear()
    registry._module_cfg_classes.update(cfg_classes)
    hub_adapters._LOADED_HUB_ADAPTERS.clear()
    hub_adapters._LOADED_HUB_ADAPTERS.update(loaded)
    for name in set(sys.modules) - modules:
        if name == hub_adapters.HUB_ADAPTER_PACKAGE or name.startswith(f"{hub_adapters.HUB_ADAPTER_PACKAGE}."):
            del sys.modules[name]


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    """A components cache holding this checkout's adapter and the warmed ``speediedan/rte`` seed; returns
    ``(adapter revision, rte revision)``."""
    from interpretune.hub import components
    from interpretune.testing.conformance.inputs import SEED_REPO, SEED_REVISION

    warmed = components._repo_dir(SEED_REPO, None)
    if SEED_REVISION not in components.cached_component_revisions(SEED_REPO):
        pytest.skip(f"{SEED_REPO}@{SEED_REVISION[:12]} is not in the components cache (CI's warm step fetches it)")
    root = tmp_path / "components"
    shutil.copytree(warmed, components._repo_dir(SEED_REPO, root), symlinks=True)
    monkeypatch.setattr(components, "IT_COMPONENTS_HUB_CACHE", root)
    monkeypatch.setenv("IT_TRUST_REMOTE_CODE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    adapter_revision = components.local_publish(ROOT / "src" / "interp_engine_adapter", REPO, cache_dir=root)
    return adapter_revision, SEED_REVISION


def test_the_example_trains_phased_from_its_run_configuration(cache, tmp_path, monkeypatch):
    from interpretune.base.components.cli import core_cli_main
    from interpretune.hub.adapters import loaded_hub_adapters

    adapter_revision, rte_revision = cache
    shipped = yaml.safe_load(EXAMPLE.read_text())
    assert [e.split("@")[0] for e in shipped["hub_adapters"]] == [REPO]
    schedule = tmp_path / "schedule.yaml"
    schedule.write_text(yaml.safe_dump(SCHEDULE))
    run = dict(shipped)
    run["hub_adapters"] = [f"{REPO}@{adapter_revision}"]
    run["session_ref"] = f"{shipped['session_ref']}@{rte_revision}"
    callback = run["run_cfg"]["callbacks"][0]["init_args"]
    callback["schedule"] = str(schedule)
    # a large `min_delta`, so the last phase plateaus within a CPU-sized run
    callback["tuning_scheduler_kwargs"]["transition"]["init_args"]["min_delta"] = 2.0
    run["run_cfg"].update(max_epochs=3, limit_train_batches=2, limit_val_batches=2)
    run["registered_cfg"]["datamodule_cfg"] = {"dataset_path": str(tmp_path / "dataset")}
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(run))
    monkeypatch.chdir(tmp_path)  # the checkpoint directory is relative

    cli = core_cli_main(run_mode=False, args=["--config", str(config)])
    composed = [c.__name__ for c in type(cli.module.it_cfg).__mro__]
    assert "ITInterpEngineConfig" in composed and "RTEBoolqConfig" in composed
    cli.runner.train()

    fts = cli.runner.run_cfg.callbacks[0].tuning_scheduler
    assert fts.curr_depth == 1, "the run should reach the schedule's last phase"
    names = dict(cli.module.named_parameters())
    thawed = {n for n, p in names.items() if p.requires_grad}
    assert any(n.startswith("model.transformer.h.6.") for n in thawed) and not any(
        n.startswith("model.transformer.h.0.") for n in thawed
    )
    record = loaded_hub_adapters()[REPO]
    assert record["revision"] == adapter_revision
    assert cli.module._it_state._init_hparams["hub_adapters"][REPO]["revision"] == adapter_revision
