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
"""This component's compositions beside their native counterparts, measured (adapter#50).

interpretune measures nothing about a hub adapter, so this component measures itself through the same delta
protocol interpretune's bundled adapters are measured with (vendored in `stats.py` and `runner.py` beside
this file; see their headers). These run where an accelerator is present and `IT_RUN_CUDA_TESTS=1` (this
repo's `cuda` marker, additive in the GPU lane). A composition without a frozen bound for the running
vendor is report-only: its artifact is written and nothing is asserted beyond the pair being well formed.
Set `IT_PROFILING_DELTA_DIR` to keep the artifacts (`<composition>.pair.json`) bounds are frozen from.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pytest
import yaml

from tests.profiling_delta.arms import COMPOSITIONS
from tests.profiling_delta.runner import measure_pair
from tests.profiling_delta.stats import evaluate

BOUNDS_FILE = Path(__file__).with_name("delta_bounds.yaml")


def _artifact_dir(tmp_path: Path) -> Path:
    return Path(os.environ["IT_PROFILING_DELTA_DIR"]) if os.environ.get("IT_PROFILING_DELTA_DIR") else tmp_path


def _frozen(vendor: str, composition: str) -> dict | None:
    bounds = yaml.safe_load(BOUNDS_FILE.read_text()) or {}
    return (bounds.get(vendor) or {}).get(composition)


@pytest.mark.cuda
@pytest.mark.parametrize("composition", sorted(COMPOSITIONS))
def test_delta_pair(tmp_path, composition):
    pytest.importorskip("interp_engine", reason="the engine this adapter wraps")
    if "circuit_tracer" in composition:
        pytest.importorskip("circuit_tracer", reason="the three-way composition needs circuit-tracer")
    out_dir = _artifact_dir(tmp_path)
    pair = measure_pair(composition, out_dir / "arms")
    (out_dir / f"{composition}.pair.json").write_text(json.dumps(pair.to_json(), indent=1))

    for arm in (pair.it, pair.upstream):
        assert all(r["available"] for r in arm["repeats"]), f"{arm['arm']} arm saw no accelerator statistics"
        assert all(r["peak_alloc"] > 0 for r in arm["repeats"])
    # bind the delta to matched arms: both consumed the same inputs, step for step
    assert [r["inputs"] for r in pair.it["repeats"]] == [r["inputs"] for r in pair.upstream["repeats"]]

    frozen = _frozen(pair.vendor, composition)
    if frozen is None:
        summary = {m: pair.deltas[m]["median"] for m in ("peak_alloc", "end_current", "slope[0]") if m in pair.deltas}
        warnings.warn(f"{composition} on {pair.vendor}: report-only (no frozen bound); median deltas {summary}")
        return
    failures = evaluate(pair.deltas, frozen)
    assert not failures, f"{composition} on {pair.vendor}: " + "; ".join(failures)
