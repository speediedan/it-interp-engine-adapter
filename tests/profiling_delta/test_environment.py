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
"""A frozen bound's environment is recorded from its artifacts and compared by name (CPU)."""

from __future__ import annotations

import json

import pytest
import yaml

from tests.profiling_delta.environment import (
    SALIENT_DISTRIBUTIONS,
    environment_mismatch,
    main,
    record,
    salient_versions,
)

ENV = {name: "1.0" for name in SALIENT_DISTRIBUTIONS}


def _artifact(tmp_path, name, vendor, environment):
    path = tmp_path / f"{name}.pair.json"
    payload = {"composition": name, "vendor": vendor, "deltas": {}}
    if environment is not None:
        payload["environment"] = environment
    path.write_text(json.dumps(payload))
    return path


def test_the_running_environment_names_every_salient_distribution():
    versions = salient_versions()
    assert list(versions) == list(SALIENT_DISTRIBUTIONS)
    assert versions["torch"], "torch is installed wherever these tests run"


def test_a_mismatch_names_each_distribution_that_moved():
    running = {**ENV, "torch": "2.13.0+cu130", "transformer-lens": "4.4.0"}
    assert environment_mismatch(ENV, running) == ["torch: 1.0 -> 2.13.0+cu130", "transformer-lens: 1.0 -> 4.4.0"]
    assert environment_mismatch(ENV, dict(ENV)) == []


def test_record_returns_each_vendors_environment(tmp_path):
    _artifact(tmp_path, "a", "cuda", ENV)
    _artifact(tmp_path, "b", "cuda", ENV)
    assert record([tmp_path]) == {"cuda": ENV}


def test_record_refuses_one_vendor_measured_in_two_environments(tmp_path):
    _artifact(tmp_path, "a", "cuda", ENV)
    _artifact(tmp_path, "b", "cuda", {**ENV, "torch": "2.0"})
    with pytest.raises(SystemExit, match="different environments: torch: 1.0 -> 2.0"):
        record([tmp_path])


def test_record_refuses_an_artifact_without_an_environment(tmp_path):
    _artifact(tmp_path, "a", "cuda", None)
    with pytest.raises(SystemExit, match="records no environment"):
        record([tmp_path])


def test_main_replaces_the_measured_vendors_and_keeps_the_others(tmp_path):
    out = tmp_path / "frozen_environments.yaml"
    out.write_text(yaml.safe_dump({"rocm": {**ENV, "torch": "2.15.0+rocm10.0"}, "cuda": {**ENV, "torch": "old"}}))
    artifacts = tmp_path / "nvidia"
    artifacts.mkdir()
    _artifact(artifacts, "a", "cuda", ENV)
    main(["--artifacts", str(artifacts), "--out", str(out)])
    written = yaml.safe_load(out.read_text())
    assert written == {"cuda": ENV, "rocm": {**ENV, "torch": "2.15.0+rocm10.0"}}
