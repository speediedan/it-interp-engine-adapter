#!/usr/bin/env bash
# The GPU test run (inside the GPU job's container, through ctr.sh). Moved out of gpu-tests.yml when the job became
# any-vendor; the reasoning below is unchanged.
#
# REPORT `collected` BESIDE `passed`. A green pytest summary answers "did the tests I SELECTED pass" and reads as "did
# the tests pass". Those differ the moment anything narrows selection, and `IT_RUN_CUDA_TESTS` is a SELECTOR in
# interpretune's conftest (it replaces the collected list rather than adding to it), not merely a gate that un-skips
# tests. This repo's conftest keeps it ADDITIVE (tests/conftest.py), so here both numbers are the full suite, and the
# comparison is what stops a narrowed run from being reported as a full one if that ever changes.
#
# `torch.cuda` is the device API on both vendors: a ROCm torch answers `torch.cuda.is_available()` for its HIP devices,
# so the `cuda` marker and IT_RUN_CUDA_TESTS mean "a GPU" on either agent. A test that genuinely needs one vendor carries
# the `nvidia_only` marker (tests/conftest.py), which skips it, with its reason, on an AMD agent.
#
# `pipefail` MATTERS HERE: pytest is piped to `tee`, and `set -e` alone reports the exit status of the LAST command in a
# pipeline. Without it a failing suite would hand the step `tee`'s success and the build would go green on red tests.
set -eo pipefail
collected=$(python -m pytest --collect-only -q 2>/dev/null | tail -1)
echo "full collection: ${collected}"
# `-rs` so skip REASONS are in the same output the run already produces, and `tee` so the assertion below reads that
# run rather than a second one: running the suite twice to check a property of the first is both slower and a lie if
# the two runs ever disagree.
IT_RUN_CUDA_TESTS=1 python -m pytest -v -rs --durations=25 2>&1 | tee /tmp/gpu_run.txt
# ASSERT NOTHING SKIPPED FOR A MISSING OPTIONAL DEPENDENCY. This is the lane that installs circuit-tracer, so a skip here
# means the install step silently stopped working. Reporting `collected` beside `passed` catches a NARROWED selection;
# this catches the other shape, where the selection is full and part of it quietly opts out.
if grep -qiE "skipped.*circuit-tracer not installed" /tmp/gpu_run.txt; then
  echo "ERROR: composition tests skipped because circuit-tracer is missing in the ONE lane"
  echo "that is supposed to provide it. The install step did not take."
  grep -iE "SKIPPED" /tmp/gpu_run.txt || true
  exit 1
fi
# AND NOTHING SKIPPED FOR WANT OF A GPU. The `cuda` marker skips when no device is visible, so a container started
# without its devices would pass every CPU test and skip the rest, green. The backend check makes that unlikely; this
# makes it impossible to miss.
if grep -qE "SKIPPED.*no CUDA device is visible" /tmp/gpu_run.txt; then
  echo "ERROR: GPU-marked tests skipped for want of a visible device on a GPU agent."
  exit 1
fi
