#!/usr/bin/env bash
# Install this package and its interpretune/engine dependencies into the job image's venv (run inside the GPU job's
# container through ctr.sh, which activates that venv). Moved out of gpu-tests.yml unchanged in substance when the job
# became any-vendor: a host step now runs each command inside the container, and a script carries these heredocs
# through `docker exec` without a second layer of quoting.
set -e
echo "=== interpreter ==="
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'hip', torch.version.hip, 'available', torch.cuda.is_available())"
echo "=== install ==="
# NO pip BOOTSTRAP, and NO `--system`. Both were wrong in the first revision of this pipeline and the image's own shape
# is why (measured on this pool's image, and confirmed against interpretune's dockers/base-cuda/Dockerfile):
#
#   PATH        /tmp/venvs/it_dev/bin:...:/usr/local/bin
#   VIRTUAL_ENV /tmp/venvs/it_dev
#
#   * The active interpreter is a `uv venv`, and uv does not put pip in one. So `python -m pip ...` fails with
#     "No module named pip" before doing anything, which is exactly how the first run of this pipeline died.
#   * `uv` is already installed system-wide at /usr/local/bin by the image, so there is nothing to bootstrap.
#   * `uv pip install --system` would target /usr/lib/python3.13/site-packages, but the image's torch (CUDA or ROCm)
#     lives in the VENV. That would have resolved a SECOND torch alongside the one this job exists to exercise. Bare
#     `uv pip install` honours VIRTUAL_ENV and installs beside it.
#
# Mirrors .github/workflows/ci.yml's install otherwise, with ONE deliberate difference: no `--torch-backend=cpu`. The
# image's GPU torch is the point of running here, and forcing the CPU variant would silently make every GPU assertion
# vacuous.
uv --version
#
# NO transformer-lens / sae-lens hand-carry. They were carried until speediedan/interpretune#401's follow-up (merged as
# c3423a5) declared jaxtyping and moved the analysis stack's module-level sae_lens import behind TYPE_CHECKING.
# transformer-lens had been silently supplying jaxtyping, which is why the pair had to be dropped together and why no
# in-repo env could reveal it. If a future interpretune reintroduces an undeclared import, this job fails at import
# rather than being masked by a pin here, which is the outcome worth having.
#
# VENNGRAM IS REQUIRED BY interpretune AND IS ON NO INDEX YET. interpretune's base `dependencies` name
# `venngram>=0.1.0.dev0`, which resolves only from git, and its git pin lives in its dependency groups, which a git
# install does not carry (as with circuit-tracer below). Read it the same way, from the interpretune `main` this job
# installs, so the two cannot drift. Delete this once venngram is on PyPI.
VG_REQ=$(python - <<'PY'
import tomllib, urllib.request, sys
url = "https://raw.githubusercontent.com/speediedan/interpretune/main/pyproject.toml"
data = tomllib.loads(urllib.request.urlopen(url, timeout=30).read().decode())
entries = sorted({e for g in data.get("dependency-groups", {}).values()
                  for e in g if isinstance(e, str) and e.startswith("venngram @")})
if len(entries) != 1:
    sys.exit(f"expected exactly one venngram pin in interpretune's dependency-groups, "
             f"found {len(entries)}: {entries}")
print(entries[0])
PY
)
echo "venngram: ${VG_REQ}"
uv pip install -e ".[test]" \
  "interpretune[conformance] @ git+https://github.com/speediedan/interpretune.git@main" "${VG_REQ}"
# finetuning-scheduler as interpretune pins it: the tuning conformance (#48) needs its views, which no release carries
# yet. Derived, never copied; `scripts/interpretune_git_pin.py` says when this goes.
uv pip install "$(python scripts/interpretune_git_pin.py finetuning-scheduler)"
#
# CIRCUIT-TRACER, DERIVED FROM INTERPRETUNE'S OWN PIN RATHER THAN RESTATED HERE.
#
# The `(core, interp_engine, circuit_tracer)` composition's tests -- the pairing class, its MRO, and the guard that
# keeps this adapter's hook refusal alive inside the composition -- skip when circuit-tracer is absent. Both CI lanes
# lacked it, so the NEWEST and least-covered code in this repo was running nowhere. Measured on the merge build for
# that composition: 55 collected, 47 passed, 8 skipped.
#
# It cannot come along with interpretune: interpretune carries it in a PEP 735 `dependency-group` (`git-deps`), and
# groups do not transit a git install the way an extra would.
#
# It also cannot become an extra while it is a GIT pin: interpretune publishes to real PyPI, and PyPI rejects
# direct-reference (git URL) dependencies in published metadata. So `git-deps` is where a git URL is the only thing it
# CAN be, not a stylistic choice to be revisited.
#
# So it is READ from interpretune's `pyproject.toml` at the ref this job installs, never copied. Copying the SHA would
# re-create exactly the hand-carry this repo removed with transformer-lens and sae-lens: a second pin that looks
# authoritative, drifts silently, and is only discovered when the two disagree. Derived, it is right by construction
# and follows interpretune's bumps.
#
# THIS BLOCK HAS A DEFINED END, and it is written down so it does not quietly become permanent. When circuit-tracer's
# two upstream PRs land and interpretune retires the fork pin for a VERSION pin, that pin can live in an ordinary
# extra, an extra transits a git install normally, and this whole derivation should be DELETED in favour of installing
# that extra. A workaround that outlives its reason is indistinguishable from architecture to whoever reads it next.
CT_REQ=$(python - <<'PY'
import tomllib, urllib.request, sys
url = "https://raw.githubusercontent.com/speediedan/interpretune/main/pyproject.toml"
data = tomllib.loads(urllib.request.urlopen(url, timeout=30).read().decode())
entries = [e for g in data.get("dependency-groups", {}).values()
           for e in g if isinstance(e, str) and e.startswith("circuit-tracer")]
if len(entries) != 1:
    sys.exit(f"expected exactly one circuit-tracer entry in interpretune's dependency-groups, "
             f"found {len(entries)}: {entries}")
print(entries[0])
PY
)
# RECORD WHAT WE ACTUALLY RESOLVED. This gate installs `interpretune @ main`, a MOVING reference, so its result is only
# meaningful against the interpretune commit that was `main` at install time -- and nothing else in the build captures
# that. Measured 2026-09-07: a gate here finished eight minutes before three interpretune PRs merged, and afterwards the
# only way to tell what it had validated was comparing timestamps against merge times.
#
# A recorded commit turns that inference into a lookup, and it is the whole cost of the fix.
python - <<'PY'
import json, importlib.metadata as md
try:
    info = json.loads(md.distribution("interpretune").read_text("direct_url.json"))["vcs_info"]
    print(f"RESOLVED interpretune: {info['commit_id']} (requested {info.get('requested_revision')})")
except Exception as exc:  # a non-VCS install has no direct_url; say so rather than fail the gate
    print(f"RESOLVED interpretune: <no direct_url> ({type(exc).__name__}) version {md.version('interpretune')}")
PY
echo "circuit-tracer, as interpretune pins it: ${CT_REQ}"
uv pip install "${CT_REQ}"
echo "=== resolved, and from WHERE (a version string would not show a second torch) ==="
python -c "import torch, interpretune, interp_engine_adapter as a; print(torch.__file__); print(interpretune.__file__); print(a.__file__)"
# THE TORCH THE JOB RUNS MUST STILL BE THE IMAGE'S. An install above that pulled a different torch would replace the
# image's CUDA or ROCm build, and on an AMD agent a CUDA wheel sees no device at all; the backend check after this step
# would then fail with a less direct message.
python -c "import os, torch; v = os.environ['IT_GPU_VENDOR']; a = 'amd' if torch.version.hip else 'nvidia' if torch.version.cuda else 'cpu'; assert a == v, (v, a, torch.__version__); print('torch backend', a, torch.__version__)"
# ASSERT THE COVERAGE EXISTS rather than inferring it from the install having been attempted. If circuit-tracer stops
# resolving, the composition tests go back to skipping and the build stays GREEN while covering less -- the precise
# failure this step was added to end.
python -c "import circuit_tracer; print('circuit_tracer', circuit_tracer.__file__)"
