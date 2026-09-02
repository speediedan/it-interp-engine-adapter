---
name: debug-remote-ci-failures
description: Debug this adapter's GitHub Actions failures - pulling run logs, distinguishing a hub-component defect from upstream interpretune drift or an interp-engine signature change, and running the fix/push/verify loop. Use when the `tests` workflow fails on a push or PR, or when CI fails in a way plain local pytest does not reproduce.
license: Apache-2.0
---

# Debugging remote CI failures

Covers the GitHub Actions workflow only. For the shared self-hosted Azure agent, use the
`az-pipelines-ops` skill (operational) and `az-pipelines-failure-triage` (this repo's half).

Adapted from the interpretune skill of the same name. **Most of that skill does not apply here** and
was dropped rather than carried: this repo has one workflow, two jobs, no OS matrix, no Windows or
macOS legs, no artifacts, and a 12-test suite that runs in seconds. What survives is the log-pulling
mechanics and a failure taxonomy specific to being a hub-delivered component.

## When to use this skill

- The `tests` workflow fails on a push or PR
- CI fails in a way local `pytest -v` does not reproduce
- A failure appeared without a change to this repo (suspect upstream drift first; see below)

## What the workflow actually is

`.github/workflows/ci.yml`, two jobs:

| Job | Trigger | Runner | Notes |
| --- | --- | --- | --- |
| `cpu` | every push and PR | `ubuntu-latest`, Python 3.12 | the gate that matters |
| `gpu` | `workflow_dispatch` only | `[self-hosted, gpu]` | **inert**: no such runner exists on the maintainer's host, so it never runs and never blocks |

`cpu` installs the pinned engine line plus `interpretune` **from source** (pre-MVP, not on PyPI), and
nothing else. There is no longer any hand-carry: see category B below for what an install failure
therefore means.

## Step 1: Pull the logs

There are no artifacts to download, so go straight to the run log.

```bash
gh run list --branch <branch> --limit 5
gh run view <run_id> --json jobs --jq '.jobs[] | {name, status, conclusion}'
gh run view <run_id> --log > /tmp/ci_<run_id>.log 2>&1

# Failing job only, when the combined log is noisy:
job_id=$(gh api "repos/speediedan/it-interp-engine-adapter/actions/runs/<run_id>/jobs" \
  --jq '.jobs[] | select(.conclusion=="failure") | .id')
gh api "repos/speediedan/it-interp-engine-adapter/actions/jobs/${job_id}/logs" > /tmp/job.log
```

**Read the failure summary line FIRST.** The loudest region of a pytest log is the traceback of the
first failure, which is often not the informative one. Start here:

```bash
grep -E "=+ .*(passed|failed|error).* =+" /tmp/ci_<run_id>.log
grep "^FAILED\|^ERROR" /tmp/ci_<run_id>.log | sort -u
```

**Report `collected` beside `passed`.** A green summary answers "did the tests I selected pass" and
reads as "did the tests pass". These differ whenever anything narrows selection, and the `gpu` job
sets `IT_RUN_CUDA_TESTS=1`, which in interpretune's conftest is a *selector* that replaces the
collected list rather than adding to it. If `selected != collected`, name the job you ran rather than
calling it a suite.

## Step 2: Categorize. Is this even ours?

This is the first question, because being a hub component means most of the surface belongs to someone
else. Work down this list in order; the first three are far more common than the last two.

### A. Upstream interpretune drift (most common)

Symptom: `ImportError` / `AttributeError` naming an `interpretune.*` symbol, with no relevant change in
this repo.

The `cpu` job installs interpretune from a git ref, so **the dependency moves under a repo that did not
change**. A run that passed yesterday can fail today from a push to interpretune.

```bash
# What did CI actually install?
grep -iE "interpretune.*(Cloning|Resolved|Building)" /tmp/ci_<run_id>.log
# Does the symbol exist in the version CI got?
python -c "import interpretune.adapters.registration as r; print(dir(r))"
```

**Rule out a stale local environment before calling it drift. It is the more common cause and it
mimics drift exactly.** Worked example, 2026-08-29: `tests/test_seam.py`'s import of
`register_dynamic_adapter` failed locally and was diagnosed as a missing interpretune API. It was not.
The symbol had landed on interpretune `main` hours earlier in commit `b1837d2`; the shared
`~/repos/interpretune` checkout was on an older feature branch, and both resolution paths on that
machine (the editable install and the repo `.env`'s `PYTHONPATH`) pointed at it. CI was green on the
same import the whole time.

The trap has teeth because `env -u PYTHONPATH` feels like it isolates and does not: an editable
install's `.pth` puts the same working tree on `sys.path` through `site-packages`. Two cheap checks
that would have caught it immediately:

```bash
python -c "import interpretune; print(interpretune.__file__)"
git -C ~/repos/interpretune rev-parse --abbrev-ref HEAD   # is that tree even on main?
gh run list --limit 3                                     # is CI red on this too, or just you?
```

**A green CI run beside a local red is evidence about your environment, not about the code.** If CI is
genuinely red on the same import, then it is drift. Do not fix it by weakening the test; raise it
with the `ie-integration` session or file it against interpretune.

### B. An undeclared dependency upstream

All three call sites now track `interpretune @ git+...@main` and install **nothing else by hand**. The
`transformer-lens` / `sae-lens` pins are gone (interpretune#401's follow-up, `c3423a5`), so this repo
no longer cushions interpretune's declaration gaps.

That is deliberate and it changes what an install failure means. Previously a missing declaration
upstream was masked here: `transformer-lens` silently supplied the undeclared `jaxtyping`, which is why
the two had to be dropped together and why no environment inside this repo could reveal the problem.
Now the same class fails loudly at import, in this job, on the commit that introduced it.

So an `ImportError` or `ModuleNotFoundError` at install or collection naming a package this repo never
asked for is **evidence about interpretune's declarations**, not about this repo. Confirm by installing
interpretune alone in a clean venv and importing the surface the entrypoint uses; if it fails there
too, the fix belongs upstream (the shape is interpretune#403 and #401) rather than in a pin here.
Re-adding a pin to make CI green would restore the masking that hid it.

### C. interp-engine minor-version signature change

Symptom: `TypeError` or `AttributeError` from an `interp_engine` symbol.

`interp-engine`'s 1.x minors change signatures with **no deprecation aliases**, by design. The pin is
`~=1.3.0` (`>=1.3.0, ==1.3.*`) precisely to prevent this; note the trailing `.0`, because a bare `~=1.3` would
admit 1.4, which is the break the pin exists to stop.

Every engine symbol is reached through `_Seam`, so a rename is a one-file change. Fix it there, and
only there: an engine call that bypasses the seam is the actual defect.

### D. Packaging / discovery

Symptom: `ModuleNotFoundError: interp_engine_adapter` after a successful install.

A single top-level module under `src/` is not auto-discoverable. `pyproject.toml` must carry **both**
`package-dir = {"" = "src"}` and `py-modules = ["interp_engine_adapter"]`. Drop either and the built
distribution contains no importable module, which surfaces as an ImportError at test time rather than
at install time. Reproduce by building, not by importing from the working tree:

```bash
python -m pip wheel . --no-deps -w /tmp/iea
python -c "import zipfile,glob; print(zipfile.ZipFile(glob.glob('/tmp/iea/*.whl')[0]).namelist())"
```

Also check `it_component.yaml`'s `adapters.entrypoint` still matches the file's path, because a stale value
fails only at hub load, which CI does not currently exercise.

### E. A genuine adapter defect

Only after A through D are ruled out. Add the regression test to `tests/test_seam.py` first: this repo owns
this adapter's correctness, and an engine release moving a signature is *supposed* to turn this repo
red rather than a framework nobody expects to break.

## Step 3: Fix, push, verify

```bash
git add -A && git commit -m "<what changed and why>"   # pre-commit may auto-fix; re-add and re-commit
git push
gh run list --branch <branch> --limit 3
gh run view <run_id> --json jobs --jq '.jobs[] | {name, status, conclusion}'
```

- **Pre-commit modifies files.** A first commit attempt can fail with hooks reporting "Failed" after
  fixing the files. Re-add and commit again; that is the hook working.
- **Pushing again cancels the in-flight run.** Batch changes rather than pushing per-fix.
- **Reproduce locally before pushing.** The suite is seconds long, so there is no excuse for using CI
  as the iteration loop. Mind the PYTHONPATH trap in `CLAUDE.local.md`: run
  `env PYTHONPATH="$PWD/src:$PWD" python -m pytest -v` to test the working tree deliberately, or
  `env -u PYTHONPATH` to isolate.

## Step 4: What not to do

- Do not weaken or skip a test to make CI green when the cause is upstream drift. The test is the
  signal; route it to interpretune.
- Do not resolve a `_Seam` refusal into a mapping to satisfy a failing case. Read the refusal's message
  first; see the hook-name section of `AGENTS.md`.
- Do not add backwards-compatibility shims for changed interpretune contracts. This ecosystem is
  pre-MVP; update the callers directly.
- Do not put build numbers or run URLs in commit messages or code comments. They identify a log that
  expires. Describe the environment shape and the symptom instead.
