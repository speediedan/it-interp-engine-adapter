# AGENTS.md: it-interp-engine-adapter

Guidance for coding agents (Claude Code and others) working in this repository.
`CLAUDE.md` is a one-line redirect to this file, so this is the single canonical copy.

## What this repo is

A **hub-delivered Interpretune adapter** composing [interp-engine](https://github.com/decoderesearch/interp-engine)
into an Interpretune session. It is the first Interpretune adapter shipped as a hub component rather
than bundled in the framework, so it doubles as the **reference pattern** for user-contributed adapter
repos. Choices here get copied. Prefer the explicable option over the clever one.

Consumers load it with `it.hub.pull("speediedan/it-interp-engine-adapter")` behind the
`IT_TRUST_REMOTE_CODE=1` trust gate. Loading executes the entrypoint in the caller's process.

## The entrypoint is executed as a single file, and that constrains everything

`src/interp_engine_adapter/adapter.py` is run by the hub loader via `spec_from_file_location` straight
out of the snapshot, under a revision-scoped synthetic module name (`it_hub_adapters.<repo>.<revision>`),
so two cached revisions cannot collide in `sys.modules`.

**`adapter.py`, not `__init__.py`, is the entrypoint, and the directory is a package only for the pip
path.** `build_component_tree` stages exactly what the manifest names plus the manifest itself, so the
published snapshot holds `adapter.py` and `it_component.yaml` and no `__init__.py`: the loader has no
sibling available to import even if the entrypoint asked for one. Verified by staging and executing it.

Three consequences that are easy to violate:

- **The entrypoint cannot import sibling modules from this repo.** There is no package context at hub
  load. Splitting it into helper modules breaks hub loading while leaving `pytest` green, because the
  tests import the pip-installed form. If the file grows, it grows in place. `__init__.py` *does* import
  a sibling, and that is not an exception to this rule: the loader never executes it, only pip consumers
  do, and package context is real on that path.
- **The loader never touches `sys.path`.** Hub consumers reach the module through
  `interpretune.hub.adapters.loaded_adapter_module(component)`, never `import interp_engine_adapter`.
  That plain import is for pip-install consumers and for this repo's own tests only.
- **`it_component.yaml`'s `adapters.entrypoint` must track the file's path.** A subdirectory path is
  supported (the loader resolves `snapshot / entrypoint` verbatim); a stale one fails only at load.

## Packaging: the manifest ships as package data

`src/interp_engine_adapter/` is a package, declared with `package-dir = {"" = "src"}` and
`packages = ["interp_engine_adapter"]`, and `it_component.yaml` ships inside it through `package-data`.
There is exactly **one** copy of that manifest in the repo. If you need it somewhere else, move it and
update `adapters.entrypoint`; do not add a second copy, because drift between a root manifest and a
shipped one fails only at hub load, which nothing in CI reaches.

**Do not ship the manifest with `[tool.setuptools.data-files]`.** Measured: that lands it at the install
root (`<prefix>/it_component.yaml`), so every component publishing a manifest that way overwrites the
last one installed, and a lookup answers plausibly for a different component entirely.

Verify a packaging change by building, and **delete `build/` first**. setuptools does not prune that
tree between builds, so after a layout change it holds the old module *and* the new one, and the wheel
contains a file that no longer exists in the source. That is a stale artifact reading as a real result:

```bash
rm -rf build dist src/*.egg-info
python -m pip wheel . --no-deps -w /tmp/iea && python - <<'PY'
import zipfile, glob; print(zipfile.ZipFile(glob.glob('/tmp/iea/*.whl')[0]).namelist())
PY
```

The wheel must contain `interp_engine_adapter/adapter.py`, `interp_engine_adapter/__init__.py` and
`interp_engine_adapter/it_component.yaml`, and **no top-level `interp_engine_adapter.py`**.

`pytest` sees only the pip path (`TestPipInstallSurface`). Nothing in CI exercises the hub path, so
verify that by hand after any layout or manifest change: stage with
`interpretune.hub.publish.build_component_tree`, then execute the staged entrypoint by path under a
synthetic module name and confirm a `vars()` scan finds `InterpEngineAdapter`.

## The seam, and the refusal that is the point of this adapter

Every `interp_engine` symbol is reached through `_Seam`. That is version insulation (interp-engine's
1.x minors change signatures with no deprecation aliases, deliberately) and it is also where the
hook-name discipline lives.

**Do not "fix" `_Seam`'s refusals into mappings.** TransformerLens fires `blocks.{i}.hook_mlp_in` on
the residual stream *before* the block norm, while the sublayer's actual argument is the norm's
*output*. On gemma-3-1b-it layer 5 those tensors share cosine similarity **0.088**, and published SAE
dashboards have been generated from the wrong one. `_Seam.point_for()` raises `UnmappableHookError`
naming what to ask for instead. A future session will read that refusal as an unhandled case and be
tempted to resolve it to the tensor the caller probably meant. That substitution is the defect class
this adapter exists to not propagate. Extend the refusal's message; do not remove the refusal.

## Correctness lives here, not upstream

Interpretune's core suite owns the **rails** (that the `adapters` kind loads, gates and registers,
tested against a fixture adapter) plus one optional contract test against the published component. It
does **not** test this adapter's behavior. So an interp-engine release that moves a signature must turn
*this* repo red rather than a framework nobody expects to break. New behavior gets a test in
`tests/test_seam.py`.

## Testing

```bash
pytest -v
```

The suite needs `interpretune` importable. It is pre-MVP and not on PyPI, so it is installed from
source. **Nothing else is hand-carried any more.** `transformer-lens` and `sae-lens` were carried
until speediedan/interpretune#401's follow-up (merged as `c3423a5`), because `interpretune.config`
reaches the analysis stack, which imported `sae_lens` at module level for an annotation and annotated
with `jaxtyping` without declaring it, while `transformer-lens` silently supplied that `jaxtyping`.
That masking is why the pair had to go together and why no environment inside this repo could have
revealed it. If a future interpretune reintroduces an undeclared import, CI here now fails at import
rather than being masked by a pin, which is the outcome worth having.

**A local red naming a missing `interpretune` symbol is usually a stale environment, not a missing
API.** `TestRegistration::test_registers_both_declared_compositions` needs `register_dynamic_adapter`,
which requires interpretune at or after commit **`b1837d2`** (the hub-rails merge). Note that PR #398's
squash subject cites the issue it closes, `#125`, rather than the PR, so grepping the log for `#398`
finds nothing. Check what you actually resolved before concluding anything about the API:

```bash
python -c "import interpretune; print(interpretune.__file__)"
git -C ~/repos/interpretune rev-parse --abbrev-ref HEAD    # if that path is what resolved
```

The failure mode is specific and easy to walk into: on a machine that keeps an editable interpretune
checkout, both usual resolution paths reach the same shared working tree, so a branch someone else has
checked out there becomes *your* interpretune. `env -u PYTHONPATH` does not save you: an editable
install puts that tree on `sys.path` through `site-packages` regardless. Reproduce green with an
interpretune installed from `main`; this repo's CI does exactly that, so a green CI run beside a local
red is strong evidence the local env is the problem.

## CI

`.github/workflows/ci.yml`, two jobs, and the shape is itself part of the reference pattern:

- **`cpu`**: hosted runner, every push and PR. The gate that matters today.
- **`gpu`**: `workflow_dispatch` only, `runs-on: [self-hosted, gpu]`. Deliberately inert: it never
  runs and never blocks a merge unless a contributor registers a GPU runner. **There is no GitHub
  Actions self-hosted runner on the maintainer's host**, so this job does not exercise anything here
  today. If it is ever pointed at real shared hardware it must take that host's GPU lease first, or it
  will contend silently with other suites. See the `gpu-lease` skill.

### The GPU gate is an Azure pipeline, not the `gpu` job above

`.azure-pipelines/gpu-tests.yml`, registered as definition **4**
(`speediedan.it-interp-engine-adapter.gpu`) in the **interpretune** Azure DevOps project. It runs in
interpretune's container image on the self-hosted `Default` pool, takes the host GPU lease, and cleans
up its own workspace so the next checkout can run. Operating it is `az-pipelines-ops`; classifying a
failure is `az-pipelines-failure-triage`.

**Cancel a queued build whose change cannot affect the result.** The pool has one agent, so a queued
build holds a place other work waits in, and the job takes the host GPU lease when it runs. A
comment-only or documentation change queues a build anyway, because `pr:` path filters match PATHS and
this repo's filters include `.azure-pipelines/**` and `src/**`: a one-word comment fix inside the
pipeline definition is indistinguishable to a filter from a rewrite of it. That judgement is the
merger's, not the filter's, so do not add excludes to compensate.

Declining to approve is **not** the same as cancelling: an unapproved build sits queued and its gate
becomes a trap for a batch approval. Cancel it, and check again after merging, because the merge
queues its own build for the same content. The `az-pipelines-ops` skill carries the command.

Both workflows and the pipeline track `interpretune @ git+...@main` and install nothing else by hand;
see Testing for why the two framework pins were removed.

## GPU work must be wrapped by hand

**This repo has no lease-aware test harness.** Some sibling repos carry one that self re-execs under
the host GPU lease, so a single acquisition covers a whole suite. Nothing here does, so every
GPU-touching command must be wrapped explicitly:

```bash
export GPU_LEASE_CMD=/path/to/gpu_lease.sh   # opt-in; a no-op when unset
$GPU_LEASE_CMD -- <cmd>
```

This matters less than it sounds today, because the suite is CPU-only and no test in it touches a
device. It matters the moment one does. See the `gpu-lease` skill for planning and recovery.

## Code style

120-char lines, ruff `E`/`W`/`F` with `E731`/`F722` ignored for jaxtyping compatibility, McCabe max 10,
`from __future__ import annotations`. Configured in `pyproject.toml` and enforced by pre-commit:

```bash
pre-commit run --all-files
```

## Repo conventions

- **Pre-MVP: no backwards-compatibility shims.** Interpretune's internal contracts change without
  deprecation notices. If one changes, update the callers and tests directly rather than adding
  fallback paths or silent coercions.
- **Shipped docs must not reference agent workstreams.** `README.md`, this file and docstrings outlive
  the work that produced them and are read by users. Park an open decision on an **open issue** and
  cite the number; a workstream reference is invisible to a reader and lost when the session closes.
  - **Check the reader can reach it.** This repo is PRIVATE and `interpretune` is PUBLIC, so an issue
    number here, cited from there, is exactly as unreachable as the workstream reference the rule
    exists to replace. When citing across that boundary, name the repo, say it is private, and carry a
    two-sentence summary of what the issue decides, so a reader without access learns the shape rather
    than meeting a pointer that reads as documented-elsewhere and is not. Revisit those summaries if
    this repo goes public.
- **No ephemeral CI references** (build numbers, run URLs, approval ids) in commit messages, code
  comments or anything else a future reader will meet without that context. Describe the environment
  shape and the symptom instead. Those ids belong in PR threads and workstream logs.
- **No AI-attribution trailers** in commit messages or PR bodies.
- **Cite a commit SHA when you mean a specific change.** These repos squash-merge, and a squash subject
  cites the *issue* it closes rather than the PR, so a PR number can be genuinely unfindable in the log,
  and a reader grepping for it concludes the citation is wrong. If a PR number is the useful reference,
  give the SHA beside it.
- **A PR that does not close its issue says so on the issue**, not only in the PR body: what landed,
  what the work obviated and why, and the remaining scope as a checklist. Once merged, a PR body sits
  behind a link nobody follows, and an issue reading as though nothing happened is indistinguishable
  from one where the remaining work is invisible.
- Before committing: `pytest -v` and `pre-commit run --all-files`.

## Skills

`.claude/skills/` carries four skills, and the distinction between them matters:

- **`gpu-lease`** and **`az-pipelines-ops`** are VENDORED from `speediedan/skills` and are byte-identical
  across this repo, `interpretune` and `finetuning-scheduler`. **Do not edit them here.** A local edit is
  discarded by the next `sync-shared-skills.sh push`, and a pre-commit hook refuses the commit to say so.
  Fix them upstream, then take the push.
- **`az-pipelines-failure-triage`** and **`debug-remote-ci-failures`** are this repo's own. The first is
  the repo-specific half of the Azure skill and points at `az-pipelines-ops` for anything operational;
  the second is deliberately forked in all three repos, because the shared content measured near zero.

The vendored pair are host-independent by construction: real host values live in the maintainer's
private `CLAUDE.local.md`, which is what keeps the copies diffable.
