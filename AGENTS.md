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

`src/interp_engine_adapter.py` is run by the hub loader via `spec_from_file_location` straight out of
the snapshot, under a revision-scoped synthetic module name (`it_hub_adapters.<repo>.<revision>`), so
two cached revisions cannot collide in `sys.modules`.

Three consequences that are easy to violate:

- **It cannot import sibling modules from this repo.** There is no package context. Splitting the
  entrypoint into helper modules breaks hub loading while leaving `pytest` green, because the tests
  import the pip-installed form. If the file grows, it grows in place.
- **The loader never touches `sys.path`.** Hub consumers reach the module through
  `interpretune.hub.adapters.loaded_adapter_module(component)`, never `import interp_engine_adapter`.
  That plain import is for pip-install consumers and for this repo's own tests only.
- **`it_component.yaml`'s `adapters.entrypoint` must track the file's path.** A subdirectory path is
  supported (the loader resolves `snapshot / entrypoint` verbatim); a stale one fails only at load.

## Packaging: the src layout needs both halves stated

A single top-level *module* under `src/` is not auto-discoverable, because setuptools' src-layout
autodiscovery finds *packages*. `pyproject.toml` therefore states `package-dir = {"" = "src"}` **and**
`py-modules = ["interp_engine_adapter"]`. Drop either and `pip install -e .` builds a distribution
with no importable module, which surfaces as an ImportError in the tests rather than at install time.

Verify a packaging change by building, not by importing from the working tree:

```bash
python -m pip wheel . --no-deps -w /tmp/iea && python - <<'PY'
import zipfile, glob; print(zipfile.ZipFile(glob.glob('/tmp/iea/*.whl')[0]).namelist())
PY
```

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
source; `transformer-lens` and `sae-lens` are hand-carried because interpretune's core config imports
both eagerly even though this adapter uses neither (tracked in speediedan/interpretune#401). See
`.github/workflows/ci.yml` for the exact install line.

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
interpretune installed from `main` (or the pinned fix branch until #404 merges); this repo's CI does
exactly that, so a green CI run beside a local red is strong evidence the local env is the problem.

## CI

`.github/workflows/ci.yml`, two jobs, and the shape is itself part of the reference pattern:

- **`cpu`**: hosted runner, every push and PR. The gate that matters today.
- **`gpu`**: `workflow_dispatch` only, `runs-on: [self-hosted, gpu]`. Deliberately inert: it never
  runs and never blocks a merge unless a contributor registers a GPU runner. **There is no GitHub
  Actions self-hosted runner on the maintainer's host**, so this job does not exercise anything here
  today. If it is ever pointed at real shared hardware it must take that host's GPU lease first, or it
  will contend silently with other suites. See the `gpu-lease` skill.

Both jobs currently pin `interpretune @ git+...@fix/declare-jsonschema-dep` rather than `@main`.
Three call sites, flipped together when speediedan/interpretune#404 merges; the workflow comments
carry the condition.

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

`.claude/skills/` carries `gpu-lease`, `az-pipelines-debug` and `debug-remote-ci-failures`. The first
two are shared near-verbatim with `interpretune` and `finetuning-scheduler` and are deliberately
**host-independent**: real host values live in the maintainer's private `CLAUDE.local.md`, not here.
Keep them that way, so the shared copies stay diffable.
