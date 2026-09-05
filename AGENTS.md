# AGENTS.md: it-interp-engine-adapter

Guidance for coding agents (Claude Code and others) working in this repository.
`CLAUDE.md` is a one-line redirect to this file, so this is the single canonical copy.

## What this repo is

A **hub-delivered Interpretune adapter** composing [interp-engine](https://github.com/decoderesearch/interp-engine)
into an Interpretune session. It is the first Interpretune adapter shipped as a hub component rather
than bundled in the framework, so it doubles as the **reference pattern** for user-contributed adapter
repos. Choices here get copied. Prefer the explicable option over the clever one.

Published (private) as `speediedan/it-interp-engine-adapter`. Consumers load it behind the
`IT_TRUST_REMOTE_CODE=1` trust gate, and loading executes the entrypoint in the caller's process.

**`it.hub.pull` alone is NOT enough, and this line used to say it was.** `pull` is manifest-first: with
no config key it fetches `it_component.yaml` and returns `(manifest, commit)`, caching nothing else.
`load_hub_adapter` then reads the entrypoint from the cached snapshot and raises "manifest declares
adapters entrypoint 'adapter.py', which is not present in the snapshot" -- accurate, and easy to read as
a broken publish rather than a missing fetch. `it.hub.load` does not close the gap either: it is
cache-only and hydrates a CONFIGURATION by key, which an adapters component does not have.

Interpretune has no API that fetches an adapters entrypoint from the Hub, so the middle step is a direct
`hf_hub_download` pinned to the commit the manifest resolved to:

```python
import os
from huggingface_hub import hf_hub_download
from interpretune.hub.adapters import load_hub_adapter
from interpretune.hub.components import IT_COMPONENTS_HUB_CACHE, pull_component_manifest

repo = "speediedan/it-interp-engine-adapter"
manifest, commit = pull_component_manifest(repo)                      # what `it.hub.pull(repo)` does
hf_hub_download(repo, manifest["adapters"]["entrypoint"], revision=commit,
                cache_dir=str(IT_COMPONENTS_HUB_CACHE))               # the step with no API
load_hub_adapter(repo)                                                # executes, with IT_TRUST_REMOTE_CODE=1
```

Verified against the published revision, which then registers `interp_engine` and reproduces the seam's
measurements. Until the fix below reaches a release this repo floors on, a reader following the one-line
version gets an error naming the wrong problem.

**This is fixed in speediedan/interpretune#459**, which makes a key-less `it.hub.pull` materialize the
payloads the manifest declares -- the adapters entrypoint among them -- pinned to the resolved commit.
Loading still never downloads, and the error above remains the refusal for a genuinely partial snapshot.
**Delete the middle step and this whole note once #459 is in a release this repo floors on**; the floor
is `interpretune[conformance]`, so it arrives with the next one. Kept until then because consumers today
run against released interpretune, where the three-step sequence is what works.

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

**There is exactly one legitimate way past that rule, and it is narrow.** `_Seam` also carries
`_MEASURED_ALIASES` and `_MEASURED_GLOBAL_ALIASES`, which map a name the engine has no spelling for onto
the point that holds the tensor. They look like the thing the paragraph above forbids and are its
opposite, and the difference is not a matter of judgement:

- `_REFUSED` names tensors that **differ**. Substituting one is answering a question nobody asked.
- The alias tables name tensors **measured identical**, at `max|d| = 0.0`, against the semantics of the
  name being aliased rather than against a nearby tensor that seemed close. `blocks.N.hook_in` is what a
  bridge resolves to a forward pre-hook on block N; captured that way it equals `resid_pre.N` at 0.0 and
  `blocks.N-1.hook_out` at 0.0 with cos 1.000000. `unembed.hook_in` equals `final_norm` at 0.0.

An addition to those tables owes three things, and a session that cannot produce all three should refuse
instead: **a measurement against the aliased name's own semantics**, not an argument that the two
obviously mean the same thing; **a guard on the architecture the measurement does not cover** (the block
input alias refuses on hyper-connection models, because `resid_pre` is only a block's input where there
is one residual stream to be pre of); and **an upstream report**, because an alias table in a consumer is
how a consumer eventually maps something it should have refused. Both current entries are reported
against `decoderesearch/interp-engine`; delete them when it grows the spellings.

`TestMeasuredAliasesAreNotTheRefusals` asserts the two tables stay disjoint and that the pre-norm
refusal is still a refusal, so an alias table growing into the refusals fails a test rather than shipping.

**The hook inventory is interpretune's vocabulary, not the engine's.** A callable `names_filter` is a
predicate over the names interpretune uses, so `_hook_vocabulary` enumerates
`HookNameResolver(architecture).supported_hooks`. Building it forward from the engine instead --
`EagerModel.points()` spelled through `point_to_tlens_hook` -- yields 252 names on gpt2 and looks
complete. It is not: the engine's reverse table holds one canonical spelling per point while its forward
table accepts aliases, so `blocks.5.hook_resid_post` is in that inventory and `blocks.5.hook_out` is not,
though both name the same tensor and interpretune uses the second. A predicate over the forward-built
inventory matched **none** of the conformance suite's five capture points, and nothing in the failure
pointed at spelling.

Which names carry a layer comes from `ComponentMap.block_components()` / `.global_components()`, and
that is the second half of the same lesson. The first version avoided reading the resolver's private
mapping by trying each base name bare AND layer-qualified and keeping whatever `resolver.resolve`
accepted. `resolve` accepts `blocks.5.unembed.hook_out` and a bare `attn.hook_z`, so the inventory
filled with names that name nothing: harmless at 13 base names, and at 34 (after
speediedan/interpretune#453) it was 252 junk entries out of 433, which made four of five ordinary
filters refuse. **Probing an API to see what it accepts measures its permissiveness, not the question
you asked**, and the gap grows silently when the vocabulary does. Avoiding a private attribute was
right; the fallback was not, and the public answer existed.

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
which requires interpretune at or after commit **`b1837d2`** (the hub-rails merge). Note that PR speediedan/interpretune#398's
squash subject cites the issue it closes, `speediedan/interpretune#125`, rather than the PR, so grepping the log for `speediedan/interpretune#398`
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
  - **Qualify every issue citation with its repo.** GitHub renders a bare `#401` as a link to the
    CURRENT repository, so a bare number here resolves to an issue in THIS repo -- which either does not
    exist or exists and is unrelated. Every citation in this repo means an `interpretune` issue, so
    write `speediedan/interpretune#401`. A wrong link that renders is worse than no link, because it
    looks checked. Verified against every citation in the tree during the pre-flip docs audit.
  - **Check the reader can reach it.** Cite only repos a reader can actually open. Everything this repo
    references is public (`interpretune`, `circuit-tracer`, `skills`); the maintainer's admin repo is
    not, so name it as private and describe the command rather than pointing at a path only one machine
    has. The rule generalizes: if a citation is unreachable, it is exactly as useful as the workstream
    reference the rule above exists to replace.
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
