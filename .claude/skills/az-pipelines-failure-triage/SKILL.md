---
name: az-pipelines-failure-triage
description: Triage a FAILING self-hosted Azure GPU build for this adapter - classify the failure as infrastructure, host-reboot damage, or test memory, and gather the agent-side evidence that classification needs. Use when a build ran and failed, or died before producing test output. For a build that has not STARTED (gated, unauthorized, or not dispatched), use az-pipelines-ops instead.
license: Apache-2.0
---

# Triaging a failed GPU build

The operational half of this skill lives in **`az-pipelines-ops`**, which is vendored from
`speediedan/skills` and shared verbatim with `interpretune` and `finetuning-scheduler`. This file is
the part that is specific to this repo and this host.

**Which one you want:**

| The build | Skill |
| --- | --- |
| has not started (gated, unauthorized resource, or never dispatched) | **`az-pipelines-ops`** |
| started and failed, or died before test output | this one |
| is queued behind a GPU lease | **`gpu-lease`** |

**Do not conclude "the agent is broken" from this file alone.** Everything below assumes
`az-pipelines-ops` Step 1 has already read the build timeline and found *no checkpoint records*. A
build blocked on a pending approval or an unauthorized resource looks identical from here, and the
approvals API reports neither reliably. That gate is stated rather than implied on purpose: an implied
gate protects the reader who arrives through the front door and not the one who skims in.

## Substitutions for `az-pipelines-ops`

That skill is written against `${ORG}` / `${PROJECT}`. For this repo:

```bash
ORG=https://dev.azure.com/speediedan
PROJECT=interpretune            # yes, interpretune: this adapter's definition lives in that project
AZ_PAT="$AZURE_DEVOPS_EXT_PAT"
```

The pipeline is `.azure-pipelines/gpu-tests.yml`, registered as definition **4**,
`speediedan.it-interp-engine-adapter.gpu`. It runs in interpretune's container image on the
self-hosted `Default` pool (queue **10**, pool **1**; those ids differ and the distinction has already
cost one wrong diagnosis).

> **Host-specific values live in `CLAUDE.local.md`, not here**, so the shared copies stay diffable.
> `$AGENT_HOME` and `$AGENT_UID` below are illustrative:
>
> ```bash
> AGENT_HOME=${AGENT_HOME:-/opt/az_pipeline_agent}
> AGENT_UID=${AGENT_UID:-998}
> ```

## Local tooling the shared half cannot name

`az-pipelines-ops` is repo-neutral, so it describes releasing a gate and registering a definition
without naming the tools, which live in the maintainer's **private** admin repo. Those paths belong
here or nowhere. The wrapper scripts below live in the maintainer's private admin repo; a
contributor without it uses `az pipelines` or the REST API directly, which is what the wrappers
call:

```bash
cd <admin-scripts>/az_pipeline_agent_scripts   # maintainer-only convenience wrapper
./manage-approvals.sh -o speediedan -p interpretune -m list
./manage-approvals.sh -o speediedan -p interpretune -m approve -i "<approval_id>" -c "<why>"
./manage-pipelines.sh -o speediedan -p interpretune -m show -i <definitionId>
```

**Identify each gate before releasing it** (`az-pipelines-ops` Step 2 gives the `$expand=steps` query).
This pointer carries the short form because a reader who arrives here for the script path would
otherwise have to reach What Not To Do at the bottom to meet the warning.

- **`-m reject-all` disposes of REAL pending runs, not only stale ones.** List first, always. There is
  no undo: rejecting a gate completes its build as `failed`.
- **`manage-pipelines.sh` CLONES an existing definition and cannot retarget the repository.** To create
  a definition for a different repo, use `az pipelines create` (see `az-pipelines-ops`). The clone path
  silently points the new definition at the template's repo, where a same-named YAML file makes it run,
  pass, and report green while testing something else.
- `restart-stack.sh` is reached via `sudo` and is authorized for agents; it is for a wedged agent, not
  for lease recovery.

## Step 1: Gather the agent-side evidence

`az-pipelines-ops` Step 3 is REST-only and deliberately does not touch the agent host. The
classification below needs the agent's own logs, so those commands live here:

```bash
tail -f "$AGENT_HOME"/_diag/Agent_*.log
ls -1t "$AGENT_HOME"/_diag/Worker_*.log | head
```

- Agent log shows only keepalive polling and no new worker log → the run never dispatched. That is an
  `az-pipelines-ops` question, not this one. Go back.
- A new worker log exists → read it, and classify below.

## Step 2: Classify the failure

The branches below are alternatives, not a sequence, so they are headings rather than numbered steps.
Only the gather-then-classify order is real.

### Infrastructure

Worker log starts, then dies before any test output. Agent log shows docker socket, SSL, or shutdown
issues. Agents are explicitly authorized to run:

```bash
sudo "$AGENT_HOME"/restart-stack.sh
```

**Two things a host reboot breaks, both in `/tmp`**, and both present as CI failures that look like the
PR's fault:

1. `/var/run/docker.sock` disappears. It is a tmpfs symlink to `/run/user/$AGENT_UID/docker.sock`.
   Symptom: `Initialize containers` fails with
   `Exit code 1 ... 'stat', arguments '-c %g /var/run/docker.sock'`.
2. The lease lock files disappear, or the directory is recreated empty. Symptom: the lease step fails
   with `is missing and cannot be created`. The container **cannot** create it: under rootless docker
   its uid is a host subuid that owns neither the file nor the directory, so it opens the lock
   read-only. The file must pre-exist on the host, agent-owned, mode **0666**. A mode of 0644 lets CI
   read the lock while making every host-side acquisition fail with `EACCES`.

The restart script provisions both, and a `tmpfiles.d` rule recreates them at boot.

### Install failed inside the container

The image's active interpreter is a `uv venv` at `/tmp/venvs/it_dev` with `VIRTUAL_ENV` set. Two
consequences, both of which have already broken this pipeline once:

- **`python -m pip ...` fails with "No module named pip".** `uv venv` does not put pip in an
  environment, and `uv` is already installed system-wide at `/usr/local/bin`. The pip bootstrap is
  correct on GitHub hosted runners, which is why it arrives here by reflex, and it can never work.
- **`uv pip install --system` targets the wrong site-packages.** The CUDA torch lives in the venv;
  `--system` installs beside `/usr/lib/python3.13/site-packages` and resolves a SECOND torch. That
  failure is silent and green. The `__file__` probe after the install step exists to catch it, because
  a version string cannot distinguish one torch from two.

### Test memory

Step exits `137`, or the job receives a shutdown signal during a heavy phase. Exit `137` on this host
is usually host RAM rather than GPU memory: swap is very small relative to RAM (see `CLAUDE.local.md`).

Before changing memory limits, confirm nothing else was running. A local full suite and this job
contend on host RAM and can OOM-kill both sides, with the local run dying silently mid-progress. A
`CUDA_VISIBLE_DEVICES=''` local run is **not** exempt, because the GPU lease does not serialize
CPU-only work.

### It ran green but tested the wrong thing

Not a failure class the build reports, and the one worth checking when a result seems too easy:

- The install resolved a second torch (see above). Read the `__file__` probe output.
- `IT_RUN_CUDA_TESTS=1` is a **selector** in interpretune's conftest, not a gate that un-skips tests.
  This repo has no conftest today, so both counts match, and that changes the moment one is added.
  Compare the collection count printed by the test step against the number that ran.

## What not to do

- **Do not restart the agent** until `az-pipelines-ops` Step 1 has ruled out both a pending approval
  and an unauthorized resource. This file's Infrastructure branch is reachable only after that, and the
  gate is restated here for anyone who arrived some other way.
- **Do not force-reset a GPU lease held by CI**, and never restart the agent to free one. See
  `gpu-lease`. Container teardown already releases it, so cancel **the run that HOLDS the lease** if it
  genuinely must be stopped.

  Note this does not conflict with `az-pipelines-ops`'s "do not treat a long wait as a hang". That rule
  is about **your own queued** build, which should be left to wait; this one is about a **different,
  running** build whose lease you were about to force. The two read as contradictory only if the
  subject is dropped, so the subject is stated in both.
- **Do not disable GPU coverage** to avoid an OOM. Confirm the phase split and fixture retention first.
- **Do not `manage-approvals.sh -m reject-all` without listing first.** It disposes of genuinely
  pending runs alongside stale ones, and a rejected gate completes its build as `failed` with no undo.
- **Do not register a definition for a different repo by cloning one.** `manage-pipelines.sh` copies
  the template's repository block wholesale and has no flag to retarget it, so the result runs green
  against the wrong repository. Use `az pipelines create`.
- **Do not put build numbers or run URLs** in commit messages or code comments. They identify a log
  that expires. Describe the environment shape and the symptom.
