---
name: az-pipelines-debug
description: Operate and debug the shared self-hosted Azure DevOps GPU agent - PAT-backed approval release, queue triage, worker dispatch checks, agent-stack recovery, and registering a new pipeline definition. Use when a gated build is stuck notStarted, when the agent is online but dispatching nothing, when a GPU step exits 137, or when this repo needs an Azure pipeline definition of its own.
license: Apache-2.0
---

# Debugging the shared Azure GPU pipeline

**This repo has no Azure pipeline definition today.** Its CI is GitHub Actions
(`.github/workflows/ci.yml`). For that, use the `debug-remote-ci-failures` skill.

This skill covers the **shared self-hosted agent** that `interpretune` and `finetuning-scheduler` run
their GPU pipelines on, because that agent shares this host's GPUs with any local work done here, and
because registering a definition for this repo is the obvious way to give it a real GPU leg.

> **Host-specific values live in `CLAUDE.local.md`, not here.** This skill is deliberately
> host-independent so the copies in the three repos stay diffable. Agent hostname, RAM/swap, GPU
> models, install directory and the agent's uid all vary by machine, so the commands below use
> `$AGENT_HOME` / `$AGENT_UID`:
>
> ```bash
> AGENT_HOME=${AGENT_HOME:-/opt/az_pipeline_agent}   # example default
> AGENT_UID=${AGENT_UID:-998}                        # example; the uid the agent runs as
> ```

## When to use this skill

- A GPU Azure build stays `notStarted` after a PR becomes ready for review
- The agent is online but no worker log is created for a queued build
- A GPU step exits `137`, or receives a shutdown signal under load
- Local GPU work here is contending with a pipeline job
- This repo needs a pipeline definition registered

## The agent shares this host's GPUs with you

The agent runs **one Azure job at a time**, so two pipeline runs never collide. The real risk is a
pipeline job landing on top of **local** work in this repo, or vice versa.

Azure GPU jobs bind-mount the lease directory into the container and hold the host `gpu` lease for
their duration (`Acquire host GPU lease`, released in an `always()` cleanup step). `flock` operates on
the inode, so the mounted lock file interlocks with host processes with **no change to the agent
installation**. It **fails open**: if the directory is not mounted, the job runs unserialized.

Nothing in this repo self re-execs under the lease, so local GPU work here must be wrapped by hand.
See the `gpu-lease` skill.

### ⛔ Never reset a lease held by CI, and never restart the agent to free one

`gpu_lease.sh --reset --force` **kills the holder process**. For a CI job that is both useless and
dangerous: the holder pid lives in the job container's PID namespace, so `--force` either fails or
kills an unrelated host process sharing that pid number. `--status` marks these with `[container]` and
`project=azure-<buildId>`, so treat either as read-only. It is also unnecessary: container teardown kills
every process inside the job and the kernel releases the lease. There is no stale-lock path.

**The host and pool are shared across three projects**, so an unexpected lease may legitimately belong
to another project's job or local suite. Check `project=` before assuming it is stale.

| Situation | Do this |
| --- | --- |
| Lease held by a CI job you want to stop | **Cancel the pipeline run.** Teardown frees the lease. |
| Lease held by another project | Leave it. Wait or coordinate; do not reset. |
| CI job timed out waiting for the lease | A genuine conflict. Let the local run finish and re-queue. |
| `--status` flags an anomaly | `gpu_lease.sh --doctor`, then plain `--reset` (free leases only). |
| Genuinely wedged **local** run | `--reset --force` is appropriate here. |

`restart-stack.sh` is for a wedged agent, not for lease recovery. Restarting it mid-job strands the
run without releasing anything the kernel would not have released anyway.

## Step 1: Verify auth, build state, and the approval gate

```bash
printenv AZURE_DEVOPS_EXT_PAT | wc -c
az pipelines build show --id <build_id> --organization https://dev.azure.com/<org> --project <project> -o table
curl -sS -u ":${AZURE_DEVOPS_EXT_PAT}" \
  "https://dev.azure.com/<org>/<project>/_apis/pipelines/approvals?state=pending&api-version=7.1-preview.1"
```

**For a `notStarted` build, read the TIMELINE first.** It is the only source that distinguishes the
three cases, and two of them are invisible to the approvals API:

```bash
curl -sS -u ":${AZURE_DEVOPS_EXT_PAT}" \
  "$ORG/$P/_apis/build/builds/<id>/timeline?api-version=7.1" | python3 -m json.tool | grep -A2 Checkpoint
```

| Timeline shows | Meaning | Do this |
| --- | --- | --- |
| `Checkpoint.Approval: pending` | human gate | approve with `manage-approvals.sh` |
| `Checkpoint.Authorization: inProgress` | a protected resource is not authorized for this definition | someone with permission must Permit it; see below |
| no checkpoint records at all | genuine queue or dispatch problem | only now inspect backlog, worker dispatch, agent health |

**A first run of a NEWLY CREATED definition lands in the second case every time**, because a cloned
definition inherits its resource authorizations and a fresh one does not. Measured 2026-08-30 on
definition 4's build 951: the approvals API returned `count: 0` and `manage-approvals.sh -m list`
printed "No pending approvals found" for a build that was very much blocked, while the timeline
carried `Checkpoint.Authorization: inProgress`. Check `pipelinePermissions` for the pool queue to
confirm, and note the queue id is not the pool id (`queues?api-version=7.1-preview.1` maps them; here
queue 10 is pool 1 `Default`). Once the resource is permitted, the human gate materializes normally.

### The queries under-report, in the dangerous direction

- **`az pipelines build list` defaults to COMPLETED builds.** A gated build is therefore invisible to
  exactly the scan you would reflexively run to find it. Pass `--status notStarted`, or query the
  approvals endpoint. A pending gate on an ESTABLISHED pipeline shows in the approvals API and not in
  a build-list scan. Do not extend that to "the timeline finds nothing": an earlier revision of this
  file said exactly that, and it is false for the authorization case above, where the timeline is the
  only place the block appears.
- **Always check `sourceBranch` and `sourceVersion`, not just the build number**, before calling a
  commit validated. Earlier successful builds are easy to mistake for the one you are waiting on.
- **Multiple test phases are Tasks inside ONE Job.** A job-level watch shows a single record, so "the
  build succeeded" is not evidence that each phase succeeded. Verify at task level.

All three are the same class: **the command answered a narrower question than the one being asked, and
the narrow answer looked like the broad one.** Re-running a narrow query more carefully returns the
same narrow answer with more confidence, so the fix is never "check again". State the command's
contract, then ask whether that is your question.

## Step 2: Release (or reject) the queued run

Preferred, the multi-mode helper in the private admin repo
(`project_admin/shared_admin_scripts/az_pipeline_agent_scripts/manage-approvals.sh`). It reads
`ADO_MCP_AUTH_TOKEN` or `AZURE_DEVOPS_EXT_PAT` and supports `list`, `approve`, `reject`,
`approve-all`, `reject-all`:

```bash
./manage-approvals.sh -o <org> -p <project> -m list
./manage-approvals.sh -o <org> -p <project> -m approve -i "<approval_id>" -c "<why>"
./manage-approvals.sh -o <org> -p <project> -m reject-all    # dispose stale pending gates
```

Rejecting a gate completes the build as `failed`, which is the terminal state for a rejected approval,
not a script error.

**Approving dispatches a memory-heavy job to the host immediately.** Before approving, confirm no full
local suite is running: a local suite and the CI container's suite contend on host **RAM**, not GPU,
and can OOM-kill both sides. The local run dies silently mid-progress (frozen log, no summary, easily
misread as a hang) while the build completes `failed`. A `CUDA_VISIBLE_DEVICES=''` local run is **not**
exempt, because the GPU lease does not serialize CPU-only work.

Fallback, approve directly:

```bash
curl -sS -X PATCH -u ":${AZURE_DEVOPS_EXT_PAT}" -H "Content-Type: application/json" \
  -d '[{"approvalId":"<approval_id>","status":"approved","comment":"<why>"}]' \
  "https://dev.azure.com/<org>/<project>/_apis/pipelines/approvals?api-version=7.1-preview.1"
```

## Step 3: Monitor dispatch and runner activity

```bash
tail -f "$AGENT_HOME"/_diag/Agent_*.log
ls -1t "$AGENT_HOME"/_diag/Worker_*.log | head
az pipelines agent list --organization https://dev.azure.com/<org> --pool-id <id> -o table
```

- Agent log shows only keepalive polling, no new worker log → still blocked upstream.
- New worker log appears → switch to it immediately for step-level detail.
- Agent offline or disabled → fix that before editing the pipeline or the tests.

## Step 4: Triage the failure class

**Queue / approval**: build stays `notStarted`, no worker log. Read the timeline (Step 1) before
concluding anything. `Checkpoint.Approval: pending` means approve first. `Checkpoint.Authorization:
inProgress` means a resource needs permitting and the agent is healthy. **An empty approvals API is
not evidence the agent is at fault** -- that inference is what routes a permissions problem into an
agent restart, and it is wrong for every newly created definition's first run.

**Infrastructure**: worker log starts then dies before test output; agent log shows docker socket,
SSL, or shutdown issues. Agents are explicitly authorized to run:

```bash
sudo "$AGENT_HOME"/restart-stack.sh
```

Two things a host **reboot** breaks, both in `/tmp`, and both present as CI failures that look like the
PR's fault:

1. `/var/run/docker.sock` disappears, a tmpfs symlink to `/run/user/$AGENT_UID/docker.sock`. Symptom:
   `Initialize containers` fails with `Exit code 1 ... 'stat', arguments '-c %g /var/run/docker.sock'`.
2. The lease lock files disappear, or the directory is recreated empty. Symptom: the lease step fails
   with `is missing and cannot be created`. The container **cannot** create it: under rootless docker
   its uid is a host subuid that does not own the file, so it opens the lock read-only. The file must
   pre-exist on the host, agent-owned, mode **0666**. A mode of 0644 lets CI read the lock while making every
   host-side acquisition fail with `EACCES`.

The restart script provisions both. A tmpfiles.d rule recreates them at boot.

**Memory**: step exits `137`, or the job receives a shutdown signal during a heavy phase. Confirm the
phase split is intact before changing memory limits; keep baseline coverage CPU-only; isolate
CUDA-gated tests into their own slice; increase swap only after ruling out phase split and fixture
retention.

## Registering a pipeline definition for this repo

Agent-actionable, and **needs no sudo**: registration is an Azure DevOps API operation authenticated by
`AZURE_DEVOPS_EXT_PAT`, not a host-privilege operation.

**The approval gate is attached to the agent POOL, not to a pipeline definition.** A newly registered
pipeline requesting `pool: Default` inherits the gate automatically, so there is no per-pipeline
approval configuration to set and nothing a registration step can silently omit.

```bash
cd ~/repos/distributed-insight/project_admin/shared_admin_scripts/az_pipeline_agent_scripts
./manage-pipelines.sh -o <org> -p <project> -m show -i <template_definition_id>
./manage-pipelines.sh -o <org> -p <project> -m create --dry-run \
  --name "<name>" --yaml "<path/to/pipeline.yml>" --template-id <id> --repo-root <repo>
```

Four things that bite:

- **The YAML must be committed and pushed first.** A definition points at a repo path.
- **Dry-run and diff against a known-good definition.** `connectedServiceId` in particular must be
  copied from the template: it is the GitHub service connection, and a definition without it cannot
  fetch the repo. The general rule, earned from an HTTP 400 `Parameter name: Url`: **copy template
  structures wholesale rather than enumerating the fields you think the API needs.**
- **The payload's `queue.id` is the definition's default queue, not the pool the run uses.** It reads
  as a hosted queue even for a self-hosted pipeline, because the YAML's `pool:` overrides it at queue
  time. Do not read a hosted `queue.id` as evidence the pipeline will be ungated.
- **Definition VARIABLES are not copied and the payload gives no hint.** A new definition starts with
  zero variables while the template's YAML may reference `$(SOME_SECRET)` freely. The reference expands
  to an empty string at run time rather than failing at registration, so it surfaces as a confusing
  auth failure on the first real run. Enumerate the YAML's `$(...)` references and create each:

  ```bash
  az pipelines variable create --pipeline-id <newId> --name <NAME> --value "<val>" --secret true \
    --organization https://dev.azure.com/<org> --project <project>
  ```

There is deliberately **no `delete` mode**, because deleting a definition discards its build history.
Configuring the approval check, creating service connections, agent pool membership and deleting
definitions all stay maintainer-only: they change the trust boundary or destroy history.

## What not to do

- Do not restart the agent to clear a stuck build until the **timeline** has ruled out BOTH a pending
  approval and an unauthorized resource. "Check for a pending approval" is not sufficient and was the
  wording here until 2026-08-30: it routes you to the approvals API, which returns `count: 0` for an
  authorization-blocked build, so the check you were told to run reports clear and you restart a
  healthy agent. A guard that conducts you through a failing check is worse than no guard, because it
  converts an uncertain reader into a confident one.
- Do not assume a build-level `queue.name` of `Azure Pipelines` means the YAML job pool changed.
- Do not force-reset a lease held by CI, or kill the agent to recover one.
- Do not permanently disable GPU coverage to avoid OOMs.
