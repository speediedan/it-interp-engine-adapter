"""Warm the conformance seed component (run inside the GPU job's container through ctr.sh).

The conformance seed (interpretune's SEED_REPO at SEED_REVISION) is a PRIVATE Hub component the suite loads cache-only
from HF_HOME; nothing else populates components on this mount, so it is warmed here (small, pinned, idempotent) with a
token scoped to the step that runs this. Fetch-only: nothing hydrates, mirroring interpretune's "Warm Hub components"
step.
"""

from interpretune.hub.components import (
    pull_component_config,
    pull_component_manifest,
    pull_component_payloads,
)
from interpretune.testing.conformance.inputs import SEED_REPO, SEED_REVISION

manifest, commit = pull_component_manifest(SEED_REPO, revision=SEED_REVISION)
pull_component_payloads(SEED_REPO, manifest, commit)
for key in sorted(((manifest.get("module") or {}).get("configs") or {})):
    pull_component_config(SEED_REPO, key, revision=SEED_REVISION)
print(f"warmed {SEED_REPO}@{commit}")
