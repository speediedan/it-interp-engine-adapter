"""Warm the conformance seed component and the delta pair's gated model (run inside the GPU job's container through
ctr.sh).

The conformance seed (interpretune's SEED_REPO at SEED_REVISION) is a PRIVATE Hub component the suite loads cache-only
from HF_HOME; nothing else populates components on this mount, so it is warmed here (small, pinned, idempotent) with a
token scoped to the step that runs this. The seed is fetch-only (nothing hydrates), mirroring interpretune's "Warm Hub
components" step; the gated model below is loaded once on CPU, because that is what caches transformers' probes.
"""

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

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

# The profiling delta pair's circuit-tracer arms load google/gemma-3-1b-pt, which is gated, and the test step carries
# no token. The files alone are not enough: transformers also probes for optional files the repo does not have
# (chat_template.jinja, adapter_config.json), and without a token the gated repo answers those probes 401. Loading
# through transformers here, with the token, caches the "does not exist" answers too, after which the tokenless loads
# succeed.
GATED = "google/gemma-3-1b-pt"
AutoConfig.from_pretrained(GATED)
AutoTokenizer.from_pretrained(GATED)
AutoModelForCausalLM.from_pretrained(GATED, dtype="auto")
GenerationConfig.from_pretrained(GATED)
print(f"warmed {GATED}")
