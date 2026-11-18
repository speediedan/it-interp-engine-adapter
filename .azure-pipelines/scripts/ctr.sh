#!/usr/bin/env bash
# Run one command inside the any-vendor GPU job's container (gpu-epsilon's templates/gpu-job.yml), from a host step,
# with the variables the in-container commands read forwarded by name. The same shape as interpretune's
# .azure-pipelines/scripts/ctr.sh, cut to what this pipeline uses.
#
#   ctr.sh <command> [args...]
#
# Forwarded when set on the host step: every IT_* variable the template sets (IT_GPU_VENDOR, IT_IMAGE, ...), the build
# id the lease label names, and HF_TOKEN, which only the seed-warming step maps. Each command activates the image's
# venv itself: `docker exec` starts no login shell, so nothing else would put /tmp/venvs/it_dev first on PATH.
set -euo pipefail

: "${IT_CTR:?IT_CTR is set by the template}"
names=(BUILD_BUILDID BUILD_REASON BUILD_SOURCEBRANCH HF_TOKEN IT_RUN_CUDA_TESTS IT_TRUST_REMOTE_CODE)
while IFS= read -r name; do names+=("$name"); done < <(compgen -e | grep '^IT_' | grep -v '^IT_CTR$' || true)
args=()
for name in "${names[@]}"; do
    [[ -n "${!name+x}" ]] && args+=(-e "$name")
done
exec docker exec "${args[@]}" "$IT_CTR" bash -c "source /tmp/venvs/it_dev/bin/activate && $*"
