#!/usr/bin/env bash
# Stop both containers and drop the shared-memory segments.
# Stopping a container does NOT unlink /dev/shm; the segment outlives it and a
# stale ring makes the next boot fail with a version/magic mismatch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"

docker stop "$CUDA_CONTAINER" "$ROCM_CONTAINER" 2>/dev/null || true
python3 "$ROOT/scripts/10-shm.py" --session "$GAN_SHM_SESSION" --action unlink
echo "torn down (session=$GAN_SHM_SESSION)"
