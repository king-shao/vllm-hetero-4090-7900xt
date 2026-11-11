#!/usr/bin/env bash
# rank 1 — RTX 4090 (CUDA build), headless PP worker. Start this one FIRST.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/work/runs/$GAN_SHM_SESSION/cache-1}"
mkdir -p "$ROOT/runs/$GAN_SHM_SESSION"

mapfile -t ENVS < <(docker_env_args)
LOG="$ROOT/runs/$GAN_SHM_SESSION/rank1-cuda.log"

docker exec -d "${ENVS[@]}" "$CUDA_CONTAINER" bash -lc "
  mkdir -p /work/runs/$GAN_SHM_SESSION
  exec vllm serve '$MODEL' ${VLLM_COMMON_ARGS[*]} \
    --node-rank 1 --headless >> /work/runs/$GAN_SHM_SESSION/rank1-cuda.log 2>&1
"

echo "rank1 (CUDA/4090) starting; log: $LOG"
