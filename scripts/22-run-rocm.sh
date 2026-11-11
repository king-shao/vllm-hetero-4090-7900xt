#!/usr/bin/env bash
# rank 0 — RX 7900 XT (ROCm build), API server. Start ~1s after rank 1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/work/runs/$GAN_SHM_SESSION/cache-0}"
mkdir -p "$ROOT/runs/$GAN_SHM_SESSION"

mapfile -t ENVS < <(docker_env_args)
LOG="$ROOT/runs/$GAN_SHM_SESSION/rank0-rocm.log"

docker exec -d "${ENVS[@]}" "$ROCM_CONTAINER" bash -lc "
  mkdir -p /work/runs/$GAN_SHM_SESSION
  exec vllm serve '$MODEL' ${VLLM_COMMON_ARGS[*]} \
    --node-rank 0 --host 127.0.0.1 --port $API_PORT \
    >> /work/runs/$GAN_SHM_SESSION/rank0-rocm.log 2>&1
"

echo "rank0 (ROCm/7900XT) starting; API on 127.0.0.1:$API_PORT"
