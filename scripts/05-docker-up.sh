#!/usr/bin/env bash
# Create the two vendor-specific containers.
#
#   MODEL_HOST=/models CUDA_GPU=0 ROCM_GPU=0 scripts/05-docker-up.sh
#
# The critical flag is --ipc=host: the transport exchanges activation slots
# through POSIX shared memory in /dev/shm, so both containers must map the very
# same IPC namespace. With the default per-container /dev/shm (64 MiB tmpfs)
# each process would create a private 8 MiB ring and never see its peer.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"

MODEL_HOST="${MODEL_HOST:-/models}"
CUDA_GPU="${CUDA_GPU:-0}"
ROCM_GPU="${ROCM_GPU:-0}"
CUDA_IMAGE="${CUDA_IMAGE:-vllm/vllm-openai:v0.27.1}"
ROCM_IMAGE="${ROCM_IMAGE:-vllm/vllm-openai-rocm:v0.27.1}"
SHM_SIZE="${SHM_SIZE:-16g}"

for c in "$CUDA_CONTAINER" "$ROCM_CONTAINER"; do
  docker rm -f "$c" >/dev/null 2>&1 || true
done

echo "starting $CUDA_CONTAINER ($CUDA_IMAGE, GPU $CUDA_GPU)"
docker run -d --name "$CUDA_CONTAINER" \
  --gpus "\"device=$CUDA_GPU\"" \
  --ipc=host \
  --shm-size="$SHM_SIZE" \
  -v "$MODEL_HOST":/models \
  -v "$ROOT":"$CONTAINER_ROOT" \
  --entrypoint /bin/bash \
  "$CUDA_IMAGE" -lc 'sleep infinity'

echo "starting $ROCM_CONTAINER ($ROCM_IMAGE, GPU $ROCM_GPU)"
docker run -d --name "$ROCM_CONTAINER" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size="$SHM_SIZE" \
  -v "$MODEL_HOST":/models \
  -v "$ROOT":"$CONTAINER_ROOT" \
  --entrypoint /bin/bash \
  "$ROCM_IMAGE" -lc 'sleep infinity'

echo
echo "containers up. next:"
echo "  docker exec $ROCM_CONTAINER rocm-smi --showproductname   # confirm 7900 XT"
echo "  docker exec $CUDA_CONTAINER nvidia-smi -L                # confirm 4090"
echo "  bash scripts/00-build.sh                                 # build native libs"
echo "  python3 scripts/10-shm.py --session \$GAN_SHM_SESSION    # create rings"
