#!/usr/bin/env bash
# Shared configuration for both ranks. Source it, or let the run scripts do it.
#
#   source scripts/env.sh
#
# Every value can be overridden from the outside environment.

# --- identity -------------------------------------------------------------
export GAN_SHM_SESSION="${GAN_SHM_SESSION:-hetero-$(date +%m%d-%H%M%S)}"
export GAN_SHM_SPSC="${GAN_SHM_SPSC:-1}"

# --- where the patch and the native libs live inside the containers -------
# Both containers mount this repository at the same path.
export HOST_ROOT="${HOST_ROOT:-/opt/vllm-hetero}"
export CONTAINER_ROOT="${CONTAINER_ROOT:-/work/spsc}"
export PYTHONPATH="${CONTAINER_ROOT}/patch"
export GAN_SHM_LIBRARY="${CONTAINER_ROOT}/native/libppshm.so"

# --- transport switches (defaults = best measured configuration) ----------
export GAN_SHM_USE_FUTEX=1
export GAN_SHM_D2D_STAGE=1
export GAN_SHM_EARLY_META=0
export GAN_SHM_RAW_DMA=1
export GAN_SHM_PACKED_RECV=1
export GAN_SHM_NATIVE_PUBLISHER=1
export GAN_SHM_FUSED=1
export GAN_SHM_PROFILE="${GAN_SHM_PROFILE:-0}"
export GAN_SHM_TIMEOUT_MS="${GAN_SHM_TIMEOUT_MS:-60000}"

# --- vLLM / runtime -------------------------------------------------------
export VLLM_NO_USAGE_STATS=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_HOST_IP=127.0.0.1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export GLOO_SOCKET_IFNAME=lo
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# --- topology -------------------------------------------------------------
export MODEL="${MODEL:-/models/Qwen2.5-3B-Instruct}"
export SERVED_NAME="${SERVED_NAME:-Qwen2.5-3B-Instruct}"
export PP_SIZE=2
export TP_SIZE=1
export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-18,18}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29745}"
export API_PORT="${API_PORT:-18082}"

# --- containers -----------------------------------------------------------
export CUDA_CONTAINER="${CUDA_CONTAINER:-vllm-hetero-cuda}"
export ROCM_CONTAINER="${ROCM_CONTAINER:-vllm-hetero-rocm}"

# --- serving limits -------------------------------------------------------
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
export GPU_UTIL="${GPU_UTIL:-0.80}"

# Common vLLM flags for both ranks. PP=2 / TP=1 / mp executor, eager mode,
# no custom all-reduce (that path assumes one vendor).
export VLLM_COMMON_ARGS=(
  --served-model-name "$SERVED_NAME"
  --dtype bfloat16
  --pipeline-parallel-size "$PP_SIZE"
  --tensor-parallel-size "$TP_SIZE"
  --distributed-executor-backend mp
  --nnodes 2
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_UTIL"
  --async-scheduling
  --enforce-eager
  --disable-custom-all-reduce
  --generation-config vllm
  --no-enable-prefix-caching
)

# Same variables, rendered as `docker exec -e K=V` arguments.
docker_env_args() {
  local k
  for k in GAN_SHM_SESSION GAN_SHM_SPSC PYTHONPATH GAN_SHM_LIBRARY \
           GAN_SHM_USE_FUTEX GAN_SHM_D2D_STAGE GAN_SHM_EARLY_META \
           GAN_SHM_RAW_DMA GAN_SHM_PACKED_RECV GAN_SHM_NATIVE_PUBLISHER \
           GAN_SHM_FUSED GAN_SHM_PROFILE GAN_SHM_TIMEOUT_MS \
           VLLM_NO_USAGE_STATS VLLM_USE_V2_MODEL_RUNNER \
           VLLM_WORKER_MULTIPROC_METHOD VLLM_HOST_IP VLLM_CACHE_ROOT \
           VLLM_PP_LAYER_PARTITION VLLM_LOGGING_LEVEL \
           GLOO_SOCKET_IFNAME OMP_NUM_THREADS; do
    [[ -n "${!k:-}" ]] && printf -- '-e\n%s=%s\n' "$k" "${!k}"
  done
}
