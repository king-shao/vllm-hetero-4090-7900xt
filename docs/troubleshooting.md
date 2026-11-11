# Troubleshooting

Each entry: symptom → cause → fix. Entries marked **[observed]** came from the
original reproduction; the rest follow from the code paths in this repository.

## `writev ... Bad address` — Gloo staging only {#writev-efault}

**[observed]**

```
gloo::IoException: writev ... Bad address
```

Cause: `GroupCoordinator.send_object` builds its buffer with
`torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)`. `pickle.dumps` returns
immutable `bytes`, and the WSAPoll/writev path in that Gloo build faults on
read-only pages.

Fix: copy into a writable buffer and detach it from the Python object:

```python
object_tensor = torch.frombuffer(bytearray(pickle.dumps(obj)), dtype=torch.uint8).clone()
```

`gloo-staging/sitecustomize.py` already does this. The SHM plane never pickles
payload bytes, so it cannot hit this.

## Hang at the first token, both logs clean

Order of operations, in this order:

1. Are the rings there? `ls /dev/shm | grep ganpp` — inside **each** container.
2. If the file exists on the host but not in a container, that container is not
   in the host IPC namespace → `--ipc=host` missing (`scripts/05-docker-up.sh`).
3. If both see it, check `GAN_SHM_SESSION` — it must be byte-identical on both
   ranks; a mismatch means rank 1 waits on a ring nobody writes.
4. Then check `GAN_SHM_TIMEOUT_MS`; on expiry the shared error word is set and
   both sides raise `EPIPE` with the queue and direction in the message.

## `pp_create ... EEXIST`

A ring with that session name already exists. Stopping containers does not
unlink `/dev/shm`.

```bash
python3 scripts/10-shm.py --session "$GAN_SHM_SESSION" --action unlink
```

## `pp_open ... ENOENT`

The ring was never created, or it was created after the worker started. Create it
before either vLLM process (`scripts/10-shm.py`), and do not create it inside a
container that is not `--ipc=host`.

## Both ranks report ~3.19 GiB of model memory

`VLLM_PP_LAYER_PARTITION` did not reach the child processes; vLLM fell back to
the even split. With `32,4` the split is 1.17 GiB (ROCm, 4 layers) vs 5.21 GiB
(CUDA, 32 layers).

Check: the variable must be set in the `docker exec -e` list *and* exported
before the script sources `scripts/env.sh`. Note vLLM validates that the numbers
sum to `num_hidden_layers` and that their count equals the PP size — a typo gives
a startup error, not a silent even split.

## 4090 utilization sits near 12 % while the 7900 XT is at 100 %

Not a transport problem. With `18,18` the AMD stage is the straggler: its BF16
kernels are slower than the CUDA image's for the same layer count, so the 4090
finishes early and idles. Move layers to the 4090 (`32,4` measured best on
Qwen2.5-3B). Below 4 layers on the AMD side throughput stops improving and then
regresses — `34,2` is slower than `33,3`.

## `AttributeError: module 'vllm.v1.worker.gpu.pp_utils' has no attribute 'PPHandler'`

The V0/V1 model runner is active. This patch hooks the V2 runner's `PPHandler`,
so run with `VLLM_USE_V2_MODEL_RUNNER=1`.

## Recompiling / silence on `pp_dma_open`

`libppdma.so` resolves the vendor runtime at load time: `libcudart.so*` on CUDA,
`libamdhip64.so*` on ROCm (including `/opt/rocm*/lib`), unless
`GAN_SHM_GPU_RUNTIME` points at a specific `libcudart.so`/`libamdhip64.so`. If
`pp_dma_open` returns a non-zero code, the runtime was not found in the process
image — pin it explicitly:

```bash
export GAN_SHM_GPU_RUNTIME=/usr/local/cuda/lib64/libcudart.so.13
```

## `cudaHostRegister failed`

Only payload pages may be registered. Registering the control page breaks the
atomics both processes rely on, and registering a range that crosses the segment
end returns `cudaErrorInvalidValue`. If the failure is on the first slot, the
segment is smaller than `slots * payload_span` — recreate it with the sizes in
`scripts/10-shm.py`.

## AMD device number changes after a reboot

ROCm enumerates `/dev/dri/renderD*` in a different order depending on probe
order. Re-check with `rocm-smi --showproductname` and start the container with
the `--device=/dev/kfd --device=/dev/dri` pair rather than pinning a render node.
The CUDA side is stable (`--gpus device=0`).

## Two processes writing the same Triton / vLLM cache

Both ranks default to the same cache root when they share a mount. Give each its
own:

```bash
VLLM_CACHE_ROOT=/work/runs/$GAN_SHM_SESSION/cache-0   # rank 0
VLLM_CACHE_ROOT=/work/runs/$GAN_SHM_SESSION/cache-1   # rank 1
```

`scripts/21-run-cuda.sh` and `scripts/22-run-rocm.sh` do this already.

## Very slow first request (minutes)

Expected. ROCm autotunes GEMMs and the CUDA side compiles/warms kernels on first
use; `--enforce-eager` removes graph capture but not this. Warm up with one
throwaway request before measuring, and never include the first request in a
benchmark run.

## Benchmark numbers far below the tables

Check, in order: `--max-num-seqs` (the sweep results assume 96), whether
`--enable-prefix-caching` crept back in (it must stay off for a like-for-like
comparison), whether both ranks got the same `VLLM_PP_LAYER_PARTITION`, and
whether the rings were recreated after changing slot sizes (a stale 8 MiB ring
silently caps the payload).

## Still stuck

Capture both logs plus:

```bash
ls -l /dev/shm | grep ganpp
env | grep -E 'GAN_SHM|VLLM_PP|GLOO'
```

The `GAN_SHM_*` block in `env` shows exactly which transport path was active;
`GAN_SHM_PROFILE=1` adds per-slot torch-profiler annotations if the stall is
inside a copy.
