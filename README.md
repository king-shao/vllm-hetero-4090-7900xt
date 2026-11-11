# vLLM Heterogeneous PP — RTX 4090 + RX 7900 XT

Pipeline-parallel inference across **one NVIDIA RTX 4090 and one AMD RX 7900 XT**
in a single machine, using upstream vLLM `0.27.1` scheduling and model code, with
only the inter-stage data plane replaced.

`中文说明见 [README.zh-CN.md](README.zh-CN.md)`

```
        rank 0 — RX 7900 XT (ROCm)                      rank 1 — RTX 4090 (CUDA)
        API server, PP stages 0..17                     headless worker, stages 18..35
        ┌───────────────────────┐                       ┌───────────────────────┐
        │  vLLM Scheduler       │                       │  vLLM Worker          │
        │  Worker.execute_model │  forward activations  │  Worker.execute_model │
        │        │              │ ═══════════════════▶  │        │              │
        │        ▼              │   /dev/shm ring       │        ▼              │
        │   PPHandler.send      │   2 × 8 MiB slots     │   PPHandler.recv      │
        │        ▲              │ ◀═══════════════════  │        │              │
        └────────┼──────────────┘  sampled tokens       └────────┼──────────────┘
                 │               feedback ring 2 × 1 MiB         │
                 └─────────────── Gloo / TCP loopback ───────────┘
                       (rendezvous + control plane only)
```

## Why this exists

NVIDIA NCCL and AMD RCCL are different libraries with different device memory
models. Two vendor-specific PyTorch builds cannot create a shared communicator,
and neither can read the peer's GPU storage — so `PP=2` across CUDA + ROCm fails
before the first token is generated, no matter how the layers are partitioned.

Two working data planes are included:

| Data plane | Mechanism | Status |
|-|-|-|
| `gloo-staging/` (legacy) | Force every process group to Gloo; every PP tensor does GPU→CPU→Gloo→CPU→GPU | Correct, simple, copy-bound |
| `patch/` + `native/` (this repo) | SPSC ring in POSIX shared memory; payload pages registered with CUDA/HIP, transferred by DMA on the caller's stream | Correct, ~5 % faster on short prompts at 8-way concurrency |

The SHM plane keeps vLLM's scheduler and `Worker.execute_model` untouched. It
replaces only:

* `GroupCoordinator.isend_tensor_dict` / `irecv_tensor_dict` — activations
* `PPHandler.receive` / `broadcast` / `get_prev_sampled_outputs` — sampling feedback
* `GroupCoordinator.__init__` — force `gloo` so rendezvous still works

## Measured results

Qwen2.5-3B-Instruct, BF16, `PP=2 TP=1`, partition `18,18`, `max-model-len=4096`,
`--enforce-eager`, output throughput in token/s:

| Prompt → output | Concurrency | Gloo staging | **SHM (this repo)** |
|-|-|-|-|
| 128 → 64 | 1 | 22.39 | **24.18** |
| 128 → 64 | 8 | 155.87 | **163.25** |
| 1024 → 2048 | 1 | 22.34 | **24.44** |
| 1024 → 2048 | 8 | 162.68 | 159.42 |

Uneven layer partition matters far more than the transport. With the legacy Gloo
plane on the same hardware:

| Partition (7900XT / 4090) | Requests ok | Output t/s |
|-|-|-|
| 18 / 18 | — | 41.04 (4-way, 4090 idle 87 % of the window) |
| 34 / 2 | 64/64 | 273.43 |
| 33 / 3 | 224/224 | 277.16 |
| **32 / 4** | **352/352** | **281.31** |

Concurrency sweep at `32,4`: 281.3 t/s @32 → 356.9 t/s @64 → **607.4 t/s @96**.

## Requirements

* Linux host with both GPUs visible (`nvidia-smi -L`, `rocm-smi --showproductname`)
* Docker with the NVIDIA container toolkit and ROCm device passthrough
* Images `vllm/vllm-openai:v0.27.1` and `vllm/vllm-openai-rocm:v0.27.1`
* `gcc`, `make` (inside either container is fine)
* A model with an even PP-friendly layer count (Qwen2.5-3B = 36 layers)

## Quick start

```bash
git clone https://github.com/king-shao/vllm-hetero-4090-7900xt
cd vllm-hetero-4090-7900xt

export MODEL_HOST=/path/to/models          # contains Qwen2.5-3B-Instruct/
export GAN_SHM_SESSION=run-$(date +%s)

bash scripts/05-docker-up.sh               # 2 containers, --ipc=host is required
bash scripts/00-build.sh                   # libppshm.so + libppdma.so
python3 scripts/10-shm.py --session "$GAN_SHM_SESSION"   # create both rings

bash scripts/21-run-cuda.sh                # rank 1, headless — start first
sleep 1
bash scripts/22-run-rocm.sh                # rank 0, API on :18082

bash scripts/30-bench.sh                   # 4-scenario matrix
bash scripts/40-teardown.sh                # stop + unlink rings
```

Health and correctness:

```bash
curl -s localhost:18082/health                       # 200
curl -s localhost:18082/v1/models | jq -r '.data[].id'
curl -s localhost:18082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"123+456=?"}],"temperature":0,"max_tokens":16}'
# → 579
```

## Repository layout

```
patch/                    Python shim, injected via PYTHONPATH
  sitecustomize.py        opt-in loader (GAN_SHM_SPSC=1)
  shm_transport.py        SPSC data plane, PPHandler + GroupCoordinator hooks
  shm_native.py           ctypes binding for libppshm.so
  raw_dma.py              ctypes binding for libppdma.so
  descriptor_checks.py    layout validation before any copy is submitted
native/
  ppshm.h / ppshm.c       POSIX shm SPSC ring, futex wake, release/acquire
  gpu_runtime_hook.c      batched + fused D2H/H2D on the caller's stream
  Makefile                builds libppshm.so and libppdma.so
gloo-staging/             the earlier CPU-staging transport (reference)
scripts/                  docker up, build, shm create, run, bench, teardown
docs/                     design, experiment log, reproduction, benchmarks
vendor/vllm/              vLLM v0.27.1 checkout (git-ignored, for diffing)
```

## Transport switches

All defaults are the measured-best configuration; every switch is an escape hatch.

| Variable | Default | Effect |
|-|-|-|
| `GAN_SHM_SPSC` | `1` | master switch for the shim |
| `GAN_SHM_SESSION` | — | ring name, identical on both ranks |
| `GAN_SHM_USE_FUTEX` | `1` | futex wake instead of spin/poll |
| `GAN_SHM_D2D_STAGE` | `1` | stage through device-to-device copies |
| `GAN_SHM_RAW_DMA` | `1` | use `libppdma.so` instead of per-tensor `copy_` |
| `GAN_SHM_FUSED` | `1` | submit D2H+H2D as one fused job |
| `GAN_SHM_NATIVE_PUBLISHER` | `1` | publish slot readiness from the native thread |
| `GAN_SHM_PACKED_RECV` | `1` | receive all tensors of a step in one pass |
| `GAN_SHM_EARLY_META` | `0` | publish metadata before payload DMA |
| `GAN_SHM_PROFILE` | `0` | torch profiler annotations per slot |
| `GAN_SHM_TIMEOUT_MS` | `60000` | futex wait ceiling before failing a step |

## Limitations

* `PP=2`, `TP=1` only — the TP all-gather path is not routed through SHM.
* Same host only. The ring lives in `/dev/shm`; there is no network transport.
* `--enforce-eager` is required; CUDA graph capture of the staged copy path is
  not implemented.
* Rank 0 must be the ROCm side in the current layout (`--nnodes 2` mp executor).
* Bandwidth is bounded by PCIe plus the slower vendor's copy engines, not by the
  ring: 8 MiB per slot is sized for the 4096-token activation of Qwen2.5-3B.

## License

Apache-2.0. `patch/` and `native/` are original work targeting the public vLLM
`0.27.1` ABI; vLLM itself is Apache-2.0 and is not vendored into this repository.

## References

* vLLM `0.27.1` — <https://github.com/vllm-project/vllm/releases/tag/v0.27.1>
* `docs/design.md` — ring layout, memory ordering, DMA path
* `docs/experiment-log.md` — the four stages that led here, including the abandoned one
* `docs/reproduction.md` — full command log, in order
* `docs/benchmarks.md` — every measured table
* `docs/troubleshooting.md` — `writev EFAULT`, immutable-bytes pickle, stale rings
