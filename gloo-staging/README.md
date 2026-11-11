# Gloo staging — the first working cross-vendor data plane

Kept because it is the fallback when the SHM ring cannot be used (no `/dev/shm`
sharing, no `--ipc=host`, kernel without the futex path), and because it is the
baseline every SHM number in `docs/benchmarks.md` is compared against.

## What it does

Forces every `GroupCoordinator` onto Gloo, then stages each PP tensor through
host memory:

```
GPU(rank A) → CPU contiguous → Gloo send (TCP loopback) → CPU → GPU(rank B)
```

## Use

```bash
# both containers
export PYTHONPATH=/work/spsc/gloo-staging
export VLLM_CROSS_VENDOR_GLOO=1        # without this the module is a no-op
export VLLM_DISABLE_PYNCCL=1
export GLOO_SOCKET_IFNAME=lo
```

Rank layout differs from the SHM plane: in the original Gloo experiment **rank 0
was the 4090** (API server) and **rank 1 the 7900 XT** (headless), while the SHM
scripts in this repo put the API on the ROCm side. Whichever you pick, the
`--node-rank` values and the container you `docker exec` into must match; mixing
them starts a server whose two halves disagree about the topology.

plus the usual `vllm serve` flags:

```
--pipeline-parallel-size 2 --tensor-parallel-size 1
--distributed-executor-backend mp --nnodes 2
--enforce-eager --disable-custom-all-reduce
```

## Patched symbols

| Symbol | Original behaviour | Staged behaviour |
|-|-|-|
| `current_platform.dist_backend` | `nccl` per vendor | `gloo` on both |
| `CudaCommunicator.send` | sends the device tensor directly | GPU→CPU, send on `cpu_group` |
| `CudaCommunicator.recv` | allocates on the local device | receive on CPU, copy to device |
| `GroupCoordinator.send_object` | `frombuffer(pickle.dumps(...))` — immutable storage, `writev` faults | `frombuffer(bytearray(...)).clone()` |
| `GroupCoordinator.isend_tensor_dict` | async device isend | synchronous per-tensor CPU staging |
| `GroupCoordinator.irecv_tensor_dict` | async device irecv | CPU receive, then copy to the descriptor's device |

## Cost

* two extra copies per pipeline step (D2H + H2D) on both ends;
* no overlap between compute and transfer;
* `TP=1` only — the all-gather fast path is dropped.

Measured on Qwen2.5-3B, `18,18`, 8-way: 155.87 tok/s short prompts, 162.68 tok/s
long prompts. See `docs/benchmarks.md` for the side-by-side with the SHM plane.

## Standalone transport test

Before starting vLLM, prove the vendor pair can exchange a BF16 tensor at all:

```bash
# rank 1
docker exec -e GLOO_SOCKET_IFNAME=lo vllm-hetero-cuda \
  python -c "
import torch, torch.distributed as dist
dist.init_process_group('gloo', rank=1, world_size=2, init_method='tcp://127.0.0.1:29610')
t = torch.empty(16, dtype=torch.bfloat16)
dist.recv(t, src=0)
print('rank1 recv', t[:4].tolist())
"
# rank 0
docker exec -e GLOO_SOCKET_IFNAME=lo vllm-hetero-rocm \
  python -c "
import torch, torch.distributed as dist
dist.init_process_group('gloo', rank=0, world_size=2, init_method='tcp://127.0.0.1:29610')
t = torch.arange(16, dtype=torch.bfloat16)
dist.send(t, dst=1)
print('rank0 sent', t[:4].tolist())
"
```

If this hangs, nothing above it will work — check `GLOO_SOCKET_IFNAME`, the port,
and that both containers can reach `127.0.0.1` (they share the host network
namespace only if started with `--network=host`; otherwise use the container IPs
and a bridge network).
