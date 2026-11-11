# Design — cross-vendor pipeline parallel data plane

Target: vLLM `0.27.1`, `PP=2`, `TP=1`, two containers on one host, rank 0 = ROCm
(RX 7900 XT), rank 1 = CUDA (RTX 4090).

## 1. Why the stock path cannot work

vLLM picks its process-group backend from `current_platform`. On CUDA that is
NCCL, on ROCm it is RCCL, and both report the string `"nccl"` through
`torch.distributed.get_backend()`. That name collision is cosmetic:

* NCCL can only allocate and operate NVIDIA communicators and CUDA device memory.
* RCCL can only allocate and operate AMD communicators and HIP device memory.
* `torch.distributed` in two different PyTorch builds cannot share one
  `ncclComm_t`.
* Neither backend can `cudaMemcpy` from, or RDMA into, the peer's device memory.

So `PP=2` dies at the first pipeline send regardless of layer partitioning. The
fix has to move the activation path off the vendor collective stack entirely.

## 2. Two working data planes

### 2.1 Gloo staging (`gloo-staging/`, the earlier experiment)

Force every `GroupCoordinator` to Gloo, then stage every PP tensor through CPU:

```
GPU(rank1) → CPU contiguous → Gloo send (TCP loopback) → CPU → GPU(rank0)
```

Two bugs had to be fixed on the way:

1. `send_object` built its buffer with `torch.frombuffer(pickle.dumps(obj), ...)`.
   `pickle.dumps` returns immutable `bytes`; Gloo's `writev` on that storage
   fails with `gloo::IoException: writev ... Bad address`. Fix: copy into a
   `bytearray(...)` and `.clone()`.
2. The V2 `isend_tensor_dict` / `irecv_tensor_dict` pair issues *async* GPU
   isend/irecv, which still crosses vendors. Fix: synchronous per-tensor CPU
   staging.

Cost: two extra copies per pipeline step, no compute/communication overlap, and
Gloo's TCP framing on a loopback interface that both ranks already share.

### 2.2 Shared-memory SPSC ring (`patch/` + `native/`, this repo)

Both processes map the *same* POSIX shared-memory segment. There is no socket,
no serialization of the payload, and no third-party collective in the data path.
Gloo is still used for rendezvous and for non-PP collectives, because two vendor
builds must agree on *some* backend to start.

```
rank1 (CUDA)                                          rank0 (ROCm)
  activation tensor (GPU)
        │ pp_dma_fused(): D2H on copy stream
        ▼
  /dev/shm slot payload ──────── same pages ────────▶ mapped + cudaHostRegister'd
                                                       │ pp_dma_fused(): H2D on model stream
                                                       ▼
                                                   activation tensor (GPU)
```

## 3. Segment layout

`pp_create(name, slots, metadata_bytes, payload_bytes)` allocates:

```
+-------------------------- 0
| pp_shared (control page)  |  magic, version, slots, capacities, spans,
|                           |  atomic error, 3 x pp_queue (free/metadata/ready)
+-------------------------- control_span = align_page(sizeof(pp_shared))
| slot 0: pp_slot_header    |  sequence, metadata_bytes, payload_bytes
|         metadata bytes    |  pickle of the tensor dict descriptors
|         padding to page   |
|         payload pages     |  activation bytes, page aligned
+-------------------------- stride = meta_span + payload_span
| slot 1 ...                |
+--------------------------
```

Measured configuration:

| Ring | Slots | Metadata / slot | Payload / slot | Total |
|-|-|-|-|-|
| `forward` | 2 | 64 KiB | 8 MiB | ~16.2 MiB |
| `feedback` | 2 | 64 KiB | 1 MiB | ~2.2 MiB |

Two slots are enough for a strict ping-pong PP schedule; the ring supports up to
8 (`PP_MAX_SLOTS`) without changing any caller.

## 4. Slot state machine

Three intrusive queues index the same slot array:

| Queue | Kind | Meaning |
|-|-|-|-|
| `free` | 0 | slot available to the sender |
| `ready` | 1 | payload DMA finished, bytes are visible |
| `metadata` | 2 | descriptors published, receiver may parse |

Sender:

```
pop(free) → write header + metadata → push(metadata)
          → submit DMA into the payload → on completion push(ready)
```

Receiver:

```
pop(metadata) → parse descriptors → validate layout
              → pop(ready) → DMA payload into device tensors → push(free)
```

`pp_queue_pop` / `pp_queue_push` take `timeout_ms`; a timeout marks the shared
`error` word with `EPIPE` and wakes the peer, so one stalled rank never leaves
the other blocked forever. Every later operation checks that error word first.

## 5. Memory ordering

Publication is release/acquire on the queue head/tail atomics plus a futex wake:

* producer writes payload bytes, then `atomic_store_explicit(head, ...,
  memory_order_release)`, then `futex_wake` if the consumer is parked;
* consumer `atomic_load_explicit(tail, ..., memory_order_acquire)`, checks the
  slot, `futex_wait` on the queue address when empty.

The payload pages are written by DMA, not by the CPU that publishes the slot, so
the ready-queue push happens from the copy-completion path
(`GAN_SHM_NATIVE_PUBLISHER=1`), after the stream event is recorded. That is why
the native publisher exists: publishing from Python after an `event.synchronize()`
would serialize the copy again.

## 6. Host registration and DMA

Each channel registers every payload slot once, at startup:

```python
self.runtime = torch.cuda.cudart()
for address in payload_addresses:
    result = self.runtime.cudaHostRegister(address, payload_span, 0)
```

Notes:

* Only payload pages are registered — never the control page, which must stay
  coherent for atomic access from both processes.
* The same call is valid on ROCm: `torch.cuda.cudart()` returns the HIP shim
  object there, and `hipHostRegister` has the same signature/semantics.
* Registered pages are pinned, so the kernel can DMA directly to/from them.
  Registration is one-shot; the segment never moves.

`libppdma.so` (`native/gpu_runtime_hook.c`) resolves the vendor runtime at
`dlopen` time — `libcudart.so*` on CUDA, `libamdhip64.so*` on ROCm, or an explicit
`GAN_SHM_GPU_RUNTIME` path. It exposes:

| Symbol | Purpose |
|-|-|
| `pp_dma_open(path, hip, control_path, device)` | bind to the runtime + the control ring |
| `pp_dma_batch(handle, copies[], count, stream)` | submit N contiguous copies in one call |
| `pp_dma_fused(handle, region, slot, generation, model_stream, copy_stream)` | D2H then H2D for one slot as a single job |
| `pp_dma_source_wait` / `pp_dma_source_query` | wait on / poll the source-side event |
| `pp_dma_publish(handle, region, slot, ready, stream)` | push the slot onto a queue from the copy thread |
| `pp_dma_release_cpu(handle, region, slot)` | return a slot without a device copy |
| `pp_dma_drain(handle, timeout_ms)` | quiesce before shutdown |

A dedicated copy thread per rank drains the job ring, so the Python thread only
enqueues. `descriptor_checks.py` rejects overlapping, misaligned or size-mismatched
descriptors *before* any target allocation or DMA submission — a bad layout is a
`ValueError`, not a corrupted activation.

## 7. Hook points in vLLM 0.27.1

`patch/shm_transport.py` monkey-patches four things and nothing else:

| Symbol | File:line in 0.27.1 | Replacement |
|-|-|-|
| `GroupCoordinator.__init__` | `distributed/parallel_state.py:380` | force backend `"gloo"`, `use_device_communicator=False` |
| `GroupCoordinator.isend_tensor_dict` | `distributed/parallel_state.py:1019` | metadata via `send_object`, payload into the forward ring |
| `GroupCoordinator.irecv_tensor_dict` | `distributed/parallel_state.py:1114` | pop metadata, validate, DMA payload into the target tensors |
| `PPHandler.get_prev_sampled_outputs` | `v1/worker/gpu/pp_utils.py:93` | read the feedback ring instead of the Go go-RPC |
| `PPHandler.receive` | `v1/worker/gpu/pp_utils.py:126` | fill a `PendingRecv` from the feedback ring |
| `PPHandler.broadcast` | `v1/worker/gpu/pp_utils.py:170` | write sampled ids into the feedback ring |

The scheduler, the model runner, `Worker.execute_model`, KV cache management and
sampling itself are untouched. That is the property that makes the patch
portable: it does not depend on vLLM's internal model execution API, only on the
PP boundary.

`PPHandler` is only reachable through `vllm.v1.worker.gpu.pp_utils`, so
`VLLM_USE_V2_MODEL_RUNNER=1` is required; the V0/V1 runner has no equivalent
object.

## 8. Why Gloo is still configured

Even with the payload path fully on SHM, vLLM initializes process groups for
non-PP work (statistics, DP coordination, weight utils). Two vendor builds must
agree on one backend, and Gloo is the only one both provide without device
memory. `GLOO_SOCKET_IFNAME=lo` keeps that traffic on loopback and avoids
picking up a slow interface.

## 9. Failure handling

* Ring timeout → `pp_set_error(EPIPE)` + wake peer → both sides raise with the
  slot/direction in the message, instead of hanging until the benchmark times out.
* Descriptor mismatch → `ValueError` before any copy is submitted.
* Slot count / capacity mismatch between create and open → header check fails at
  `pp_open`.
* On teardown, `atexit` drains the copy thread and only then unmaps. A hard kill
  leaves the segment mapped until the process dies, which is safe: the ring is a
  file in `/dev/shm` and is re-created by `scripts/10-shm.py`.

## 10. Known limits

* `TP=1` only: the TP all-gather is not routed through SHM, and
  `--disable-custom-all-reduce` is required for the same reason.
* Eager mode only: CUDA graph capture of the staged copy path is not implemented.
* Same host only: `/dev/shm` is the transport.
* Payload capacity is fixed at creation. 8 MiB covers a 4096-token
  Qwen2.5-3B activation; larger models/lengths need a larger slot, and the
  segment must then be recreated.
* The 4090 side still holds 32 of 36 layers in the best configuration — the
  7900 XT contributes compute, not capacity, because ROCm lacks some of the BF16
  kernel paths the CUDA image uses.
