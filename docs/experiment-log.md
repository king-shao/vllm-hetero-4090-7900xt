# Experiment log — how this repository was reached

Condensed from the original Feishu experiment records (nine documents under
"推理实验": the top-level plan, the Qwen2.5-3B PP writeup, the comparison log,
the result summary table, the 14B test, the technical article, and three
one-page records). Kept in the repository because the negative results are what
justify the final shape of the patch.

## Stage 0 — native PP on a homogeneous pair (reference)

Two RTX 4090s, two containers on a `vllm-pp-net` bridge, Ray head + worker, one
GPU each. Model Qwen3.8-27B-FP8, `--pipeline-parallel-size 2`, Ray backend,
`max-model-len=1024`, `--enforce-eager`, 64-way concurrency.

Outcome: it runs, and it exposes the problem the rest of the work is about —
**memory and compute load are visibly skewed between the two ranks** even with an
even partition, because the first stage holds the embedding plus the first block
of KV and the second stage owns the sampling path. This is the reference run that
made "partition and balance" the main axis, not "make it compile".

## Stage 1 — the naive cross-vendor attempt fails

Same topology, one 4090 + one 7900 XT. `PP=2` dies before the first token:

* `current_platform.dist_backend` is `nccl` on both sides, but one is NCCL and the
  other RCCL;
* `CudaCommunicator.send` hands a *device* tensor to the process group, and
  neither vendor can address the peer's device memory;
* `isend_tensor_dict` / `irecv_tensor_dict` do the same thing asynchronously.

Conclusion recorded at the time: this cannot be fixed by making NCCL and RCCL
compatible. The activation path has to leave the vendor collective stack.

## Stage 2 — a scheduler of our own (abandoned)

An early attempt replaced vLLM's scheduling so the two halves could be driven
directly. It shipped, and it left a structural defect: the first half of the run
was serial and only the second half overlapped, because without vLLM's scheduler
there is no batching window to hide the transfer behind. Abandoned.

The lesson that survived: **keep vLLM's scheduler and model code; replace only
the data plane.** Every later iteration follows that rule.

## Stage 3 — reuse vLLM's scheduler, stage through CPU (Gloo)

Force all process groups to Gloo, keep the scheduler, stage each PP tensor
through host memory. Two implementation bugs had to be fixed first (both
documented in `docs/troubleshooting.md`):

1. `send_object` publishes an *immutable* `pickle.dumps` buffer → Gloo `writev`
   fails with `Bad address`;
2. the V2 tensor-dict path still issues async device isend/irecv → CPU staging
   has to be synchronous.

Result on Qwen2.5-3B-Instruct, `18,18`, 8-way: 155.87 tok/s (short), 162.68
tok/s (long). Working, but the 4090 sat at ~12.6 % average utilization during
the 4-way run at 41.04 tok/s — the AMD stage was the straggler.

### Stage 3a — partition sweep

`VLLM_PP_LAYER_PARTITION`: 18/18 → 34/2 → 33/3 → **32/4**. Throughput
41.04 → 273.43 → 277.16 → **281.31** tok/s. The curve turns between 33/3 and
34/2, so 32/4 is a knee, and the AMD side keeps 4 real layers rather than a
token single layer.

### Stage 3b — concurrency sweep at 32/4

281.31 @32 → 356.93 @64 → **607.38 @96** tok/s; 96 is bounded by ~102.35
sequences of KV capacity in the log, not by an arbitrary constant.

### Stage 3c — homogeneous control

Same scheduler, same `18,18`, two 4090s over Gloo: 20.01 / 248.27 / 20.29 /
231.07 tok/s. Single stream is *slower* than the heterogeneous pair (the
heterogeneous pair wins by ~10 % on latency-bound loads), while 8-way steady
state is 42–59 % faster. That split is the whole argument that stage imbalance
plus ROCm kernel parity — not bandwidth — is the limit.

## Stage 4 — shared memory instead of TCP (this repository)

Replace the staging path with a POSIX shared-memory SPSC ring, register the
payload pages with CUDA/HIP, submit D2H+H2D as one fused job on the caller's
streams, and publish slot readiness from the native copy thread.

Measured against Stage 3 on identical settings:

| Load | Gloo staging | SHM | Δ |
|-|-|-|-|
| 128→64 c1 | 22.39 | 24.18 | +8.0 % |
| 128→64 c8 | 155.87 | 163.25 | +4.7 % |
| 1024→2048 c1 | 22.34 | 24.44 | +9.4 % |
| 1024→2048 c8 | 162.68 | 159.42 | −2.0 % |

The gain is on latency-bound loads; at 8-way long output the extra ring
bookkeeping costs ~2 %. Published as-is rather than tuned away: the transport is
not the bottleneck at that operating point.

## What the log says to do next

1. **ROCm kernel parity.** The 42–59 % steady-state gap against a 4090 pair is
   compute, not transport: equal layer counts are not equal work.
2. **TP=1 is a ceiling.** The all-gather path is still vendor-only, so the patch
   cannot be combined with tensor parallelism yet.
3. **Graph capture.** `--enforce-eager` costs real throughput; capturing the
   staged copy path is the obvious next win.
4. **Slot capacity is static.** 8 MiB covers Qwen2.5-3B at 4096 tokens; a larger
   model means recreating the segment.

## Data provenance

| Claim | Source |
|-|-|
| Stage 0 topology and skew | 推理实验 (top-level plan), 64-way run, profile screenshots |
| NCCL/RCCL failure and the three broken call sites | Qwen2.5-3B模型PP异构并行 §7 |
| Gloo staging numbers, partition and concurrency sweeps | 对比实验 (comparison log), 对比实验-结果汇总表格 |
| SHM numbers and design | 异构环境并发优化实验--SHM section of the comparison log |
| Homogeneous 4090 control | 对比实验 §autoDL 同构环境 |
| 14B run | Qwen 2.5-14B模型测试 |
