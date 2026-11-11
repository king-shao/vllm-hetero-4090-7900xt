# Benchmarks

All numbers are `vllm bench serve`, `--dataset-name random`, `--seed 0`,
`--ignore-eos`, 100 prompts. Throughput is **output** token/s unless marked
otherwise.

## Environment

| Item | Value |
|-|-|
| Host | single machine, one RTX 4090 + one RX 7900 XT |
| CUDA container | `vllm/vllm-openai:v0.27.1` (Python 3.12, PyTorch 2.13.0+cu130) |
| ROCm container | `vllm/vllm-openai-rocm:v0.27.1` |
| Model | Qwen2.5-3B-Instruct, BF16, 36 layers |
| Parallelism | `PP=2 TP=1`, `--enforce-eager`, `--disable-custom-all-reduce` |
| Limits | `max-model-len=4096`, `max-num-batched-tokens=1024`, `gpu-mem-util=0.80` |
| Transport shared flags | `GLOO_SOCKET_IFNAME=lo`, `VLLM_USE_V2_MODEL_RUNNER=1` |

## 1. Data plane: Gloo staging vs SHM

Partition `18,18` (uncorrected), `max-num-seqs=8`.

| Prompt → output | Conc. | Gloo staging | SHM | Δ |
|-|-|-|-|-|
| 128 → 64 | 1 | 22.39 | **24.18** | +8.0 % |
| 128 → 64 | 8 | 155.87 | **163.25** | +4.7 % |
| 1024 → 2048 | 1 | 22.34 | **24.44** | +9.4 % |
| 1024 → 2048 | 8 | 162.68 | 159.42 | −2.0 % |

Reading: the SHM plane wins where per-step latency dominates (single stream,
short prompts). At 8-way long-output concurrency the pipeline is compute-bound on
the 4090 and the extra ring bookkeeping costs ~2 %. Neither plane fixes stage
imbalance; that is section 2.

## 2. Layer partition sweep (Gloo staging, `max-num-seqs=96` unless noted)

Ranks are (ROCm 7900 XT / CUDA 4090). `VLLM_PP_LAYER_PARTITION` must sum to 36.

| Partition | Requests | Output tok/s | Note |
|-|-|-|-|
| 18 / 18 | 4-way run | 41.04 | 4090 busy only 12.63 % of the window — 7900 XT is the straggler |
| 34 / 2 | 64 ok / 0 fail | 273.43 | 4090 now the bottleneck |
| 33 / 3 | 224 ok / 0 fail | 277.16 | −1.48 % vs 32/4 |
| **32 / 4** | **352 ok / 0 fail** | **281.31** | best |

Why not push AMD down to 1–2 layers: below 4 the 4090 stage saturates and the
throughput curve flattens *and* turns (34/2 slower than 33/3), so 32/4 is the
knee, not an arbitrary pick.

## 3. Concurrency sweep at `32,4`

| Concurrency | Output tok/s | Total tok/s | Mean latency | 4090 util mean / peak |
|-|-|-|-|-|
| 32 | 281.31 | 2844.70 | 29.10 s | 25.13 % / 100 % |
| 64 | 356.93 | 3609.44 | 33.73 s | 29.82 % / 100 % |
| **96** | **607.38** | **6142.33** | 39.77 s | 34.24 % / 100 % |

96 is the ceiling for this configuration: the log reports ~102.35 concurrent
4096-token KV sequences of capacity, so 96 fits with a margin.

## 4. Cross-vendor vs homogeneous (same scheduler, same partition)

Both rows reuse vLLM's own scheduler; only the data plane and the peer GPU differ.
Homogeneous baseline is 4090 + 4090 on AutoDL over Gloo.

| Load | 4090 + 4090 | 4090 + 7900 XT | Delta |
|-|-|-|-|
| 128 → 64, c1 | 20.01 | 22.39 | −10.6 % (hetero faster) |
| 128 → 64, c8 | 248.27 | 155.87 | 59.3 % lower |
| 1024 → 2048, c1 | 20.29 | 22.34 | −9.2 % (hetero faster) |
| 1024 → 2048, c8 | 231.07 | 162.68 | 42 % lower |

Interpretation:

* Single stream: the heterogeneous pair is *ahead*. With one request in flight
  the pipeline is latency-bound and the 7900 XT stage is not the critical path;
  the measured difference comes from the two containers' copy paths, not compute.
* 8-way steady state: the gap is real and comes from vendor compute asymmetry plus
  stage imbalance — the AMD stage's BF16 kernels are not the same code path as
  the CUDA image's, so equal layer counts are not equal work.

## 5. Native PP baseline (both GPUs CUDA, Ray, 27B FP8)

Reference run of the *unmodified* path on two 4090s, to show what "correct but
unbalanced" looks like before any transport work:

* Model Qwen3.8-27B-FP8, `--pipeline-parallel-size 2`, Ray backend, two
  containers joined by a `vllm-pp-net` bridge, `max-model-len=1024`,
  `--enforce-eager`, 64-way concurrency.
* Result: 64 concurrent requests complete, but both memory and compute load are
  visibly skewed between ranks (see the profile screenshots in the original
  experiment log). This is the baseline the partition sweep in section 2 exists
  to correct.

## 6. Summary

1. Cross-vendor `PP=2` is achievable with upstream vLLM 0.27.1 plus a transport
   patch: no scheduler fork, no model-code change.
2. Transport choice moves throughput by single-digit percent; **partition choice
   moves it by 6.8×** (41.04 → 281.31 tok/s).
3. SHM beats CPU-staged Gloo on latency-sensitive loads; the gap closes and
   slightly inverts on long-output 8-way concurrency.
4. Against a homogeneous 4090 pair at 8-way concurrency the heterogeneous setup
   is 42–59 % slower — the remaining work is kernel parity on the ROCm side, not
   the ring.
