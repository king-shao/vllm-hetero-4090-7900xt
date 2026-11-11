# vLLM 单机异构流水线并行 —— RTX 4090 + RX 7900 XT

在**同一台机器**上，用一张 NVIDIA RTX 4090 和一张 AMD RX 7900 XT 做流水线并行推理。
调度器、模型代码、KV cache 管理全部使用上游 vLLM `0.27.1` 原样实现，只替换两张卡之间的
**激活值传输与采样反馈**这一段。

English: [README.md](README.md)

```
        rank 0 — RX 7900 XT (ROCm)                      rank 1 — RTX 4090 (CUDA)
        API 服务, PP 第 0..17 层                        headless worker, 第 18..35 层
        ┌───────────────────────┐                       ┌───────────────────────┐
        │  vLLM Scheduler       │      前向激活值        │  vLLM Worker          │
        │  Worker.execute_model │ ═══════════════════▶  │  Worker.execute_model │
        │        │              │   /dev/shm 环形队列    │        │              │
        │        ▼              │   2 × 8 MiB slot      │        ▼              │
        │   PPHandler.send      │ ◀═══════════════════  │   PPHandler.recv      │
        └────────┼──────────────┘      采样结果反馈        └────────┼──────────────┘
                 │              反馈环 2 × 1 MiB                  │
                 └─────────────── Gloo / TCP loopback ───────────┘
                        （仅用于建链与非 PP 集合通信）
```

## 为什么要做这个

NVIDIA NCCL 与 AMD RCCL 是两套独立的通信库，显存模型也不同：两个厂商各自的 PyTorch
构建无法创建同一个 communicator，也读不到对方显存。所以 CUDA + ROCm 混插跑 `PP=2`
时，无论怎么切层，第一个 pipeline send 就会失败。

仓库里保留了两套可用的数据面：

| 数据面 | 机制 | 状态 |
|-|-|-|
| `gloo-staging/`（早期方案） | 所有 process group 强制走 Gloo，每个 PP tensor 走 GPU→CPU→Gloo→CPU→GPU | 正确，实现简单，受拷贝带宽限制 |
| `patch/` + `native/`（本仓库主方案） | POSIX 共享内存 SPSC 环形队列，payload 页注册给 CUDA/HIP，由 DMA 在调用方 stream 上搬运 | 正确，短请求 8 并发下快约 5 % |

SHM 方案**不动** vLLM 的调度和 `Worker.execute_model`，只替换四处：

* `GroupCoordinator.isend_tensor_dict` / `irecv_tensor_dict` —— 激活值
* `PPHandler.receive` / `broadcast` / `get_prev_sampled_outputs` —— 采样反馈
* `GroupCoordinator.__init__` —— 强制 `gloo`，保证建链仍然可用

## 实测结果

Qwen2.5-3B-Instruct，BF16，`PP=2 TP=1`，分层 `18,18`，`max-model-len=4096`，
`--enforce-eager`，输出吞吐（token/s）：

| 输入→输出 | 并发 | Gloo 中转 | **SHM（本仓库）** |
|-|-|-|-|
| 128 → 64 | 1 | 22.39 | **24.18** |
| 128 → 64 | 8 | 155.87 | **163.25** |
| 1024 → 2048 | 1 | 22.34 | **24.44** |
| 1024 → 2048 | 8 | 162.68 | 159.42 |

**分层比数据面重要得多。** 同为早期 Gloo 数据面：

| 分层（7900XT / 4090） | 成功请求 | 输出 tok/s |
|-|-|-|
| 18 / 18 | — | 41.04（4 并发，4090 窗口内平均只有 12.63 % 负载） |
| 34 / 2 | 64/64 | 273.43 |
| 33 / 3 | 224/224 | 277.16 |
| **32 / 4** | **352/352** | **281.31** |

固定 `32,4` 扫并发：281.3 t/s @32 → 356.9 t/s @64 → **607.4 t/s @96**。

## 环境要求

* Linux 宿主机，两张卡都能看到（`nvidia-smi -L`、`rocm-smi --showproductname`）
* Docker + NVIDIA container toolkit + ROCm 设备直通
* 镜像 `vllm/vllm-openai:v0.27.1` 与 `vllm/vllm-openai-rocm:v0.27.1`
* `gcc`、`make`（在任一容器里编译即可）
* 层数能被 PP 整除的模型（Qwen2.5-3B = 36 层）

## 快速开始

```bash
git clone https://github.com/king-shao/vllm-hetero-4090-7900xt
cd vllm-hetero-4090-7900xt

export MODEL_HOST=/path/to/models          # 里面有 Qwen2.5-3B-Instruct/
export GAN_SHM_SESSION=run-$(date +%s)

bash scripts/05-docker-up.sh               # 两个容器，必须 --ipc=host
bash scripts/00-build.sh                   # 编 libppshm.so / libppdma.so
python3 scripts/10-shm.py --session "$GAN_SHM_SESSION"   # 建两条共享内存环

bash scripts/21-run-cuda.sh                # rank 1，headless，先起
sleep 1
bash scripts/22-run-rocm.sh                # rank 0，API 在 :18082

bash scripts/30-bench.sh                   # 四组场景
bash scripts/40-teardown.sh                # 停容器 + 删共享内存
```

正确性验证：

```bash
curl -s localhost:18082/health                       # 200
curl -s localhost:18082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"123+456=?"}],"temperature":0,"max_tokens":16}'
# → 579
```

## 目录结构

```
patch/                    通过 PYTHONPATH 注入的 Python 补丁层
  sitecustomize.py        开关（GAN_SHM_SPSC=1 才加载）
  shm_transport.py        SPSC 数据面，挂 PPHandler + GroupCoordinator
  shm_native.py           libppshm.so 的 ctypes 绑定
  raw_dma.py              libppdma.so 的 ctypes 绑定
  descriptor_checks.py    提交任何 DMA 之前的布局校验
native/
  ppshm.h / ppshm.c       POSIX 共享内存 SPSC 环、futex 唤醒、release/acquire
  gpu_runtime_hook.c      批量 + 融合 D2H/H2D，跑在调用方 stream 上
  Makefile                产出 libppshm.so 与 libppdma.so
gloo-staging/             早期 CPU 中转方案（留作对照与回退）
scripts/                  docker 起容器、编译、建环、启动、压测、清理
docs/                     设计说明、复现步骤、全部压测数据、踩坑
vendor/vllm/              vLLM v0.27.1 源码（已 gitignore，仅用于对照 diff）
```

## 传输开关

默认值即实测最优配置，其余开关作为排障和消融用。

| 变量 | 默认 | 作用 |
|-|-|-|
| `GAN_SHM_SPSC` | `1` | 补丁总开关 |
| `GAN_SHM_SESSION` | — | 环名，两端必须一致 |
| `GAN_SHM_USE_FUTEX` | `1` | 用 futex 唤醒代替自旋/轮询 |
| `GAN_SHM_D2D_STAGE` | `1` | 通过设备内拷贝做中转 |
| `GAN_SHM_RAW_DMA` | `1` | 用 `libppdma.so` 而不是逐 tensor `copy_` |
| `GAN_SHM_FUSED` | `1` | D2H + H2D 合成一次提交 |
| `GAN_SHM_NATIVE_PUBLISHER` | `1` | 由 native 拷贝线程发布 slot 就绪 |
| `GAN_SHM_PACKED_RECV` | `1` | 一步内所有 tensor 一次收完 |
| `GAN_SHM_EARLY_META` | `0` | 先发 metadata 再传 payload |
| `GAN_SHM_PROFILE` | `0` | 每个 slot 打 torch profiler 标注 |
| `GAN_SHM_TIMEOUT_MS` | `60000` | futex 等待上限，超时即失败而不是挂死 |

## 已知限制

* 只支持 `PP=2`、`TP=1`：TP 的 all-gather 没有走 SHM。
* 只支持同机：传输介质是 `/dev/shm`，没有跨网络实现。
* 必须 `--enforce-eager`：中转拷贝路径没有实现 CUDA graph 捕获。
* 当前布局要求 rank 0 在 ROCm 侧（`--nnodes 2` mp executor）。
* 带宽受 PCIe 与较慢一侧的拷贝引擎限制，不是环形队列的限制；8 MiB/slot 是按
  Qwen2.5-3B 的 4096-token 激活值定的。
* 异构 8 并发相比双 4090 同构仍低 42–59 %，瓶颈在 ROCm 侧的 kernel 实现，不在通信。

## 许可

Apache-2.0。`patch/` 与 `native/` 是针对公开 vLLM `0.27.1` ABI 的独立实现；vLLM 本身
是 Apache-2.0，未复制进本仓库。

## 参考

* vLLM `0.27.1` — <https://github.com/vllm-project/vllm/releases/tag/v0.27.1>
* `docs/design.md` —— 环形队列布局、内存序、DMA 路径
* `docs/experiment-log.md` —— 从双 4090 基线到 SHM 的四个阶段（含被放弃的方案）
* `docs/reproduction.md` —— 按顺序的完整命令与判据
* `docs/benchmarks.md` —— 全部实测表格
* `docs/troubleshooting.md` —— `writev EFAULT`、不可变 bytes、残留共享内存
