# Reproduction

Ordered command log. Steps 1–3 are host setup, 4–6 start the pair, 7–8 verify,
9 benchmarks, 10 teardown. Every step names its pass/fail evidence.

Assumes the repository is checked out at `/opt/vllm-hetero` on the host and the
model tree is at `/models`.

## 0. Preconditions

```bash
nvidia-smi -L                  # one RTX 4090
rocm-smi --showproductname     # one RX 7900 XT (gfx1100)
docker --version
ls /models/Qwen2.5-3B-Instruct/config.json
```

`config.json` must report `num_hidden_layers: 36`; the partition in
`VLLM_PP_LAYER_PARTITION` has to sum to that number.

## 1. Containers

```bash
cd /opt/vllm-hetero
export MODEL_HOST=/models
export GAN_SHM_SESSION=repro-1008a
bash scripts/05-docker-up.sh
```

Pass: two containers `Up`. **`--ipc=host` is mandatory** — without it each
container gets its own 64 MiB `/dev/shm` and the ring is never shared.

## 2. Native libraries

```bash
bash scripts/00-build.sh
```

Pass: `libppshm.so` and `libppdma.so` exist under `native/`. Host and containers
see the same file because the repo is bind-mounted.

```bash
docker exec vllm-hetero-cuda ldd /work/spsc/native/libppdma.so | grep -E 'dl|pthread'
```

## 3. Shared-memory rings

```bash
python3 scripts/10-shm.py --session "$GAN_SHM_SESSION"
```

Pass: `created forward ...`, `created feedback ...`. On the host:

```bash
ls -l /dev/shm | grep ganpp
# -rw------- 1 user user 16.2M ... /ganpp_repro-1008a_forward
# -rw------- 1 user user  2.2M ... /ganpp_repro-1008a_feedback
```

A leftover segment from an earlier run must be unlinked first, otherwise
`pp_create` fails with `EEXIST`.

## 4. Start rank 1 (CUDA 4090, headless)

```bash
bash scripts/21-run-cuda.sh
```

Watch:

```bash
tail -f runs/$GAN_SHM_SESSION/rank1-cuda.log
```

Pass markers, in order:

```
GAN_SHM_SPSC transport loaded
PP rank 1 ... TP rank 0 ...
Model loading took ... GiB
```

## 5. Start rank 0 (ROCm 7900 XT, API)

```bash
sleep 1
bash scripts/22-run-rocm.sh
tail -f runs/$GAN_SHM_SESSION/rank0-rocm.log
```

Cold start on ROCm can take several minutes (weight load + kernel autotune).

## 6. Health

```bash
curl -s -o /dev/null -w '%{http_code}\n' localhost:18082/health      # 200
curl -s localhost:18082/v1/models | jq -r '.data[].id'              # Qwen2.5-3B-Instruct
```

## 7. Correctness

```bash
curl -s localhost:18082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-3B-Instruct",
       "messages":[{"role":"user","content":"计算 123+456，只输出最终数字。"}],
       "temperature":0,"max_tokens":16}' | jq -r '.choices[0].message.content'
```

Expected: `579`. A wrong-but-fluent answer means the slot hand-off or the
descriptor layout is wrong, not the model.

## 8. Prove the topology

```bash
grep 'PP rank 0' runs/$GAN_SHM_SESSION/rank0-rocm.log | tail -1
grep 'PP rank 1' runs/$GAN_SHM_SESSION/rank1-cuda.log | tail -1
grep 'backend=gloo' runs/$GAN_SHM_SESSION/rank0-rocm.log | tail -1
grep 'quantization=None, quantization_config=None' runs/$GAN_SHM_SESSION/rank0-rocm.log | tail -1
```

Expected last line contains all of: `dtype=torch.bfloat16`,
`pipeline_parallel_size=2`, `tensor_parallel_size=1`, `quantization=None`.

Stage split actually loaded:

```bash
grep 'Model loading took' runs/$GAN_SHM_SESSION/*.log
# PP0 / 7900 XT: 1.17 GiB       (4 layers with 32,4)
# PP1 / 4090:    5.21 GiB       (32 layers with 32,4)
```

Both sides at ~3.19 GiB means `VLLM_PP_LAYER_PARTITION` never reached the child
processes — check `scripts/env.sh` and whether the value is exported before
`docker exec`.

## 9. Benchmark

```bash
SCENARIOS="128:64:1 128:64:8 1024:2048:1 1024:2048:8" bash scripts/30-bench.sh
```

Results land in `runs/$GAN_SHM_SESSION/*.json` next to a `.txt` tail of the
terminal output.

Partition sweep — recreate the rings is **not** needed, restart the servers with
a different `VLLM_PP_LAYER_PARTITION`:

```bash
for p in 18,18 33,3 32,4 34,2; do
  bash scripts/40-teardown.sh
  GAN_SHM_SESSION=part-${p//,/-} bash scripts/05-docker-up.sh
  ...
done
```

## 10. Teardown

```bash
bash scripts/40-teardown.sh
ls /dev/shm | grep ganpp || echo "rings gone"
```

Stopping containers does **not** unlink `/dev/shm`; `40-teardown.sh` does both.

## Troubleshooting quick index

| Symptom | Section |
|-|-|
| `writev ... Bad address` | `docs/troubleshooting.md#writev-efault` |
| `pp_open ... ENOENT` | ring not created, or created in another IPC namespace |
| `pp_create ... EEXIST` | stale ring from a previous session |
| hang at first token | slot timeout; check both ranks' logs for `EPIPE` |
| 4090 idle 87 % of the window | stage imbalance, not transport — see `docs/benchmarks.md#2-layer-partition-sweep` |
