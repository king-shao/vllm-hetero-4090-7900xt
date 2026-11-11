#!/usr/bin/env bash
# Benchmark matrix used in the paper/README numbers.
#
#   SCENARIOS="128:64:1 128:64:8 1024:2048:1 1024:2048:8" scripts/30-bench.sh
#
# Each scenario is INPUT_LEN:OUTPUT_LEN:CONCURRENCY. Results land in
# runs/<session>/*.json and are also printed to stdout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"

SCENARIOS="${SCENARIOS:-128:64:1 128:64:8 1024:2048:1 1024:2048:8}"
OUT="$ROOT/runs/$GAN_SHM_SESSION"
mkdir -p "$OUT"

# Wait for the server, cold start can take several minutes on ROCm.
echo -n "waiting for http://127.0.0.1:$API_PORT/health "
for i in $(seq 1 180); do
  if [[ "$(curl -fsS -o /dev/null -w '%{http_code}' "http://127.0.0.1:$API_PORT/health" || true)" == "200" ]]; then
    echo "ok"; break
  fi
  echo -n "."; sleep 5
  if [[ $i -eq 180 ]]; then echo " timeout"; exit 1; fi
done

for s in $SCENARIOS; do
  IFS=: read -r in_len out_len conc <<<"$s"
  name="in${in_len}-out${out_len}-c${conc}"
  echo "== $name =="
  docker exec -i "$CUDA_CONTAINER" vllm bench serve \
    --backend openai-chat \
    --base-url "http://127.0.0.1:$API_PORT" \
    --endpoint /v1/chat/completions \
    --model "$SERVED_NAME" \
    --tokenizer "$MODEL" --tokenizer-mode hf \
    --dataset-name random \
    --random-input-len "$in_len" --random-output-len "$out_len" \
    --num-prompts "${NUM_PROMPTS:-100}" \
    --request-rate inf \
    --max-concurrency "$conc" \
    --ignore-eos --seed "${SEED:-0}" \
    --save-result --result-dir "/work/runs/$GAN_SHM_SESSION" \
    --result-filename "$name.json" \
    | tail -20 | tee "$OUT/$name.txt"
done

echo "results: $OUT"
