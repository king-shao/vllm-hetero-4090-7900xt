#!/usr/bin/env bash
# Fetch the vLLM source tree this patch targets, for diffing and for building a
# custom image. The checkout is git-ignored on purpose: the runtime is the
# official image, and vendoring 110 MiB of source into this repository would not
# make the patch any more correct.
#
#   bash scripts/50-fetch-vllm.sh            # -> vendor/vllm @ v0.27.1
#   VLLM_REF=v0.28.0 bash scripts/50-fetch-vllm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="${VLLM_REF:-v0.27.1}"
DEST="$ROOT/vendor/vllm"

if [[ -d "$DEST/.git" ]]; then
  echo "existing checkout: $(git -C "$DEST" describe --tags --always 2>/dev/null)"
  git -C "$DEST" fetch --depth 1 origin "refs/tags/$REF:refs/tags/$REF" 2>/dev/null || true
  git -C "$DEST" checkout -q "$REF"
else
  mkdir -p "$ROOT/vendor"
  git clone --depth 1 --branch "$REF" https://github.com/vllm-project/vllm.git "$DEST"
fi

echo "vLLM at $DEST"
git -C "$DEST" log -1 --format='  ref   %d%n  commit %H%n  date   %cd' --date=short
echo
echo "verify the symbols this patch hooks still exist:"
grep -n '^class GroupCoordinator' "$DEST/vllm/distributed/parallel_state.py"
grep -n '    def isend_tensor_dict\|    def irecv_tensor_dict' "$DEST/vllm/distributed/parallel_state.py"
grep -n 'class PPHandler\|    def receive\|    def broadcast\|    def get_prev_sampled_outputs' \
  "$DEST/vllm/v1/worker/gpu/pp_utils.py"
