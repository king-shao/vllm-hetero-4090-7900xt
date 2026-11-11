#!/usr/bin/env bash
# Build libppshm.so + libppdma.so. Must run on Linux (host or inside a container).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/native}"

case "$(uname -s)" in
  Linux) ;;
  *) echo "ERROR: the native libraries are Linux/glibc only (found $(uname -s))." >&2
     echo "       Build them inside the vLLM container, or on any Linux host." >&2
     exit 1 ;;
esac

make -C "$ROOT/native" BUILD="$BUILD_DIR" all
ls -l "$BUILD_DIR"/libppshm.so "$BUILD_DIR"/libppdma.so
