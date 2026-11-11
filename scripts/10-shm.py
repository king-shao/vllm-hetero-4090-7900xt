#!/usr/bin/env python3
"""Create (or tear down) the two shared-memory rings used by the PP transport.

Both ranks must agree on GAN_SHM_SESSION; the rings are created once, from the
host, before either vLLM process starts.

    forward  : activation slots, default 2 x 8 MiB + 64 KiB metadata each
    feedback : sampled-token slots, default 2 x 1 MiB + 64 KiB metadata each

Run on the host (not inside a container) so the segment survives container
restarts and both containers can map the same /dev/shm inode.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ.get("GAN_SHM_PATCH", str(ROOT / "patch")))

os.environ.setdefault("GAN_SHM_LIBRARY", str(ROOT / "native" / "libppshm.so"))

from shm_native import create, unlink  # noqa: E402

PLAN = {
    "forward": dict(slots=2, meta=64 * 1024, payload=8 * 1024 * 1024),
    "feedback": dict(slots=2, meta=64 * 1024, payload=1024 * 1024),
}


def do_create(session: str) -> None:
    for direction in ("forward", "feedback"):
        p = PLAN[direction]
        try:
            create(session, direction, p["slots"], p["meta"], p["payload"])
            print(f"created {direction}: {p['slots']} slots, "
                  f"{p['payload'] // 1024 // 1024} MiB payload each")
        except BaseException:
            # Roll back a half-created pair so the next attempt starts clean.
            unlink(session, direction)
            raise


def do_unlink(session: str) -> None:
    for direction in ("forward", "feedback"):
        try:
            unlink(session, direction)
            print(f"unlinked {direction}")
        except FileNotFoundError:
            print(f"{direction}: not present")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=os.environ.get("GAN_SHM_SESSION", ""),
                    help="1-40 chars of [A-Za-z0-9_-]; must match on both ranks")
    ap.add_argument("--action", choices=["create", "unlink"], default="create")
    args = ap.parse_args()

    session = args.session
    if not session or not all(c.isalnum() or c in "_-" for c in session) or len(session) > 40:
        ap.error("--session must be 1-40 chars of [A-Za-z0-9_-]")

    if args.action == "create":
        do_create(session)
        print(f"SHM ready (session={session})")
    else:
        do_unlink(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
