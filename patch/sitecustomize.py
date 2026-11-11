"""Opt-in transport shim loaded by Python when this directory is PYTHONPATH."""

import os

if os.environ.get("GAN_SHM_SPSC") == "1":
    import shm_transport  # noqa: F401

