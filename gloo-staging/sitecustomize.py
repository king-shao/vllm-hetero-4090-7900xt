"""Experimental CUDA/ROCm vLLM pipeline-parallel compatibility shim.

Enabled only when VLLM_CROSS_VENDOR_GLOO=1. It forces vLLM process groups to
use Gloo and stages pipeline point-to-point tensors through CPU memory.
"""

from __future__ import annotations

import os


if os.environ.get("VLLM_CROSS_VENDOR_GLOO") == "1":
    import pickle

    import torch

    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )
    from vllm.distributed.parallel_state import (
        GroupCoordinator,
        TensorMetadata,
        _split_tensor_dict,
    )
    from vllm.platforms import current_platform

    # CUDA and ROCm both call their native backend "nccl", but NVIDIA NCCL and
    # AMD RCCL cannot create one cross-vendor communicator.
    type(current_platform).dist_backend = "gloo"
    current_platform.dist_backend = "gloo"

    def _cpu_staged_send(
        self: CudaCommunicator,
        tensor: torch.Tensor,
        dst: int | None = None,
    ) -> None:
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        cpu_tensor = tensor.detach().to(device="cpu", non_blocking=False).contiguous()
        torch.distributed.send(cpu_tensor, self.ranks[dst], self.cpu_group)

    def _cpu_staged_recv(
        self: CudaCommunicator,
        size: torch.Size,
        dtype: torch.dtype,
        src: int | None = None,
    ) -> torch.Tensor:
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        cpu_tensor = torch.empty(size, dtype=dtype, device="cpu")
        torch.distributed.recv(cpu_tensor, self.ranks[src], self.cpu_group)
        return cpu_tensor.to(device=self.device, non_blocking=False)

    def _writable_send_object(
        self: GroupCoordinator,
        obj: object,
        dst: int,
    ) -> None:
        """Send metadata through Gloo using writable CPU storage.

        vLLM's default torch.frombuffer(pickle.dumps(...)) tensor is backed by
        immutable bytes. With the CUDA and ROCm PyTorch builds used here,
        Gloo's writev fails with EFAULT ("Bad address") on that storage.
        """
        if dst >= self.world_size or dst == self.rank_in_group:
            raise ValueError(f"invalid destination rank: {dst}")
        object_tensor = torch.frombuffer(
            bytearray(pickle.dumps(obj)), dtype=torch.uint8
        ).clone()
        size_tensor = torch.tensor([object_tensor.numel()], dtype=torch.long)
        torch.distributed.send(size_tensor, self.ranks[dst], self.cpu_group)
        torch.distributed.send(object_tensor, self.ranks[dst], self.cpu_group)

    def _cpu_staged_isend_tensor_dict(
        self: GroupCoordinator,
        tensor_dict: dict[str, object],
        dst: int | None = None,
        all_gather_group: GroupCoordinator | None = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> list[object]:
        """Synchronous TP=1 pipeline send through writable CPU tensors."""
        del all_gather_group, all_gather_tensors
        if self.world_size <= 1:
            return []
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        if dst >= self.world_size:
            raise ValueError(f"invalid destination rank: {dst}")

        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        self.send_object(metadata_list, dst=dst)
        for tensor in tensor_list:
            if tensor.numel() == 0:
                continue
            cpu_tensor = tensor.detach().to("cpu", non_blocking=False).contiguous()
            torch.distributed.send(cpu_tensor, self.ranks[dst], self.cpu_group)
        return []

    def _cpu_staged_irecv_tensor_dict(
        self: GroupCoordinator,
        src: int | None = None,
        all_gather_group: GroupCoordinator | None = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> tuple[dict[str, object] | None, list[object], list[object]]:
        """Synchronous TP=1 pipeline receive through writable CPU tensors."""
        del all_gather_group, all_gather_tensors
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None, [], []
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        if src >= self.world_size:
            raise ValueError(f"invalid source rank: {src}")

        metadata_list = self.recv_object(src=src)
        tensor_dict: dict[str, object] = {}
        for key, value in metadata_list:
            if isinstance(value, TensorMetadata):
                if value.size.numel() == 0:
                    tensor_dict[key] = torch.empty(
                        value.size, dtype=value.dtype, device=value.device
                    )
                    continue
                cpu_tensor = torch.empty(value.size, dtype=value.dtype, device="cpu")
                torch.distributed.recv(
                    cpu_tensor, self.ranks[src], self.cpu_group
                )
                tensor_dict[key] = cpu_tensor.to(
                    device=value.device, non_blocking=False
                )
            else:
                tensor_dict[key] = value
        return tensor_dict, [], []

    CudaCommunicator.send = _cpu_staged_send
    CudaCommunicator.recv = _cpu_staged_recv
    GroupCoordinator.send_object = _writable_send_object
    GroupCoordinator.isend_tensor_dict = _cpu_staged_isend_tensor_dict
    GroupCoordinator.irecv_tensor_dict = _cpu_staged_irecv_tensor_dict
