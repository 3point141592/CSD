"""Runtime selection, deterministic precision policy, and CUDA diagnostics."""

from __future__ import annotations

import platform
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class RuntimeInfo:
    python: str
    platform: str
    torch_version: str
    cuda_available: bool
    torch_cuda_version: str | None
    device: str
    device_name: str
    dtype: str
    tf32_matmul: bool
    tf32_cudnn: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def configure_runtime(device: str | torch.device = "auto") -> torch.device:
    """Select CUDA when available and disable precision-changing shortcuts."""
    if str(device) == "auto":
        selected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if selected.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    return selected


def runtime_info(device: torch.device, dtype: torch.dtype) -> RuntimeInfo:
    cuda = torch.cuda.is_available()
    return RuntimeInfo(
        python=sys.version.replace("\n", " "),
        platform=platform.platform(),
        torch_version=torch.__version__,
        cuda_available=cuda,
        torch_cuda_version=torch.version.cuda,
        device=str(device),
        device_name=(
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        dtype=str(dtype).replace("torch.", ""),
        tf32_matmul=(bool(torch.backends.cuda.matmul.allow_tf32) if cuda else False),
        tf32_cudnn=(bool(torch.backends.cudnn.allow_tf32) if cuda else False),
    )


def print_runtime(info: RuntimeInfo) -> None:
    print(f"PyTorch version: {info.torch_version}")
    print(f"CUDA available: {info.cuda_available}")
    print(f"PyTorch CUDA version: {info.torch_cuda_version}")
    print(f"Selected device: {info.device}")
    print(f"GPU model: {info.device_name if info.cuda_available else 'none'}")
    print(f"dtype: {info.dtype}")
    print(f"TF32 matmul/cudnn: {info.tf32_matmul}/{info.tf32_cudnn}")


def assert_device(tensors: Iterable[torch.Tensor], device: torch.device) -> None:
    expected_index = (
        torch.cuda.current_device()
        if device.type == "cuda" and device.index is None
        else device.index
    )
    wrong = [
        str(tensor.device)
        for tensor in tensors
        if tensor.device.type != device.type
        or (device.type == "cuda" and tensor.device.index != expected_index)
    ]
    if wrong:
        raise RuntimeError(f"Major tensors are not on {device}: found {wrong[:5]}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
