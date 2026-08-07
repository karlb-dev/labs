"""Hard CUDA gate for every Phase 4 model-scale entrypoint."""
from __future__ import annotations

import subprocess

import torch


def require_cuda_gpu() -> dict:
    """Refuse CPU fallback and exercise CUDA in the calling process."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible in this process. HARD STOP: do not run "
            "model loading, inference, lens fitting, or scoring on CPU. "
            "Relaunch this exact command with host GPU access.")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    left = torch.randn((1024, 1024), device=device, dtype=torch.float16)
    right = torch.randn((1024, 1024), device=device, dtype=torch.float16)
    product = left @ right
    torch.cuda.synchronize(device)
    finite = bool(torch.isfinite(product).all().item())
    del left, right, product
    if not finite:
        raise RuntimeError("CUDA smoke operation produced non-finite values")
    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except Exception:
        driver = "unavailable"
    return {
        "torch_cuda_available": True,
        "device_index": 0,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "capability": [int(properties.major), int(properties.minor)],
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "driver_version": driver,
        "smoke_operation": "fp16-matmul-1024",
        "smoke_finite": True,
    }


def assert_model_on_cuda(model) -> str:
    try:
        parameter = next(model.parameters())
    except StopIteration as error:
        raise RuntimeError("model has no parameters to locate") from error
    if parameter.device.type != "cuda":
        raise RuntimeError(
            f"model parameter is on {parameter.device}; CPU fallback is "
            "forbidden")
    return str(parameter.device)
