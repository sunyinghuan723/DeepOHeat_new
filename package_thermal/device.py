"""Device selection helpers for package-level thermal surrogate scripts."""

from __future__ import annotations

import warnings

import torch


def resolve_device(requested: str | None) -> torch.device:
    """Resolve cpu/cuda/cuda:N/auto with explicit errors for unavailable GPUs."""

    value = (requested or "auto").strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        warnings.warn(
            "[PACKAGE-THERMAL] --device auto selected CPU because CUDA is not available",
            RuntimeWarning,
        )
        return torch.device("cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested --device cuda but torch.cuda.is_available() is false")
        return torch.device("cuda:0")
    if value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"requested --device {value} but torch.cuda.is_available() is false"
            )
        try:
            index = int(value.split(":", 1)[1])
        except ValueError as exc:
            raise RuntimeError(f"invalid CUDA device string: {requested}") from exc
        count = torch.cuda.device_count()
        if index < 0 or index >= count:
            raise RuntimeError(
                f"requested --device {value} but only {count} CUDA device(s) are visible"
            )
        return torch.device(value)
    raise RuntimeError(
        f"unsupported --device {requested!r}; use cpu, cuda, cuda:0, cuda:1, or auto"
    )


def device_metadata(device: torch.device) -> dict:
    """Return JSON-serializable metadata for logs and reports."""

    info = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpu_name": "",
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device.index or 0)
    return info


def log_device(prefix: str, device: torch.device) -> None:
    meta = device_metadata(device)
    gpu = f" gpu_name={meta['gpu_name']}" if meta["gpu_name"] else ""
    print(f"{prefix} device={meta['device']}{gpu}")
