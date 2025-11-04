from __future__ import annotations

from typing import Optional, Union

import torch

TensorLike = Union[torch.Tensor, torch.nn.Parameter]
DeviceLike = Union[str, torch.device]

__all__ = ["resolve_device", "to_device_tensor", "move_to_device"]


def resolve_device(device: DeviceLike | None) -> torch.device:
    """Normalize device specifications to a :class:`torch.device`."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


def to_device_tensor(
    x, device: str | torch.device | None, *, dtype: Optional[torch.dtype] = None,
    non_blocking: bool | None = None,
) -> torch.Tensor:
    dev = resolve_device(device)
    t = torch.as_tensor(x, dtype=dtype)
    if dev.type == "cuda" and t.device.type == "cpu":
        t = t.pin_memory()
    if non_blocking is None:
        non_blocking = (dev.type == "cuda")
    return t.to(dev, non_blocking=non_blocking)


def move_to_device(
    x: TensorLike, device: DeviceLike | None, *, non_blocking: Optional[bool] = None,
) -> torch.Tensor:
    """Move `x` to `device` without changing dtype."""
    if isinstance(x, torch.Tensor):
        dtype = x.dtype
    else:
        dtype = None
    return to_device_tensor(x, device, dtype=dtype, non_blocking=non_blocking)
