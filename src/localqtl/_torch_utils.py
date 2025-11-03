from __future__ import annotations

import torch
from typing import Optional, Union

TensorLike = Union[torch.Tensor, torch.nn.Parameter]
DeviceLike = Union[str, torch.device]

__all__ = ["resolve_device", "move_to_device"]

def resolve_device(device: DeviceLike) -> torch.device:
    """Normalize device specifications to a :class:`torch.device`."""
    return device if isinstance(device, torch.device) else torch.device(device)


def move_to_device(
        tensor: TensorLike, device: torch.device, *,
        non_blocking: Optional[bool] = None,
) -> torch.Tensor:
    """Move ``tensor`` to ``device`` avoiding redundant copies.
    """
    if tensor.device == device:
        return tensor

    if non_blocking is None:
        non_blocking = device.type == "cuda"

    return tensor.to(device=device, non_blocking=non_blocking)
