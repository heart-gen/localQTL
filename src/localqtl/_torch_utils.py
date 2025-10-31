from __future__ import annotations

from typing import Optional, Union

import torch

TensorLike = Union[torch.Tensor, torch.nn.Parameter]
DeviceLike = Union[str, torch.device]


def resolve_device(device: DeviceLike) -> torch.device:
    """Normalize device specifications to a :class:`torch.device`."""
    return device if isinstance(device, torch.device) else torch.device(device)


def move_to_device(
    tensor: TensorLike,
    device: torch.device,
    *,
    non_blocking: Optional[bool] = None,
) -> torch.Tensor:
    """Move ``tensor`` to ``device`` avoiding redundant copies.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to move.
    device : torch.device
        Destination device.
    non_blocking : bool, optional
        When ``True`` and ``device`` is CUDA, allow asynchronous copies if the
        source tensor is in pinned memory. If ``None`` (default), the flag is
        enabled automatically for CUDA destinations.
    """
    if tensor.device == device:
        return tensor

    if non_blocking is None:
        non_blocking = device.type == "cuda"

    return tensor.to(device=device, non_blocking=non_blocking)


__all__ = ["resolve_device", "move_to_device"]
