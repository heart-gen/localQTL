from __future__ import annotations
from typing import Any, Optional, Union, Iterable
import numpy as np
import torch

TensorLike = Union[torch.Tensor, torch.nn.Parameter]
DeviceLike = Union[str, torch.device]

__all__ = ["resolve_device", "to_device_tensor", "move_to_device"]

def resolve_device(device: DeviceLike) -> torch.device:
    """Normalize device specifications to a :class:`torch.device`."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


def to_device_tensor(
        x: ArrayLike, device: DeviceLike | None = None, *,
        dtype: Optional[torch.dtype] = torch.float32,
        non_blocking: Optional[bool] = None,
        copy: bool = False,
) -> torch.Tensor:
    """
    Convert array-like to torch.Tensor, enforce dtype, and move to device.
    Fast-paths avoid redundant copies/moves.

    - Accepts torch.Tensor, numpy.ndarray, lists/tuples.
    - Defaults to float32 (good for GPU math).
    - non_blocking defaults to True for cuda, else False.
    """
    dev = resolve_device(device)

    if isinstance(x, torch.Tensor):
        t = x.clone() if copy else x
        need_move = (t.device != dev)
        need_dtype = (dtype is not None and t.dtype != dtype)
        if need_move or need_dtype:
            if non_blocking is None:
                non_blocking = (dev.type == "cuda")
            t = t.to(device=dev, dtype=(dtype or t.dtype), non_blocking=non_blocking)
        return t

    # Prefer asarray -> from_numpy for zero-copy when dtype already matches
    if isinstance(x, np.ndarray):
        np_dtype = (
            np.float32 if dtype in (None, torch.float32) else
            np.float64 if dtype == torch.float64 else
            np.float16 if dtype == torch.float16 else
            None
        )
        arr = x if (np_dtype is None or x.dtype == np_dtype) else x.astype(np_dtype, copy=not copy)
        t = torch.from_numpy(arr)
    else:
        # Generic python sequences
        t = torch.as_tensor(x)

    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype=dtype)

    if t.device != dev:
        if non_blocking is None:
            non_blocking = (dev.type == "cuda")
        t = t.to(device=dev, non_blocking=non_blocking)
    return t


def move_to_device(
        x: TensorLike, device: DeviceLike, *,
        non_blocking: Optional[bool] = None, copy: bool = False,
) -> torch.Tensor:
    """
    Thin wrapper over `_to_device_tensor` that moves `x` to `device`
    without changing dtype.

    Parameters
    ----------
    x : TensorLike
        A torch.Tensor or array-like.
    device : DeviceLike
        Target device (e.g., "cuda", "cpu", torch.device(...)).
    non_blocking : bool, optional
        If None, defaults to True when moving to CUDA, else False.
    copy : bool, default False
        If True and `x` is a tensor, clone before moving.

    Returns
    -------
    torch.Tensor
        Tensor on `device` with original dtype preserved.
    """
    return _to_device_tensor(
        x, device=device, dtype=None,
        non_blocking=non_blocking, copy=copy,
    )
