from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable

import torch

from ..utils import subseed
from ..regression_kernels import (
    run_batch_regression_with_permutations,
    prep_ctx_for_perm,
    perm_chunk_r2,
)

__all__ = [
    "PermutationStream",
    "perm_chunk_generator",
    "compute_perm_r2_max",
]


def _combine_seed(seed: int | None, key: str | int | None) -> int | None:
    if seed is None:
        return None
    if key is None:
        return int(seed)
    return subseed(int(seed), key)


@dataclass
class PermutationStream:
    """Stateful permutation chunk generator living on the requested device."""

    n_samples: int
    nperm: int
    device: torch.device
    chunk_size: int
    seed: int | None = None
    mode: str = "generator"

    def __post_init__(self) -> None:
        self.n_samples = int(self.n_samples)
        self.nperm = int(self.nperm)
        self.chunk_size = max(1, int(self.chunk_size))
        self.mode = (self.mode or "generator").lower()
        if self.mode not in {"generator", "cpu_pinned", "gpu"}:
            self.mode = "generator"

        self._generated = 0
        self._cpu_gen = torch.Generator(device="cpu")
        if self.seed is not None:
            self._cpu_gen.manual_seed(self.seed)

        self._gpu_gen: torch.Generator | None = None
        if self.mode == "gpu" and self.device.type == "cuda":
            self._gpu_gen = torch.Generator(device=self.device)
            if self.seed is not None:
                self._gpu_gen.manual_seed(self.seed)

    @property
    def generated(self) -> int:
        return self._generated

    @property
    def remaining(self) -> int:
        return max(self.nperm - self._generated, 0)

    def reset(self) -> None:
        """Reset the internal generator to the initial seed."""
        self._generated = 0
        if self.seed is not None:
            self._cpu_gen.manual_seed(self.seed)
            if self._gpu_gen is not None:
                self._gpu_gen.manual_seed(self.seed)

    def _generate_cpu_chunk(self, count: int) -> torch.Tensor:
        rows = [torch.randperm(self.n_samples, generator=self._cpu_gen) for _ in range(count)]
        chunk = torch.stack(rows, dim=0)
        if self.mode == "cpu_pinned":
            chunk = chunk.pin_memory()
        if self.device.type != "cpu":
            chunk = chunk.to(self.device, non_blocking=True)
        return chunk

    def _generate_gpu_chunk(self, count: int) -> torch.Tensor:
        if self._gpu_gen is None:
            return self._generate_cpu_chunk(count)
        rows = [
            torch.randperm(self.n_samples, generator=self._gpu_gen, device=self.device)
            for _ in range(count)
        ]
        return torch.stack(rows, dim=0)

    def next_chunk(self, limit: int | None = None) -> torch.Tensor | None:
        if self._generated >= self.nperm:
            return None

        target = self.nperm - self._generated
        if limit is not None:
            target = min(target, int(limit))
        count = min(self.chunk_size, target)
        if count <= 0:
            return None

        if self.mode == "gpu" and self.device.type == "cuda":
            chunk = self._generate_gpu_chunk(count)
        else:
            chunk = self._generate_cpu_chunk(count)

        self._generated += int(chunk.shape[0])
        return chunk

    def iter_chunks(self, limit: int | None = None) -> Iterable[torch.Tensor]:
        produced = 0
        while True:
            if limit is not None and produced >= limit:
                break
            remaining = None if limit is None else limit - produced
            chunk = self.next_chunk(remaining)
            if chunk is None:
                break
            produced += int(chunk.shape[0])
            yield chunk


def perm_chunk_generator(
        n_samples: int, nperm: int, chunk_size: int, seed: int | None, key: str | None,
        device: torch.device, mode: str = "generator") -> Generator[torch.Tensor, None, None]:
    """Convenience generator yielding permutation index chunks on demand."""
    stream = PermutationStream(
        n_samples=n_samples,
        nperm=nperm,
        device=device,
        chunk_size=chunk_size,
        seed=_combine_seed(seed, key),
        mode=mode,
    )
    for chunk in stream.iter_chunks():
        yield chunk


@torch.no_grad()
def compute_perm_r2_max(
        y_resid: torch.Tensor,
        G_resid: torch.Tensor,
        H_resid: torch.Tensor | None,
        k_eff: int,
        perm_stream: PermutationStream,
        max_permutations: int | None = None,
        return_nominal: bool = False,
        mixed_precision: str | None = None,
) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
    """Run chunked permutations from a stream, optionally returning nominal stats and r²."""
    device = y_resid.device
    device_str = device.type if isinstance(device, torch.device) else str(device)
    y_resid = y_resid.contiguous()
    G_resid = G_resid.contiguous()
    if H_resid is not None:
        H_resid = H_resid.contiguous()

    remaining = perm_stream.remaining
    if remaining == 0:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        return (None, None, None, None, empty)

    target = remaining if max_permutations is None else min(remaining, int(max_permutations))
    if target <= 0:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        return (None, None, None, None, empty)

    n_samples = y_resid.shape[0]
    r2_perm_max = torch.empty((target,), device=device, dtype=torch.float32)
    offset = 0

    nominal_b = nominal_s = nominal_t = r2_nominal = None

    if H_resid is None:
        if return_nominal:
            nominal_b, nominal_s, nominal_t, _ = run_batch_regression_with_permutations(
                y=y_resid, G=G_resid, H=None, y_perm=None, k_eff=k_eff,
                device=device_str,
                mixed_precision=mixed_precision,
            )
            tvals = nominal_t[:, 0].double()
            t2 = tvals.pow(2)
            dof = max(n_samples - 1 - int(k_eff), 1)
            r2_nominal = (t2 / (t2 + float(dof))).to(torch.float32)

        for sel in perm_stream.iter_chunks(target):
            chunk = sel.shape[0]
            flat = sel.reshape(-1)
            y_perm = y_resid.index_select(0, flat).view(chunk, n_samples).transpose(0, 1)
            _, _, _, r2_block = run_batch_regression_with_permutations(
                y=y_resid, G=G_resid, H=None, y_perm=y_perm, k_eff=k_eff,
                device=device_str,
                mixed_precision=mixed_precision,
            )
            r2_perm_max[offset:offset + chunk] = r2_block.to(torch.float32)
            offset += chunk

    else:
        ctx, b_nom, s_nom, t_nom = prep_ctx_for_perm(
            y_resid, G_resid, H_resid, k_eff, mixed_precision=mixed_precision,
        )
        if return_nominal:
            nominal_b, nominal_s, nominal_t = b_nom, s_nom, t_nom
            tvals = nominal_t[:, 0].double()
            t2 = tvals.pow(2)
            dof = ctx.get("dof", max(n_samples - 1 - int(k_eff), 1))
            r2_nominal = (t2 / (t2 + float(dof))).to(torch.float32)

        for sel in perm_stream.iter_chunks(target):
            chunk = sel.shape[0]
            flat = sel.reshape(-1)
            y_perm = y_resid.index_select(0, flat).view(chunk, n_samples).transpose(0, 1)
            r2_block = perm_chunk_r2(
                ctx, H_resid, G_resid, y_perm, mixed_precision=mixed_precision
            )
            r2_perm_max[offset:offset + chunk] = r2_block.to(torch.float32)
            offset += chunk

    if offset < target:
        r2_perm_max = r2_perm_max[:offset]

    return nominal_b, nominal_s, nominal_t, r2_nominal, r2_perm_max
