"""Exact finite (T,k) revelation costs; route bits 0=-1, 1=+1.

Route ID is a binary integer in chronological order. For any route ``i``, its
first T movements have ID ``i >> k``. ``Q2`` below means ``2*Q``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _check(T: int, k: int) -> int:
    if T < 1 or k < 0 or T + k > 30:
        raise ValueError("require T>=1, k>=0, T+k<=30")
    return 1 << (T + k)


def ordered_cost_path(i: int, j: int, T: int, k: int) -> int:
    """Independent direct physical simulation, with i initially left of j."""
    n = _check(T, k)
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError("route ID out of range")
    H = T + k
    moves_i = tuple(1 if i & (1 << (H - t)) else -1 for t in range(1, H + 1))
    moves_j = tuple(1 if j & (1 << (H - t)) else -1 for t in range(1, H + 1))
    if moves_i[:T] == moves_j[:T]:
        return T
    left, right = 0, 2
    for t, (vi, vj) in enumerate(zip(moves_i, moves_j), start=1):
        left += vi
        right += vj
        if left == right:
            return t
        if left > right:
            raise ArithmeticError("crossed without meeting")
    return H + (right - left) // 2


def ordered_cost_bitcount(i: int, j: int, T: int, k: int) -> int:
    """Independent prefix-popcount first-passage computation."""
    n = _check(T, k)
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError("route ID out of range")
    H = T + k
    if i >> k == j >> k:
        return T
    for t in range(1, H + 1):
        ones_i = (i >> (H - t)).bit_count()
        ones_j = (j >> (H - t)).bit_count()
        if ones_i - ones_j == 1:
            return t
        if ones_i - ones_j > 1:
            raise ArithmeticError("crossed without meeting")
    return H + 1 + j.bit_count() - i.bit_count()


def pair_exact(i: int, j: int, T: int, k: int, ordered_cost=ordered_cost_bitcount) -> tuple[int, int]:
    """Return (Q2_ij, B_ij), both exact integers."""
    return (ordered_cost(i, j, T, k) + ordered_cost(j, i, T, k),
            int(i >> k != j >> k))


@dataclass(frozen=True)
class BlockBuilder:
    """Streaming vectorized first-passage generator independent of popcounts."""

    T: int
    k: int

    def __post_init__(self):
        n = _check(self.T, self.k)
        H = self.T + self.k
        ids = np.arange(n, dtype=np.uint32)
        shifts = np.arange(H - 1, -1, -1, dtype=np.uint32)
        steps = ((ids[:, None] >> shifts[None, :]) & 1).astype(np.int16) * 2 - 1
        object.__setattr__(self, "displacements", np.cumsum(steps, axis=1, dtype=np.int16))

    @property
    def n(self) -> int:
        return 1 << (self.T + self.k)

    def block(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """Rows [start,stop), all columns; Q2 uint8 and B bool, no dense retention."""
        if not (0 <= start <= stop <= self.n):
            raise ValueError("invalid row range")
        rows = np.arange(start, stop, dtype=np.int32)
        cols = np.arange(self.n, dtype=np.int32)
        shape = (len(rows), self.n)
        ordered = np.zeros(shape, dtype=np.int16)
        reversed_order = np.zeros(shape, dtype=np.int16)
        alive = np.ones(shape, dtype=bool)
        alive_reverse = np.ones(shape, dtype=bool)
        for t in range(self.T + self.k):
            difference = (self.displacements[None, :, t]
                          - self.displacements[rows, None, t])
            meet = alive & (difference == -2)
            ordered[meet] = t + 1
            alive[meet] = False
            meet_reverse = alive_reverse & (difference == 2)
            reversed_order[meet_reverse] = t + 1
            alive_reverse[meet_reverse] = False
        final_difference = (self.displacements[None, :, -1]
                            - self.displacements[rows, None, -1])
        ordered[alive] = self.T + self.k + 1 + final_difference[alive] // 2
        reversed_order[alive_reverse] = self.T + self.k + 1 - final_difference[alive_reverse] // 2
        B = (rows[:, None] >> self.k) != (cols[None, :] >> self.k)
        ordered[~B] = self.T
        reversed_order[~B] = self.T
        Q2 = ordered + reversed_order
        if np.any(Q2 < 0) or np.any(Q2 > 4 * (self.T + self.k) + 2):
            raise ArithmeticError("cost outside analytical range")
        return Q2.astype(np.uint8), B


def quadratic_stream(builder: BlockBuilder, weights: np.ndarray, block_rows: int = 64) -> tuple[float, float]:
    """Return (p^T Q2 p, p^T B p), for any float probability vector p."""
    p = np.asarray(weights, dtype=np.float64)
    if p.shape != (builder.n,) or np.min(p) < 0 or not np.isclose(p.sum(), 1):
        raise ValueError("weights must be a probability vector of the correct length")
    numerator = 0.0
    for first in range(0, builder.n, block_rows):
        last = min(first + block_rows, builder.n)
        Q2, _ = builder.block(first, last)
        numerator += float(p[first:last] @ (Q2 @ p))
    prefix_mass = p.reshape((1 << builder.T), (1 << builder.k)).sum(axis=1)
    denominator = float(1.0 - prefix_mass @ prefix_mass)
    return numerator, denominator
