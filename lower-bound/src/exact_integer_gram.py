#!/usr/bin/env python3
"""Exact integer Gram products, with a bounded-integer float64 BLAS engine.

This module does not approximate integer products.  Its BLAS engine splits each
input integer into base-2**19 digits.  Before each dot product it verifies that
the sum of absolute products is less than 2**53.  Every product and every partial
sum is then an exactly representable binary64 integer, so ordinary IEEE-754
binary64 multiplication, addition, and fused multiply-add give the integer
answer, in any accumulation order.  A signed-int64 engine uses base-2**24 digits
and separately checks all products and accumulations against 2**63.

No numerical eigenvalue, tolerance, or rounded factorization is used here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


BITS = 19
BASE = 1 << BITS


def maximum_absolute(a: np.ndarray) -> int:
    if not a.size:
        return 0
    return max(abs(int(a.min())), abs(int(a.max())))


def split_signed(a: np.ndarray, bits: int = BITS) -> list[np.ndarray]:
    """Return exact signed most-significant / unsigned lower base-2**bits digits."""
    if not 1 <= bits <= 30:
        raise ValueError("Digit width must be between 1 and 30")
    if a.dtype.kind not in "iu" or a.ndim != 2:
        raise ValueError("Expected a two-dimensional integer matrix")
    if a.dtype.kind == "u" and int(a.max(initial=0)) >= 1 << 63:
        raise ValueError("Unsigned entry does not fit signed int64")
    a = a.astype(np.int64, copy=True)
    base = 1 << bits
    digits = []
    while maximum_absolute(a) >= base:
        digits.append(a & (base - 1))
        a >>= bits
    digits.append(a)
    return digits


def combine_digits(digits: list[np.ndarray], bits: int = BITS) -> np.ndarray:
    """Evaluate sum(digits[k] * 2**(bits*k)) using unbounded Python integers."""
    result = digits[-1].astype(object)
    for digit in reversed(digits[:-1]):
        result = result * (1 << bits) + digit.astype(object)
    return result


class ExactGram:
    """Prepare a fixed integer matrix, then evaluate row blocks of A A^T."""

    def __init__(self, a: np.ndarray, engine: str = "blas"):
        if engine not in {"blas", "int64"}:
            raise ValueError("engine must be blas or int64")
        self.engine = engine
        self.bits = 19 if engine == "blas" else 24
        self.shape = a.shape
        self.digits = split_signed(a, self.bits)
        self.maxima = [maximum_absolute(d) for d in self.digits]
        self.width = a.shape[1]
        self.bounds = [[self.width * u * v for v in self.maxima]
                       for u in self.maxima]
        limit = 1 << (53 if engine == "blas" else 63)
        if any(bound >= limit for row in self.bounds for bound in row):
            raise ValueError("The exact dot-product accumulation bound fails")
        # Each convolution coefficient is an int64 sum of limb products.
        counts = [sum(self.bounds[i][j]
                      for i in range(len(self.digits))
                      for j in range(len(self.digits)) if i + j == k)
                  for k in range(2 * len(self.digits) - 1)]
        if max(counts) >= 1 << 63:
            raise ValueError("The exact convolution accumulation bound fails")
        self.convolution_bounds = counts
        if engine == "blas":
            self.operands = [d.astype(np.float64) for d in self.digits]
            # Every digit is at most 2**19-1 in absolute value.
            if any(not np.array_equal(x.astype(np.int64), d)
                   for x, d in zip(self.operands, self.digits)):
                raise ArithmeticError("Integer-to-binary64 conversion changed a digit")
        else:
            self.operands = self.digits
        self.blocks_computed = 0
        self.limb_products_computed = 0

    def block_digits(self, lo: int, hi: int, record: bool = True) -> list[np.ndarray]:
        """Unnormalised exact convolution coefficients for (A A^T)[lo:hi]."""
        if not 0 <= lo <= hi <= self.shape[0]:
            raise ValueError("Invalid row block")
        out = [np.zeros((hi - lo, self.shape[0]), dtype=np.int64)
               for _ in self.convolution_bounds]
        for i, left in enumerate(self.operands):
            for j, right in enumerate(self.operands):
                product = left[lo:hi] @ right.T
                if self.engine == "blas":
                    if product.dtype != np.dtype(np.float64):
                        raise ArithmeticError("The BLAS engine did not return binary64")
                    if not np.isfinite(product).all():
                        raise ArithmeticError("Nonfinite exact-integer product")
                    # These are defensive implementation checks, not tolerances.
                    if not np.equal(product, np.trunc(product)).all():
                        raise ArithmeticError("A purported exact product is nonintegral")
                    if maximum_absolute(product) > self.bounds[i][j]:
                        raise ArithmeticError("A product exceeds its proved bound")
                    product = product.astype(np.int64)
                out[i + j] += product
                if record:
                    self.limb_products_computed += 1
        if record:
            self.blocks_computed += 1
        return out

    def block(self, lo: int, hi: int, record: bool = True) -> np.ndarray:
        """Return exact Python-integer Gram entries for a contiguous row block."""
        return combine_digits(self.block_digits(lo, hi, record=record), self.bits)

    def metadata(self) -> dict:
        return {
            "engine": self.engine,
            "shape": list(self.shape),
            "digit_bits": self.bits,
            "digit_count": len(self.digits),
            "digit_maximum_absolute": self.maxima,
            "largest_dot_product_absolute_sum_bound": max(map(max, self.bounds)),
            "largest_convolution_absolute_sum_bound": max(self.convolution_bounds),
            "blocks_computed": self.blocks_computed,
            "limb_products_computed": self.limb_products_computed,
            "exactness": ("Every multiply and every possible partial sum is an integer "
                          "of absolute value below 2**53" if self.engine == "blas"
                          else "Every multiply and partial sum fits signed int64"),
        }


def self_test() -> dict:
    """Independent Python-integer products test signed digits and large entries."""
    cases = [
        np.array([[0, 1, -1], [-BASE, BASE - 1, BASE],
                  [2**52 + 7, -(2**57 + 11), 1234567890123456]], dtype=np.int64),
        np.array([[2**63 - 1, -(2**63)],
                  [-(2**63), 2**63 - 1]], dtype=np.int64),
        np.array([[-(1 << 24), (1 << 24) - 1, 1 << 24],
                  [(1 << 48) - 1, -(1 << 48), (1 << 48) + 1]], dtype=np.int64),
        np.empty((2, 0), dtype=np.int64),
    ]
    rng = np.random.default_rng(217699)
    for bits, n, m in [(0, 11, 7), (18, 47, 29), (38, 67, 53), (59, 31, 81)]:
        cases.append(rng.integers(-(1 << bits), 1 << bits, (n, m), dtype=np.int64))
    checks = 0
    for a in cases:
        expected = a.astype(object) @ a.astype(object).T
        for engine in ("blas", "int64"):
            gram = ExactGram(a, engine)
            if not np.array_equal(combine_digits(gram.digits, gram.bits), a):
                raise ArithmeticError("Signed digit reconstruction failed")
            got = np.vstack([gram.block(lo, min(lo + 9, len(a)))
                             for lo in range(0, len(a), 9)])
            if not np.array_equal(got, expected):
                raise ArithmeticError("Independent exact Gram equality failed")
            checks += got.size
    return {"status": "PASS", "cases": len(cases), "engines": ["blas", "int64"],
            "exact_entry_equalities": checks,
            "reference": "NumPy object products using arbitrary-precision Python integers"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--benchmark-size", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    result = self_test()
    if args.benchmark_size:
        n = args.benchmark_size
        rng = np.random.default_rng(882147)
        a = np.tril(rng.integers(-(1 << 51), 1 << 51, (n, n), dtype=np.int64))
        timer = time.perf_counter()
        gram = ExactGram(a)
        product = gram.block(0, min(128, n))
        result["benchmark"] = {**gram.metadata(), "seconds": time.perf_counter() - timer,
                               "returned_entries": product.size}
    result["seconds"] = time.perf_counter() - start
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
