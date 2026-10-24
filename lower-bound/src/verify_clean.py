#!/usr/bin/env python3
"""Independent, complete exact Gram-residual PSD certificate verification.

The source matrix is reconstructed from locked rational data. No generator,
old inverse witness, numerical eigenvalue, or PSD property of a source term is
trusted. The default arithmetic engine uses guarded signed integers only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))


def peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, AttributeError):
        return None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(args):
    started = time.perf_counter()
    if not __debug__:
        raise RuntimeError("Run without Python -O or PYTHONOPTIMIZE")
    if args.workers < 1 or args.block < 1:
        raise ValueError("workers and block must be positive")
    # Set before importing NumPy. The integer engine does not use BLAS.
    os.environ["OPENBLAS_NUM_THREADS"] = "4" if args.engine == "blas" else "1"
    os.environ["MKL_NUM_THREADS"] = "4" if args.engine == "blas" else "1"
    import numpy as np
    from exact_integer_gram import ExactGram
    from exact_reconstruct import LockedMatrix

    root = args.root.resolve()
    factor_path = root / "witness" / "L_lower.npy"
    partition_path = root / "witness" / "partition.json"
    partition = json.loads(partition_path.read_text())
    if args.negative_control == "wrong_partition":
        partition["classes"][0][0], partition["classes"][1][0] = (
            partition["classes"][1][0], partition["classes"][0][0])

    matrix = LockedMatrix(root, progress=True)
    actual_partition = {"empty": matrix.empty, "classes": matrix.classes}
    if partition != actual_partition:
        raise ArithmeticError("PARTITION_IDENTITY_FAILED: classes do not match locked P")
    n = matrix.order
    if n != 4192 or len(matrix.empty) != 540:
        raise ArithmeticError("UNEXPECTED_LOCKED_MATRIX_DIMENSIONS")
    packed = np.load(factor_path, allow_pickle=False)
    if packed.dtype != np.dtype(np.int64) or packed.shape != (n*(n+1)//2,):
        raise ValueError("FACTOR_FORMAT: expected 8,788,528 signed-int64 lower-triangle entries")
    factor = np.zeros((n, n), dtype=np.int64)
    cursor = 0
    for i in range(n):
        factor[i, :i+1] = packed[cursor:cursor+i+1]
        cursor += i+1
    del packed
    if args.negative_control == "zero_factor":
        factor[:] = 0
    elif args.negative_control == "large_factor":
        bound = max(abs(int(factor.min())), abs(int(factor.max())))
        if 2*bound >= 1 << 63:
            raise ArithmeticError("NEGATIVE_CONTROL_FACTOR_WOULD_OVERFLOW")
        factor *= 2
    factor.flags.writeable = False

    exponent = 46
    square_grid = 1 << (2*exponent)
    denominator = math.lcm(matrix.denominator, square_grid)
    h_multiplier = denominator // matrix.denominator
    gram_multiplier = denominator // square_grid
    raw_multiplier = denominator // matrix.base_denominator
    threshold_denominator = 1 << 31
    contrast_checks = 0
    if not args.physical_residual:
        for rows in matrix.classes:
            if len(rows) > 1:
                surplus = 11*matrix.base_denominator - matrix.raw_diagonal[rows[0]]
                if surplus*threshold_denominator <= matrix.base_denominator:
                    raise ArithmeticError("CONTRAST_SURPLUS_NOT_GREATER_THAN_2^-31")
                contrast_checks += 1

    gram = ExactGram(factor, engine=args.engine)
    for digit in gram.digits:
        digit.flags.writeable = False
    for operand in gram.operands:
        operand.flags.writeable = False
    multiplicities = list(map(int, matrix.multiplicities))

    def check_block(lo, hi):
        g = gram.block(lo, hi, record=False)
        h = matrix.block(np.arange(lo, hi, dtype=np.int64))
        residual = h*h_multiplier - g*gram_multiplier
        values = []
        for local, i in enumerate(range(lo, hi)):
            row = residual[local]
            if args.physical_residual:
                weighted_sum = sum(m*abs(int(z)) for m, z in zip(multiplicities, row))
                cross_sum = weighted_sum - multiplicities[i]*abs(int(row[i]))
                gram_diagonal = int(g[local, i])*gram_multiplier
                diagonal = 11*denominator - gram_diagonal
                raw = matrix.raw_diagonal[matrix.classes[i][0]]*raw_multiplier
                within = raw - gram_diagonal
                slack = diagonal - (multiplicities[i]-1)*abs(within) - cross_sum
            else:
                diagonal = int(row[i])
                slack = diagonal - sum(abs(int(z)) for z in row) + abs(diagonal)
            if slack*threshold_denominator <= denominator:
                raise ArithmeticError(
                    f"RESIDUAL_ROW_NOT_GREATER_THAN_2^-31: row={i}, "
                    f"slack={Fraction(slack, denominator)}")
            values.append((i, slack))
        return values

    ranges = [(lo, min(lo+args.block, n)) for lo in range(0, n, args.block)]
    slacks = [None]*n
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = []
    completed = 0
    last_progress = time.perf_counter()
    try:
        futures = [pool.submit(check_block, lo, hi) for lo, hi in ranges]
        for future in as_completed(futures):
            for i, slack in future.result():
                if slacks[i] is not None:
                    raise ArithmeticError("DUPLICATE_VERIFICATION_ROW")
                slacks[i] = slack
                completed += 1
            now = time.perf_counter()
            if now-last_progress >= 15 or completed == n:
                print(f"EXACT_RESIDUAL_ROWS {completed}/{n} seconds={now-started:.3f}", flush=True)
                last_progress = now
    finally:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    if completed != n or any(value is None for value in slacks):
        raise ArithmeticError("INCOMPLETE_VERIFICATION: no PSD claim is permitted")

    if args.physical_residual:
        physical_slacks = [None]*8192
        for rows, value in zip(matrix.classes, slacks):
            for i in rows:
                physical_slacks[i] = value
        for i in matrix.empty:
            physical_slacks[i] = 11*denominator
        if any(value is None for value in physical_slacks):
            raise ArithmeticError("INCOMPLETE_PHYSICAL_ROW_COVERAGE")
        reported_slacks = physical_slacks
    else:
        reported_slacks = slacks
    winner = min(range(len(reported_slacks)), key=lambda i: reported_slacks[i])
    metadata = gram.metadata()
    metadata["blocks_computed"] = len(ranges)
    metadata["limb_products_computed"] = len(ranges)*len(gram.digits)**2
    result = {
        "status": "VERIFIED_CLEAN_CERTIFICATE",
        "method": ("direct_physical_gram_and_diagonally_dominant_residual" if args.physical_residual
                   else "exact_twin_quotient_gram_and_diagonally_dominant_residual"),
        "target_exact": "211571753466829574204178595602447/50000000000000000000000000000000",
        "locked_order": 8192, "factor_order": n,
        "physical_factor_shape": [8192, n] if args.physical_residual else None,
        "factor_dyadic_exponent": exponent,
        "packed_integer_entries": n*(n+1)//2,
        "nonempty_classes": n, "empty_vertices": len(matrix.empty),
        "contrast_class_inequalities": contrast_checks,
        "distinct_residual_rows_checked": completed,
        "residual_row_inequalities_certified": len(reported_slacks),
        "all_ordered_reduced_residual_entries_evaluated": n*n,
        "strict_row_slack_lower_bound": "1/2147483648",
        "minimum_row_slack_exact": str(Fraction(reported_slacks[winner], denominator)),
        "minimum_row_slack_index": winner,
        "row_slack_common_denominator": str(denominator),
        "row_slack_numerators": [str(value) for value in reported_slacks],
        "matrix_reconstruction": matrix.summary(),
        "gram_arithmetic": metadata,
        "workers": args.workers, "row_block_size": args.block,
        "runtime_seconds": time.perf_counter()-started,
        "peak_rss_bytes": peak_rss_bytes(),
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__, "arithmetic_backend": args.engine,
        "blas_thread_environment": os.environ.get("OPENBLAS_NUM_THREADS"),
        "factor_sha256": sha(factor_path), "partition_sha256": sha(partition_path),
        "verifier_sha256": sha(Path(__file__)),
        "negative_control": args.negative_control,
        "independence": "Reconstructs locked rational P; no certificate generator or old inverse witness is imported",
        "complete": True,
    }
    if args.negative_control:
        raise ArithmeticError("NEGATIVE_CONTROL_UNEXPECTEDLY_PASSED")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--engine", choices=["int64", "blas"], default="int64")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--block", type=int, default=64)
    method = parser.add_mutually_exclusive_group()
    method.add_argument("--physical-residual", dest="physical_residual", action="store_true")
    method.add_argument("--quotient-residual", dest="physical_residual", action="store_false")
    parser.set_defaults(physical_residual=True)
    parser.add_argument("--negative-control", choices=["zero_factor", "large_factor", "wrong_partition"])
    args = parser.parse_args()
    if args.workers is None:
        args.workers = 1 if args.engine == "blas" else 4
    out = args.out or args.root / "results" / "verification.json"
    started = time.perf_counter()
    try:
        result = verify(args)
        exit_code = 0
    except Exception as error:
        result = {
            "status": "REJECTED", "complete": False,
            "error_type": type(error).__name__, "error": str(error),
            "negative_control": args.negative_control,
            "runtime_seconds": time.perf_counter()-started,
            "peak_rss_bytes": peak_rss_bytes(),
            "traceback": traceback.format_exc(),
            "psd_claim": "No PSD claim is made by this failed run",
        }
        exit_code = 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+"\n")
    concise = {k: v for k, v in result.items()
               if k not in {"row_slack_numerators", "traceback", "matrix_reconstruction"}}
    print(json.dumps(concise, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
