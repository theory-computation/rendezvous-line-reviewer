#!/usr/bin/env python3
"""Optional exact cross-check using a separate direct physical-row formula.

Uses the packaged locked-matrix reconstruction and exact integer Gram engine;
does not import the primary verifier or a certificate generator. Accepts either
the packed lower triangle or a dense signed-int64 factor with denominator 2**46.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value*1024)
    except (ImportError, AttributeError):
        return None


def verify(root, witness=None, engine="blas", block_size=64):
    start = time.perf_counter()
    if not __debug__:
        raise RuntimeError("Run without Python -O or PYTHONOPTIMIZE")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    os.environ["OPENBLAS_NUM_THREADS"] = "4" if engine == "blas" else "1"
    os.environ["MKL_NUM_THREADS"] = "4" if engine == "blas" else "1"
    import numpy as np
    from exact_reconstruct import LockedMatrix
    from exact_integer_gram import ExactGram

    root = Path(root).resolve()
    witness = Path(witness) if witness is not None else Path("witness/L_lower.npy")
    if not witness.is_absolute():
        witness = root / witness
    matrix = LockedMatrix(root, progress=True)
    partition_path = root / "witness/partition.json"
    partition = json.loads(partition_path.read_text())
    if partition != {"empty": matrix.empty, "classes": matrix.classes}:
        raise ArithmeticError("PARTITION_IDENTITY_FAILED")
    n = matrix.order
    if n != 4192 or len(matrix.empty) != 540:
        raise ArithmeticError("UNEXPECTED_LOCKED_DIMENSIONS")
    stored = np.load(witness, allow_pickle=False)
    if stored.dtype != np.dtype("int64"):
        raise ValueError("Witness must contain signed-int64 integers")
    if stored.shape == (n*(n+1)//2,):
        witness_format = "packed lower triangle in row order"
        factor = np.zeros((n, n), dtype=np.int64)
        cursor = 0
        for i in range(n):
            factor[i, :i+1] = stored[cursor:cursor+i+1]
            cursor += i+1
    elif stored.shape == (n, n):
        witness_format = "dense lower-triangular array"
        factor = stored
        if np.count_nonzero(np.triu(factor, 1)):
            raise ValueError("Dense witness has nonzero entries above its diagonal")
    else:
        raise ValueError("Expected packed or dense lower-triangular factor of order 4192")
    del stored
    gram = ExactGram(factor, engine)
    bits = 46
    factor_square_denominator = 1 << (2*bits)
    denominator = matrix.denominator*factor_square_denominator
    multiplicities = matrix.multiplicities.astype(object)
    slacks, diagonals, off_sums = [], [], []
    last_progress = time.perf_counter()
    for lo in range(0, n, block_size):
        hi = min(n, lo+block_size)
        h = matrix.block(np.arange(lo, hi))
        g = gram.block(lo, hi)
        residual = h*factor_square_denominator-g*matrix.denominator
        weighted = (np.abs(residual)*multiplicities[None, :]).sum(axis=1)
        for a, r in enumerate(range(lo, hi)):
            m = int(multiplicities[r])
            i = int(matrix.representatives[r])
            diagonal = 11*denominator-int(g[a, r])*matrix.denominator
            within = (matrix.raw_diagonal[i]*matrix.diagonal_factor*factor_square_denominator
                      - int(g[a, r])*matrix.denominator)
            cross = int(weighted[a])-m*abs(int(residual[a, r]))
            off_sum = (m-1)*abs(within)+cross
            slack = diagonal-off_sum
            if slack*(1 << 31) <= denominator:
                raise ArithmeticError(
                    "DIRECT_PHYSICAL_ROW_NOT_GREATER_THAN_2^-31: "
                    f"class={r}, representative={i}, slack={Fraction(slack, denominator)}")
            slacks.append(slack)
            diagonals.append(diagonal)
            off_sums.append(off_sum)
        now = time.perf_counter()
        if now-last_progress >= 15 or hi == n:
            print(f"INDEPENDENT_DIRECT_ROWS {hi}/{n} seconds={now-start:.3f}", flush=True)
            last_progress = now
    if len(slacks) != n:
        raise ArithmeticError("INCOMPLETE_CLASS_VERIFICATION")
    physical_slacks = [None]*8192
    for rows, slack in zip(matrix.classes, slacks):
        for i in rows:
            if physical_slacks[i] is not None:
                raise ArithmeticError("DUPLICATE_PHYSICAL_ROW")
            physical_slacks[i] = slack
    for i in matrix.empty:
        if physical_slacks[i] is not None:
            raise ArithmeticError("DUPLICATE_EMPTY_ROW")
        physical_slacks[i] = 11*denominator
    if any(value is None for value in physical_slacks):
        raise ArithmeticError("INCOMPLETE_PHYSICAL_ROW_COVERAGE")
    minimum = min(range(8192), key=lambda i: physical_slacks[i])
    return {
        "status": "VERIFIED_DIRECT_GRAM_PLUS_SDD", "complete": True,
        "target_exact": "211571753466829574204178595602447/50000000000000000000000000000000",
        "physical_order": 8192, "factor_shape": [8192, n],
        "factor_denominator_exponent": bits, "witness_format": witness_format,
        "witness_sha256": sha(witness), "partition_sha256": sha(partition_path),
        "verifier_sha256": sha(Path(__file__)),
        "compressed_nonempty_row_checks": n,
        "zero_factor_rows_with_residual_diagonal_11": len(matrix.empty),
        "all_physical_row_checks": 8192,
        "row_slack_numerators": list(map(str, physical_slacks)),
        "row_slack_common_denominator": str(denominator),
        "class_slack_numerators": list(map(str, slacks)),
        "class_diagonal_numerators": list(map(str, diagonals)),
        "class_absolute_offdiagonal_sum_numerators": list(map(str, off_sums)),
        "minimum_slack_physical_index": minimum,
        "minimum_slack_exact": str(Fraction(physical_slacks[minimum], denominator)),
        "strict_row_slack_lower_bound": "1/2147483648",
        "certified_P_lower_bound": "1/2147483648",
        "gram_arithmetic": gram.metadata(),
        "runtime_seconds": time.perf_counter()-start,
        "peak_rss_bytes": peak_rss_bytes(),
        "platform": platform.platform(), "python": platform.python_version(),
        "numpy": np.__version__,
        "independence": "Separate direct physical-row formula; shared packaged matrix reconstruction and exact Gram primitives; no primary verifier or generator imported",
        "positivity_of_original_components_used": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Unpacked certificate package root")
    parser.add_argument("--witness", type=Path, help="Packed or dense witness; relative paths are resolved under --root (default: witness/L_lower.npy)")
    parser.add_argument("--output", type=Path, help="Receipt path (default: ROOT/results/independent_verification.json)")
    parser.add_argument("--engine", choices=("blas", "int64"), default="blas")
    parser.add_argument("--block-size", type=int, default=64)
    args = parser.parse_args()
    output = args.output or args.root / "results/independent_verification.json"
    start = time.perf_counter()
    try:
        result = verify(args.root, args.witness, args.engine, args.block_size)
        code = 0
    except Exception as error:
        result = {"status": "REJECTED", "complete": False,
                  "error_type": type(error).__name__, "error": str(error),
                  "runtime_seconds": time.perf_counter()-start,
                  "peak_rss_bytes": peak_rss_bytes(), "traceback": traceback.format_exc(),
                  "psd_claim": "No PSD claim is made by this failed run"}
        code = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")
    concise = {k: v for k, v in result.items()
               if k not in {"row_slack_numerators", "class_slack_numerators",
                            "class_diagonal_numerators", "class_absolute_offdiagonal_sum_numerators", "traceback"}}
    print(json.dumps(concise, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
