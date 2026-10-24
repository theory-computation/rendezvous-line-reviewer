# T=11, k=2 lower bound

The certificate proves, for initial distance 2,

    R >= 211571753466829574204178595602447/50000000000000000000000000000000

It verifies `Q_{11,2} - f B_{11,2} = P + N`, where `f` is the fraction above, `P` is positive semidefinite, and `N` is entrywise nonnegative. The matrices have order 8,192.

## Certificate files

- `certificates/FPLUS/` and `inputs/parent/`: exact rational coefficients, integer arrays, and reconstruction data.
- `witness/L_lower.npy`: packed integer PSD witness. `witness/partition.json` specifies its physical rows; `witness/format.json` describes the encoding.
- `src/` and `vendor/`: reconstruction and verification programs, including independent PSD and physical checks.
- `SHA256SUMS.txt` and `verify_hashes.py`: file-integrity check.

All supplied mathematical data files are required by the verification programs or describe the witness representation.

## Check the certificate

From this directory, in Windows CMD:

```cmd
py -3 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
call RUN_VERIFY.cmd
```

The command checks file hashes, verifies the PSD witness exactly, and runs the full-domain checks with both chronological and displacement physical oracles. A successful run exits with code 0 and prints `ALL_EXACT_VERIFICATIONS_PASS`; its output files are written to `results/`. Do not use Python's `-O` option, which disables assertions.

The PSD check constructs `E = P - 2^(-92) Z Z^T` and verifies `E_ii > sum_{j != i} |E_ij| + 2^(-31)` for all 8,192 rows. Together with the full-domain checks, this establishes the stated decomposition and lower bound.

An additional independent PSD checker is available:

```cmd
python src\verify_direct_independent.py --root . --engine int64
```

Success reports `VERIFIED_DIRECT_GRAM_PLUS_SDD` with `complete: true`. This checks the PSD witness; the full-domain checks in `RUN_VERIFY.cmd` are also needed for the lower bound.
