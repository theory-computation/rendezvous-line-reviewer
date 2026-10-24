"""Transport integrity only; the PSD theorem is checked separately."""
from pathlib import Path
import hashlib

root = Path(__file__).resolve().parent
count = 0
for line in (root / "SHA256SUMS.txt").read_text().splitlines():
    expected, name = line.split("  ", 1)
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit("HASH_MISMATCH: " + name)
    count += 1
print("PACKAGE_HASHES_PASS", count)
