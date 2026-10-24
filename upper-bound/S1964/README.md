# S1964 upper bound

The certificate proves

    R <= 2833204836704511365843123991651/666438807624376979383282747045

`strategy.tsv` contains 1964 canonical movements, representing 3928 oriented movements of lengths 16 through 28. Its columns are `canonical_word`, `length`, and `pair_multiplicity`. Bits 0 and 1 mean unit displacements -1 and +1. Each word and its reflection receive probability `pair_multiplicity/(2C)`, where `C = 999999999999995` is the sum of the multiplicities. The agents resample only after choosing the same oriented movement.

## Check the certificate

From this directory, in Windows CMD:

```cmd
python -m pip install -r requirements.txt
python prefix_verify.py strategy.tsv
```

A successful reproduction exits with code 0, reports `status: PASS`, and prints `exact_R` equal to the fraction above. It checks that every pair of distinct oriented movements meets within their common prefix and computes the exact renewal value `p^T M^S p / (1 - ||p||_2^2)`. The output must report `1964` for `support` and `15429184` for `ordered_orientation_tests`. Do not use Python's `-O` option, which disables assertions.

## Independent checks

The C++ programs independently compute the same value using literal velocity segments and ordered integer-position pairs. With a C++17 `g++` compiler supporting `__int128` in PATH:

```cmd
g++ -O2 -std=c++17 verify_segments.cpp -o verify_segments.exe
verify_segments.exe strategy.tsv
g++ -O2 -std=c++17 ordered_statistics.cpp -o ordered_statistics.exe
ordered_statistics.exe strategy.tsv
```

Both must exit with code 0 and print:

    numerator:   16999229020227068195058743949906
    denominator: 3998632845746261876299696482270

Their ratio is the bound above. `ordered_statistics.cpp` also returns the exact first-epoch meeting and failure masses, with their common denominator. `MANIFEST.sha256` contains integrity hashes for the supplied files.
