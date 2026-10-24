# Symmetric rendezvous on the line

Exact computational certificates for two indistinguishable unit-speed agents initially separated by distance 2.

| Result | Certificate and reproduction instructions |
| --- | --- |
| S7026 upper bound | [upper-bound/S7026](upper-bound/S7026/README.md) |
| S1964 upper bound | [upper-bound/S1964](upper-bound/S1964/README.md) |
| T=11, k=2 lower bound | [lower-bound](lower-bound/README.md) |

Each directory states the exact bound, identifies its witness, and gives Windows CMD commands and the expected result. Upper-bound checks require Python 3 and NumPy; the independent C++ checks also require a C++17 GCC compiler supporting `__int128`. Lower-bound checks require Python 3 and NumPy. Full reproduction performs substantial exact computations.
