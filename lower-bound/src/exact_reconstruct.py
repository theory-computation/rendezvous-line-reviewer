#!/usr/bin/env python3
"""Reconstruct locked P and its exact twin quotient, without PSD assumptions.

All acceptance arithmetic is integer/rational.  Native products use NumPy
signed-int64 arithmetic only after bounding every partial sum below 2**63.
"""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np

if not __debug__:
    raise RuntimeError("Exact verification requires Python without -O or PYTHONOPTIMIZE.")

F = Fraction
N = 8192
LOCK = "211571753466829574204178595602447/50000000000000000000000000000000"
CERT_HASH = "4b4fd86832ff3a6221f07732325240a3b064409c6dd79614d34993c29a71194e"
PARENT_JSON_HASHES = {
    "parent_rank_one_certificate.json": "f9cbbb377d901bf87eb42215fa6fb983ae40f96352f3b3c4b6cdb1e66678cac7",
    "candidate_best.json": "f29206f77834a13f6ec77d6c658e32c7438487455ae7ad606441c090063bb4d3",
    "certificate.json": "422d0d360af855ff357e321afd91fb5457c5a47cdd8afef70c953431b35088f3",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_matrix(name):
    """Definition only: no positivity claim about this matrix is used."""
    if name == "PETERSEN":
        n = 10
        edges = ([(i, (i+1) % 5) for i in range(5)]
                 + [(i, i+5) for i in range(5)]
                 + [(i+5, (i+2) % 5+5) for i in range(5)])
    else:
        assert name in {"C5", "C7", "C9", "C11", "C13"}
        n = int(name[1:])
        edges = [(i, (i+1) % n) for i in range(n)]
    H = np.ones((n, n), dtype=np.int64)
    for i, j in edges:
        H[i, j] = H[j, i] = -1
    return H


class LockedMatrix:
    def __init__(self, root, progress=False):
        self.started = time.time()
        root = Path(root)
        parent = root / "inputs" / "parent"
        data = root / "certificates" / "FPLUS"
        cert = json.loads((data / "certificate.json").read_text())
        assert sha(data / "certificate.json") == CERT_HASH
        for name, expected_hash in PARENT_JSON_HASHES.items():
            assert sha(parent / name) == expected_hash, ("CHANGED_LOCKED_SOURCE", name)
        pc = json.loads((parent / "parent_rank_one_certificate.json").read_text())
        base = json.loads((parent / "candidate_best.json").read_text())
        suffix = json.loads((parent / "certificate.json").read_text())
        assert cert["target"] == LOCK and (cert["T"], cert["k"]) == (11, 2)
        for name, value in pc["input_array_sha256"].items():
            assert sha(parent / name) == value
        get = lambda name: np.load(parent / (name + ".npy"), allow_pickle=False)
        U, W = get("gram_U_num"), get("gram_V_num")
        gu, gv = get("t11_group_u"), get("t11_group_v")
        assert U.shape == W.shape == (1888, 1888)
        assert U.dtype == W.dtype == np.dtype("int64")
        wu = np.asarray(cert["parent_u_weights"], dtype=np.int64)[gu]
        wv = np.asarray(cert["parent_v_weights"], dtype=np.int64)[gv]

        def native_gram(Z, weights):
            # Cauchy bounds each absolute partial sum by max_i sum_k Z_ik^2
            # times max_k |w_k|, with row norms themselves proved in range.
            bound = int(np.max(np.abs(Z)))
            assert Z.shape[1] * bound * bound < 2**63
            norms = np.sum(Z * Z, axis=1)
            assert int(np.max(norms)) * int(np.max(np.abs(weights))) < 2**63
            G = (Z * weights) @ Z.T
            assert np.array_equal(G, G.T)
            return G

        self.G1 = native_gram(U, wu)
        self.G2 = native_gram(W, wv)
        assert int(np.max(np.abs(self.G1))) + int(np.max(np.abs(self.G2))) < 2**63
        cf = np.asarray(pc["coefficient_numerators"], dtype=np.int64)
        assert int(np.max(np.sum(np.abs(U), axis=1))) * int(np.max(np.abs(cf))) < 2**63
        vv_native = U @ cf
        lookup = {}
        for k, (a, b) in enumerate(zip(get("t11_parent_reps"), get("t11_parent_refl"))):
            assert int(a) not in lookup and int(b) not in lookup
            lookup[int(a)], lookup[int(b)] = (k, 1), (k, -1)
        self.index = np.zeros(N, dtype=np.int64)
        self.sign = np.zeros(N, dtype=np.int64)
        for i in range(N):
            h = i // 2
            self.index[i], self.sign[i] = lookup.get(h//2 + 2048*(1-h % 2), (0, 0))
        self.v = vv_native[self.index] * (self.sign != 0)
        self.C = get("column_numerators")
        assert hashlib.sha256(self.C.astype("<i8").tobytes()).hexdigest() == suffix["column_numerators_sha256"]
        self.X = np.load(data / "global_column_numerators.npy", allow_pickle=False)
        assert sha(data / "global_column_numerators.npy") == cert["global_column_sha256"]
        assert self.C.shape == (N, 27) and self.X.shape == (N, 40)
        sigma = F(cert["psd_core_multiplier"])
        alpha = F(pc["downdate_coefficient"])
        gd, cd = int(pc["gram_denominator"]), int(pc["coefficient_denominator"])
        sd = int(base["suffix_eta_denominator"]) * int(suffix["column_denominator"])**2
        xd = int(cert["global_weight_denominator"]) * int(cert["global_column_denominator"])**2
        eta = list(map(int, base["suffix_eta_numerators"]))
        omega = list(map(int, cert["global_weight_numerators"]))
        self.graphs = []
        horn_columns = []
        for atom in cert["horn_factors"]:
            rows = atom["feature_groups"]
            H = graph_matrix(atom.get("motif", "C5"))
            assert len(rows) == len(H)
            groups = sorted(set(g for row in rows for g in row))
            assert all(type(g) is int and 0 <= g < 2048 for g in groups)
            E = np.array([[row.count(g) for g in groups] for row in rows], dtype=np.int64)
            raw = E.T @ H @ E
            weight = sigma * F(atom["integer_weight"], cert["horn_weight_denominator"])
            self.graphs.append((groups, raw, weight))
            for row in rows:
                col = np.zeros(2048, dtype=np.int64)
                for g in row:
                    col[g] += 1
                horn_columns.append(col)
        horn_features = np.stack(horn_columns, axis=1)
        edges = [(int(i), int(j), F(w)) for i, j, w in cert["edges"]]
        assert len(edges) == cert["edge_count"]
        coeffs = [sigma/gd, -sigma*alpha/cd**2, sigma/sd, sigma/xd]
        coeffs += [w for _, _, w in self.graphs] + [w for _, _, w in edges]
        self.base_denominator = math.lcm(*(q.denominator for q in coeffs))
        scale = self.base_denominator
        integer = lambda q: (scale*q).numerator if (scale*q).denominator == 1 else None
        self.kgram, self.kdown, ks, kx = map(integer, coeffs[:4])
        assert all(v is not None for v in (self.kgram, self.kdown, ks, kx))
        # Exact sparse corrections.  No decomposition of these corrections into
        # PSD terms is formed or used.
        corr = [defaultdict(int) for _ in range(N)]
        for Z, weights, mult in ((self.C, eta, ks), (self.X, omega, kx)):
            for c, w in enumerate(weights):
                rows = np.flatnonzero(Z[:, c])
                vals = Z[rows, c]
                assert int(np.max(np.abs(vals), initial=0))**2 < 2**63
                outer = vals[:, None] * vals[None, :]
                for a, i in enumerate(rows):
                    target = corr[int(i)]
                    for j, value in zip(rows, outer[a]):
                        target[int(j)] += mult*w*int(value)
        for groups, H, w in self.graphs:
            mult = integer(w)
            for a, g in enumerate(groups):
                for b, h in enumerate(groups):
                    value = mult*int(H[a, b])
                    if value:
                        for i in range(4*g, 4*g+4):
                            for j in range(4*h, 4*h+4):
                                corr[i][j] += value
        adjacency = [dict() for _ in range(N)]
        for i, j, w in edges:
            assert 0 <= i < j < N and j not in adjacency[i]
            adjacency[i][j] = adjacency[j][i] = w
            corr[i][j] -= integer(w)
            corr[j][i] -= integer(w)
        self.correction = [dict((j, z) for j, z in row.items() if z) for row in corr]
        self.raw_diagonal = []
        for i in range(N):
            ix, sg = int(self.index[i]), int(self.sign[i])
            g = int(self.G1[ix, ix]) + int(self.G2[ix, ix]) if sg else 0
            self.raw_diagonal.append(self.kgram*g + self.kdown*int(self.v[i])**2 + self.correction[i].get(i, 0))
        # Complete feature signatures certify equal rows of every defining
        # term before resetting the diagonal.  Exact adjacency equality also
        # certifies equal rows of the arbitrary sparse edge correction.
        signatures = defaultdict(list)
        for i in range(N):
            key = (int(self.index[i]), int(self.sign[i]), self.C[i].tobytes(),
                   self.X[i].tobytes(), horn_features[i//4].tobytes(),
                   tuple(sorted(adjacency[i].items())))
            signatures[key].append(i)
        self.classes, self.empty = [], []
        for key, rows in signatures.items():
            if key[1] == 0 and not any(key[2]) and not any(key[3]) and not any(key[4]) and not key[5]:
                self.empty.extend(rows)
            else:
                self.classes.append(rows)
        assert sorted(self.empty + [i for rows in self.classes for i in rows]) == list(range(N))
        self.representatives = np.array([rows[0] for rows in self.classes], dtype=np.int64)
        self.multiplicities = np.array([len(rows) for rows in self.classes], dtype=np.int64)
        self.order = len(self.classes)
        self.diagonal_factor = math.lcm(*map(int, self.multiplicities))
        self.denominator = scale * self.diagonal_factor
        self.min_contrast_remainder = None
        for rows in self.classes:
            i = rows[0]
            assert all(self.raw_diagonal[j] == self.raw_diagonal[i] for j in rows)
            if len(rows) > 1:
                d = 11*scale - self.raw_diagonal[i]
                # Diagnostic only. The direct certificate uses no contrast positivity.
                self.min_contrast_remainder = d if self.min_contrast_remainder is None else min(d, self.min_contrast_remainder)
        assert all(self.raw_diagonal[i] == 0 and not self.correction[i] for i in self.empty)
        if progress:
            print("EXACT_RECONSTRUCTION_READY", self.order, len(self.empty), round(time.time()-self.started, 3), flush=True)

    def block(self, rows, cols=None, reduced=True):
        """Return exact integer numerator, common denominator self.denominator."""
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.arange(self.order if reduced else N) if cols is None else np.asarray(cols, dtype=np.int64)
        dimension = self.order if reduced else N
        assert rows.ndim == cols.ndim == 1
        assert np.all((rows >= 0) & (rows < dimension)) and np.all((cols >= 0) & (cols < dimension))
        assert len(set(map(int, rows))) == len(rows) and len(set(map(int, cols))) == len(cols)
        ir = self.representatives[rows] if reduced else rows
        ic = self.representatives[cols] if reduced else cols
        ix, jx = self.index[ir], self.index[ic]
        sr, sc = self.sign[ir], self.sign[ic]
        native = self.G1[np.ix_(ix, jx)] + sr[:, None]*sc[None, :]*self.G2[np.ix_(ix, jx)]
        native *= (sr[:, None] != 0) & (sc[None, :] != 0)
        out = self.kgram * native.astype(object)
        out += self.kdown * self.v[ir].astype(object)[:, None] * self.v[ic].astype(object)[None, :]
        col_lookup = {int(j): p for p, j in enumerate(ic)}
        for a, i in enumerate(ir):
            for j, z in self.correction[int(i)].items():
                b = col_lookup.get(j)
                if b is not None:
                    out[a, b] += z
            b = col_lookup.get(int(i))
            if b is not None:
                if reduced:
                    m = int(self.multiplicities[int(rows[a])])
                    # Apply common factor after correction below to avoid any
                    # division until exact integrality is known.
                    out[a, b] = self.raw_diagonal[int(i)]
                else:
                    out[a, b] = 11*self.base_denominator
        out *= self.diagonal_factor
        if reduced:
            for a, i in enumerate(ir):
                b = col_lookup.get(int(i))
                if b is not None:
                    m = int(self.multiplicities[int(rows[a])])
                    assert self.diagonal_factor % m == 0
                    out[a, b] += (11*self.base_denominator-self.raw_diagonal[int(i)])*(self.diagonal_factor//m)
        return out

    def dyadic_floor(self, rows, cols=None, bits=100, reduced=True):
        return (self.block(rows, cols, reduced) * (1 << bits)) // self.denominator

    def summary(self):
        from collections import Counter
        return {"locked_order": N, "reduced_order": self.order,
                "empty_vertices": len(self.empty),
                "contrast_dimension": sum(len(c)-1 for c in self.classes),
                "class_size_counts": dict(Counter(map(len, self.classes))),
                "common_denominator": str(self.denominator),
                "minimum_exact_contrast_remainder": str(F(self.min_contrast_remainder, self.base_denominator)),
                "reconstruction_seconds": time.time()-self.started}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out")
    a = p.parse_args()
    M = LockedMatrix(a.root, progress=True)
    result = M.summary()
    print(json.dumps(result, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2)+"\n")
