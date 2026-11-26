"""Per-predicted-class isotonic calibration, shared by select_calibrator.py, fit_temperature.py and
evaluate_calibration.py so all three apply exactly the same rule.

For a voxel whose RAW argmax is r, each class probability p_k (raw softmax) is mapped through its own
monotone non-decreasing curve g_{r,k}, fitted by isotonic regression of 1[y == k] on p_k over the
calibration voxels predicted as r. The mapped values are floored at EPS (so NLL stays finite) and
renormalized to sum to 1:

    q_k = max(g_{r,k}(p_k), EPS) / sum_j max(g_{r,j}(p_j), EPS)

Unlike a temperature, g can bend: it can be over- and underconfident in different ranges of p.
The renormalization means the argmax CAN change, so Dice must be re-reported.

Fitting works on sufficient statistics: per (r, k), the voxel count, the sum of p_k and the number of
voxels with y == k in NBINS equal-width bins of p_k. Isotonic regression (pool-adjacent-violators,
weighted by count) runs on the bin means; the curve is linear interpolation between the bin means,
constant beyond the first and last one. Groups with fewer than MIN_COUNT voxels keep the identity.
"""
import numpy as np

NBINS = 1000
EPS = 1e-4
MIN_COUNT = 1000


def softmax(z, axis=0):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def empty_stats(K, nbins=NBINS):
    return {k: np.zeros((K, K, nbins)) for k in ("cnt", "sp", "pos")}


def add_stats(st, P, Y, R=None):
    """P (K, n) raw probabilities, Y (n,) labels, R (n,) raw argmax (default: argmax of P)."""
    K, nb = P.shape[0], st["cnt"].shape[2]
    R = P.argmax(axis=0) if R is None else R
    for r in range(K):
        m = R == r
        if not m.any():
            continue
        Pr, Yr = P[:, m], Y[m]
        for k in range(K):
            b = np.minimum((Pr[k] * nb).astype(np.int64), nb - 1)
            st["cnt"][r, k] += np.bincount(b, minlength=nb)
            st["sp"][r, k] += np.bincount(b, weights=Pr[k].astype(np.float64), minlength=nb)
            st["pos"][r, k] += np.bincount(b, weights=(Yr == k).astype(np.float64), minlength=nb)
    return st


def pav(y, w):
    """Weighted isotonic (non-decreasing) regression by pool-adjacent-violators."""
    vals, wts, sizes = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            wsum = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wsum
            n = sizes[-2] + sizes[-1]
            vals[-2:], wts[-2:], sizes[-2:] = [v], [wsum], [n]
    return np.repeat(vals, sizes)


def fit(st, min_count=MIN_COUNT):
    """-> params dict (JSON-serializable): maps[r][k] = {"x": bin means of p, "y": fitted frequencies}."""
    K = st["cnt"].shape[0]
    maps, counts = [], []
    for r in range(K):
        n_r = int(st["cnt"][r, 0].sum())  # every voxel of group r appears once per k
        counts.append(n_r)
        row = []
        for k in range(K):
            c = st["cnt"][r, k]
            nz = c > 0
            if n_r < min_count or nz.sum() < 2:
                row.append({"x": [0.0, 1.0], "y": [0.0, 1.0], "identity": True})
                continue
            x = st["sp"][r, k][nz] / c[nz]
            y = pav(st["pos"][r, k][nz] / c[nz], c[nz])
            row.append({"x": x.tolist(), "y": y.tolist()})
        maps.append(row)
    return {"method": "isotonic", "nbins": int(st["cnt"].shape[2]), "eps": EPS, "min_count": min_count,
            "pred_class_counts": counts, "maps": maps}


def apply(P, params, R=None):
    """P (K, n) raw probabilities -> calibrated probabilities (K, n)."""
    K = P.shape[0]
    R = P.argmax(axis=0) if R is None else R
    Q = np.empty_like(P, dtype=np.float32)
    for r in range(K):
        m = R == r
        if not m.any():
            continue
        for k in range(K):
            g = params["maps"][r][k]
            Q[k, m] = np.interp(P[k, m], g["x"], g["y"])
    np.maximum(Q, params.get("eps", EPS), out=Q)
    Q /= Q.sum(axis=0, keepdims=True)
    return Q


def calibrated_scores(z, params):
    """z (K, n) raw logits -> log calibrated probabilities (their softmax is the calibrated probability)."""
    return np.log(apply(softmax(z.astype(np.float32)), params, R=z.argmax(axis=0)))
