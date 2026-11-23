"""Step 2: patient-level calibration/test split of the cached held-out cases, then fit a post-hoc calibrator.

All fitting uses the CALIBRATION patients only; test patients are never read here.
Voxels are subsampled per case (a full volume is ~12M voxels; all of them would not fit in RAM).

--region all  sample uniformly from the whole volume (background dominates: >99% of voxels)
--region fg   sample only from the organ region: voxels whose label OR raw argmax is not background.
              This is the same region as nll_fg / ece_*_fg in evaluate_calibration.py.

--method temperature  one scalar T: p = softmax(z / T). Cannot change the argmax.
--method vector       vector scaling (Guo et al. 2017): p = softmax(w * z + b), one w_k and b_k per class
                      (b_0 fixed to 0, softmax is shift-invariant). CAN change the argmax, so Dice must be re-reported.

Writes <cache>/split.json (only if missing; otherwise checks it is identical) and the calibrator JSON:
temperature.json for the default (--region all --method temperature), else calib_<region>_<method>.json.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True, help="data/cache/<hash> folder from cache_logits.py")
ap.add_argument("--cal-frac", type=float, default=0.5)
ap.add_argument("--voxels-per-case", type=int, default=200_000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--region", choices=["all", "fg"], default="all")
ap.add_argument("--method", choices=["temperature", "vector"], default="temperature")
ap.add_argument("--out", default=None, help="output file name inside --cache")
a = ap.parse_args()
cache = Path(a.cache)
out_name = a.out or ("temperature.json" if (a.region, a.method) == ("all", "temperature")
                     else f"calib_{a.region}_{a.method}.json")

# 1. patient-level split (fixed seed). Reuse an existing split.json so every calibrator is fitted
#    and evaluated on exactly the same patients; refuse to continue if it would change.
cases = sorted(p.stem for p in cache.glob("case_*.npz"))
shuffled = cases[:]
random.Random(a.seed).shuffle(shuffled)
n_cal = round(len(cases) * a.cal_frac)
split = {"seed": a.seed, "calibration": sorted(shuffled[:n_cal]), "test": sorted(shuffled[n_cal:])}
assert not set(split["calibration"]) & set(split["test"])
split_path = cache / "split.json"
if split_path.exists():
    old = json.load(open(split_path))
    assert old == split, f"{split_path} differs from the split these arguments produce; refusing to overwrite it"
else:
    json.dump(split, open(split_path, "w"), indent=1)
print(f"{len(cases)} cases -> calibration {len(split['calibration'])}, test {len(split['test'])}")

# 2. gather a voxel sample of logits + labels from the calibration patients
rng = np.random.default_rng(a.seed)
Z, Y = [], []
for cid in split["calibration"]:
    z = np.load(cache / f"{cid}.npz")
    logits = z["logits"].reshape(z["logits"].shape[0], -1)  # (C, N) fp16
    label = z["label"].reshape(-1)
    if a.region == "fg":
        pool = np.flatnonzero((label != 0) | (logits.argmax(axis=0) != 0))
    else:
        pool = np.arange(label.size)
    idx = rng.choice(pool, size=min(a.voxels_per_case, pool.size), replace=False)
    Z.append(logits[:, idx].T.astype(np.float32))  # (n, C)
    Y.append(label[idx].astype(np.int64))
Z, Y = np.concatenate(Z), np.concatenate(Y)
n, K = Z.shape
print(f"region={a.region} method={a.method}: fitting on {n:,} voxels, class counts {np.bincount(Y, minlength=K).tolist()}")


def nll_scores(s):
    return float(np.mean(logsumexp(s, axis=1) - s[np.arange(n), Y]))


out = {"method": a.method, "region": a.region, "n_voxels": int(n), "voxels_per_case": a.voxels_per_case,
       "seed": a.seed, "split": "split.json", "fitted_on": "calibration",
       "class_counts": np.bincount(Y, minlength=K).tolist(), "nll_T1": nll_scores(Z)}

# 3. fit
if a.method == "temperature":
    # optimize over log T so T stays positive
    res = minimize_scalar(lambda lt: nll_scores(Z / np.exp(lt)), bounds=(np.log(0.05), np.log(20)), method="bounded")
    T = float(np.exp(res.x))
    out.update({"T": T, "nll_fit": nll_scores(Z / T)})
    print(f"T = {T:.4f}   calibration NLL ({a.region}): T=1 {out['nll_T1']:.5f} -> fitted {out['nll_fit']:.5f}")
else:
    onehot = np.zeros((n, K), dtype=np.float32)
    onehot[np.arange(n), Y] = 1.0

    def f_and_grad(theta):
        w, b = theta[:K], np.concatenate([[0.0], theta[K:]])
        s = Z * w.astype(np.float32) + b.astype(np.float32)
        lse = logsumexp(s, axis=1)
        f = float(np.mean(lse - s[np.arange(n), Y]))
        G = (np.exp(s - lse[:, None]) - onehot) / n  # d f / d s
        gw = (G * Z).sum(axis=0, dtype=np.float64)
        gb = G.sum(axis=0, dtype=np.float64)[1:]
        return f, np.concatenate([gw, gb])

    theta0 = np.concatenate([np.ones(K), np.zeros(K - 1)])
    res = minimize(f_and_grad, theta0, jac=True, method="L-BFGS-B",
                   bounds=[(1e-3, 100.0)] * K + [(None, None)] * (K - 1))
    assert res.success, res.message
    w, b = res.x[:K], np.concatenate([[0.0], res.x[K:]])
    out.update({"w": w.tolist(), "b": b.tolist(), "nll_fit": float(res.fun), "iterations": int(res.nit)})
    print("w =", np.round(w, 4).tolist(), "  b =", np.round(b, 4).tolist())
    print(f"calibration NLL ({a.region}): identity {out['nll_T1']:.5f} -> fitted {out['nll_fit']:.5f}")

json.dump(out, open(cache / out_name, "w"), indent=1)
print("wrote", cache / out_name)
