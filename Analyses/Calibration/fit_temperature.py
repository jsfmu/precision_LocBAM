"""Step 2: patient-level calibration/test split of the cached held-out cases, then fit one temperature T.

T minimizes voxel NLL of softmax(logits / T) on the CALIBRATION patients only.
Voxels are subsampled uniformly per case (a full volume is ~12M voxels; all of them would not fit in RAM).
Writes <cache>/split.json and <cache>/temperature.json. Test patients are not touched here.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True, help="data/cache/<hash> folder from cache_logits.py")
ap.add_argument("--cal-frac", type=float, default=0.5)
ap.add_argument("--voxels-per-case", type=int, default=200_000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
cache = Path(a.cache)

# 1. patient-level split (fixed seed, saved so step 3 evaluates exactly these test patients)
cases = sorted(p.stem for p in cache.glob("case_*.npz"))
shuffled = cases[:]
random.Random(a.seed).shuffle(shuffled)
n_cal = round(len(cases) * a.cal_frac)
split = {"seed": a.seed, "calibration": sorted(shuffled[:n_cal]), "test": sorted(shuffled[n_cal:])}
assert not set(split["calibration"]) & set(split["test"])
json.dump(split, open(cache / "split.json", "w"), indent=1)
print(f"{len(cases)} cases -> calibration {len(split['calibration'])}, test {len(split['test'])}")

# 2. gather a uniform voxel sample of logits + labels from the calibration patients
rng = np.random.default_rng(a.seed)
Z, Y = [], []
for cid in split["calibration"]:
    z = np.load(cache / f"{cid}.npz")
    logits = z["logits"].reshape(z["logits"].shape[0], -1)  # (C, N) fp16
    label = z["label"].reshape(-1)
    idx = rng.choice(label.size, size=min(a.voxels_per_case, label.size), replace=False)
    Z.append(logits[:, idx].T.astype(np.float32))  # (n, C)
    Y.append(label[idx].astype(np.int64))
Z, Y = np.concatenate(Z), np.concatenate(Y)
print(f"fitting on {len(Y):,} voxels, class counts {np.bincount(Y, minlength=Z.shape[1]).tolist()}")

# 3. NLL as a function of T; optimize over log T so T stays positive
def nll(T):
    s = Z / T
    return float(np.mean(logsumexp(s, axis=1) - s[np.arange(len(Y)), Y]))

res = minimize_scalar(lambda lt: nll(np.exp(lt)), bounds=(np.log(0.05), np.log(20)), method="bounded")
T = float(np.exp(res.x))
out = {"T": T, "nll_T1": nll(1.0), "nll_T": nll(T), "n_voxels": int(len(Y)),
       "voxels_per_case": a.voxels_per_case, "seed": a.seed, "split": "split.json"}
json.dump(out, open(cache / "temperature.json", "w"), indent=1)
print(f"T = {T:.4f}   calibration NLL: T=1 {out['nll_T1']:.5f} -> T* {out['nll_T']:.5f}")
print("wrote", cache / "split.json", "and", cache / "temperature.json")
