"""Step 2: patient-level calibration/test split of the cached held-out cases, then fit a post-hoc calibrator.

All fitting uses the CALIBRATION patients only; test patients are never read here.
Voxels are subsampled per case (a full volume is ~12M voxels; all of them would not fit in RAM).

--region all  sample uniformly from the whole volume (background dominates: >99% of voxels)
--region fg   sample only from the organ region: voxels whose label OR raw argmax is not background.
              This is the same region as nll_fg / ece_*_fg in evaluate_calibration.py. It uses the ground
              truth, so a calibrator fitted here cannot be restricted to this region at inference time.
--region pred sample only from the PREDICTED region: raw argmax != background, binary-dilated by --dilate
              voxels. Needs no labels, so evaluate_calibration.py applies such a calibrator only inside
              this region and leaves every other voxel at T = 1 (a two-region calibrator usable at inference).

--method temperature  one scalar T: p = softmax(z / T). Cannot change the argmax.
--method classwise    one temperature per class, no bias: p = softmax(z_k / T_k). CAN change the argmax.
--method predclass    one temperature per PREDICTED class: a voxel whose raw argmax is k gets p = softmax(z / T_k).
                      Every voxel is divided by one positive scalar, so the argmax CANNOT change. The T_k are
                      fitted independently (NLL separates by group). Only with --region pred.
--method vector       vector scaling (Guo et al. 2017): p = softmax(w * z + b), one w_k and b_k per class
                      (b_0 fixed to 0, softmax is shift-invariant). CAN change the argmax, so Dice must be re-reported.

Writes <cache>/split.json (only if missing; otherwise checks it is identical) and the calibrator JSON:
temperature.json for the default (--region all --method temperature), else calib_<region>_<method>.json
(region "pred" is written as pred<dilate>, e.g. calib_pred2_classwise.json).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp


def pred_region(logits4d, dilate):
    """(C, X, Y, Z) raw logits -> bool (X, Y, Z): raw argmax != 0, dilated by `dilate` voxels (6-connected)."""
    m = logits4d.argmax(axis=0) != 0
    if dilate > 0:
        m = binary_dilation(m, structure=generate_binary_structure(3, 1), iterations=dilate)
    return m

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True, help="data/cache/<hash> folder from cache_logits.py")
ap.add_argument("--cal-frac", type=float, default=0.5)
ap.add_argument("--voxels-per-case", type=int, default=200_000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--region", choices=["all", "fg", "pred"], default="all")
ap.add_argument("--dilate", type=int, default=2, help="--region pred: dilation of the predicted mask, in voxels")
ap.add_argument("--method", choices=["temperature", "classwise", "predclass", "vector"], default="temperature")
ap.add_argument("--out", default=None, help="output file name inside --cache")
a = ap.parse_args()
assert a.method != "predclass" or a.region == "pred", "--method predclass needs --region pred"
cache = Path(a.cache)
region_tag = f"pred{a.dilate}" if a.region == "pred" else a.region
out_name = a.out or ("temperature.json" if (a.region, a.method) == ("all", "temperature")
                     else f"calib_{region_tag}_{a.method}.json")

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
    logits4d = z["logits"]
    logits = logits4d.reshape(logits4d.shape[0], -1)  # (C, N) fp16
    label = z["label"].reshape(-1)
    if a.region == "fg":
        pool = np.flatnonzero((label != 0) | (logits.argmax(axis=0) != 0))
    elif a.region == "pred":
        pool = np.flatnonzero(pred_region(logits4d, a.dilate).reshape(-1))
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


out = {"method": a.method, "region": a.region, "dilate": a.dilate if a.region == "pred" else None,
       "applied_to": "pred region only, T=1 elsewhere" if a.region == "pred" else "whole volume",
       "n_voxels": int(n), "voxels_per_case": a.voxels_per_case,
       "seed": a.seed, "split": "split.json", "fitted_on": "calibration",
       "class_counts": np.bincount(Y, minlength=K).tolist(), "nll_T1": nll_scores(Z)}

# 3. fit
if a.method == "temperature":
    # optimize over log T so T stays positive
    res = minimize_scalar(lambda lt: nll_scores(Z / np.exp(lt)), bounds=(np.log(0.05), np.log(20)), method="bounded")
    T = float(np.exp(res.x))
    out.update({"T": T, "nll_fit": nll_scores(Z / T)})
    print(f"T = {T:.4f}   calibration NLL ({a.region}): T=1 {out['nll_T1']:.5f} -> fitted {out['nll_fit']:.5f}")
elif a.method == "predclass":
    R = Z.argmax(axis=1)
    Ts, counts, total = [], [], 0.0
    for k in range(K):
        m = R == k
        counts.append(int(m.sum()))
        if m.sum() < 1000:  # too few voxels predicted as k to fit; leave that group uncalibrated
            Ts.append(1.0)
            continue
        Zk, Yk = Z[m], Y[m]
        f = lambda lt: float(np.mean(logsumexp(Zk / np.exp(lt), axis=1) - Zk[np.arange(len(Yk)), Yk] / np.exp(lt)))
        res = minimize_scalar(f, bounds=(np.log(0.05), np.log(20)), method="bounded")
        Ts.append(float(np.exp(res.x)))
    T_vox = np.asarray(Ts, dtype=np.float32)[R][:, None]
    out.update({"T_by_pred_class": Ts, "pred_class_counts": counts, "nll_fit": nll_scores(Z / T_vox)})
    print("T by predicted class =", {k: round(t, 4) for k, t in enumerate(Ts)}, " voxels:", counts)
    print(f"calibration NLL ({region_tag}): identity {out['nll_T1']:.5f} -> fitted {out['nll_fit']:.5f}")
else:
    onehot = np.zeros((n, K), dtype=np.float32)
    onehot[np.arange(n), Y] = 1.0

    # scores s = w * z + b; classwise fixes b = 0 and reports T_k = 1 / w_k
    use_bias = a.method == "vector"

    def f_and_grad(theta):
        w = theta[:K]
        b = np.concatenate([[0.0], theta[K:]]) if use_bias else np.zeros(K)
        s = Z * w.astype(np.float32) + b.astype(np.float32)
        lse = logsumexp(s, axis=1)
        f = float(np.mean(lse - s[np.arange(n), Y]))
        G = (np.exp(s - lse[:, None]) - onehot) / n  # d f / d s
        gw = (G * Z).sum(axis=0, dtype=np.float64)
        if not use_bias:
            return f, gw
        return f, np.concatenate([gw, G.sum(axis=0, dtype=np.float64)[1:]])

    theta0 = np.concatenate([np.ones(K), np.zeros(K - 1)]) if use_bias else np.ones(K)
    bounds = [(1e-3, 100.0)] * K + ([(None, None)] * (K - 1) if use_bias else [])
    res = minimize(f_and_grad, theta0, jac=True, method="L-BFGS-B", bounds=bounds)
    assert res.success, res.message
    w = res.x[:K]
    out.update({"nll_fit": float(res.fun), "iterations": int(res.nit)})
    if use_bias:
        b = np.concatenate([[0.0], res.x[K:]])
        out.update({"w": w.tolist(), "b": b.tolist()})
        print("w =", np.round(w, 4).tolist(), "  b =", np.round(b, 4).tolist())
    else:
        out["T_per_class"] = (1.0 / w).tolist()
        print("T per class =", {k: round(float(1 / wk), 4) for k, wk in enumerate(w)})
    print(f"calibration NLL ({region_tag}): identity {out['nll_T1']:.5f} -> fitted {out['nll_fit']:.5f}")

json.dump(out, open(cache / out_name, "w"), indent=1)
print("wrote", cache / out_name)
