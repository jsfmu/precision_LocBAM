"""Step 2b: choose the two-region calibrator (method x dilation) by cross-validation on CALIBRATION patients only.

The test patients are never read. Candidates are calibrators applied only inside the predicted region
(raw argmax != background, dilated by d voxels), identity (T = 1) everywhere else:
  temperature  one T for the whole region
  predclass    one T per predicted class (argmax-preserving)
  isotonic     one monotone curve per (predicted class, class), renormalized (isotonic_map.py; CAN change argmax)

Criterion: change in mean whole-volume NLL on held-out calibration patients (patient-level K-fold).
NLL is a proper scoring rule; using the whole volume means a larger region is only rewarded if the
voxels it adds actually get better, and outside the region nothing changes, so only region voxels
contribute to the change:  delta = sum_p |R_d,p| * mean_{v in R_d,p}(nll_cal - nll_raw) / sum_p N_p.

To fit in memory each patient keeps one uniform sample of its largest region (with each voxel's
dilation level), which is also a uniform sample of every smaller region; the mean above is estimated
from it and scaled by the exact region size.

Writes <cache>/calibrator_selection.json and prints the fit_temperature.py command for the winner.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

import isotonic_map as iso

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--dilations", type=int, nargs="+", default=[0, 1, 2, 4, 8])
ap.add_argument("--methods", nargs="+", default=["temperature", "predclass"], choices=["temperature", "predclass", "isotonic"])
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--sample-per-case", type=int, default=400_000, help="voxels kept per patient from the largest region")
ap.add_argument("--fit-per-case", type=int, default=100_000, help="voxels per training patient used in each fit")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="calibrator_selection.json", help="output file name inside --cache")
a = ap.parse_args()
cache = Path(a.cache)
D = sorted(a.dilations)
split = json.load(open(cache / "split.json"))
patients = split["calibration"]
rng = np.random.default_rng(a.seed)
st = generate_binary_structure(3, 1)

# ---------------------------------------------------------------- one pass over calibration patients
data = {}
t0 = time.time()
for i, cid in enumerate(patients):
    z = np.load(cache / f"{cid}.npz")
    L, Y = z["logits"], z["label"]
    K = L.shape[0]
    raw_fg = L.argmax(axis=0) != 0
    level = np.full(Y.shape, 255, dtype=np.uint8)
    region, prev = raw_fg, 0
    sizes = []
    for j, d in enumerate(D):
        if d > prev:
            region = binary_dilation(region, structure=st, iterations=d - prev)
            prev = d
        level[region & (level == 255)] = j
        sizes.append(int(region.sum()))
    idx = np.flatnonzero(level.reshape(-1) != 255)
    if idx.size > a.sample_per_case:
        idx = np.sort(rng.choice(idx, size=a.sample_per_case, replace=False))
    Lf = L.reshape(K, -1)
    data[cid] = {"z": Lf[:, idx].T.copy(), "y": Y.reshape(-1)[idx].astype(np.int64),
                 "lvl": level.reshape(-1)[idx], "sizes": sizes, "N": int(Y.size)}
    print(f"[{i + 1}/{len(patients)}] {cid}  region sizes {sizes}  {time.time() - t0:.0f}s", flush=True)


def nll_vec(Z, Yv, T):
    """per-voxel NLL of softmax(Z / T); T scalar or (n,) array."""
    s = Z / (T if np.isscalar(T) else T[:, None])
    return logsumexp(s, axis=1) - s[np.arange(len(Yv)), Yv]


def fit_T(Z, Yv):
    if len(Yv) == 0:
        return 1.0
    res = minimize_scalar(lambda lt: float(np.mean(nll_vec(Z, Yv, float(np.exp(lt))))),
                          bounds=(np.log(0.05), np.log(20)), method="bounded")
    return float(np.exp(res.x))


def fit(method, Z, Yv):
    if method == "isotonic":
        return iso.fit(iso.add_stats(iso.empty_stats(Z.shape[1]), iso.softmax(Z.T), Yv))
    if method == "temperature":
        return {"T": fit_T(Z, Yv)}
    R = Z.argmax(axis=1)
    return {"T_by_pred_class": [fit_T(Z[R == k], Yv[R == k]) if (R == k).sum() >= 1000 else 1.0
                                for k in range(Z.shape[1])]}


def method_nll(method, params, Z, Yv):
    """per-voxel NLL of the calibrated probabilities."""
    if method == "isotonic":
        S = iso.calibrated_scores(Z.T, params)  # (K, n) log-probabilities
        return -S[Yv, np.arange(len(Yv))]
    return nll_vec(Z, Yv, T_of(method, params, Z))


def summary(method, params):
    """what is kept in the JSON per fold (isotonic curves are too large to store 5 x dilations times)."""
    if method == "isotonic":
        return {"pred_class_counts": params["pred_class_counts"]}
    return params


def T_of(method, params, Z):
    if method == "temperature":
        return params["T"]
    return np.asarray(params["T_by_pred_class"], dtype=np.float32)[Z.argmax(axis=1)]


# ---------------------------------------------------------------- K-fold CV over patients
order = patients[:]
random.Random(a.seed).shuffle(order)
folds = [sorted(order[f::a.folds]) for f in range(a.folds)]
results = {m: {d: {"delta_nll_whole_volume": [], "delta_nll_region": [], "params": []} for d in D} for m in a.methods}
for f, held in enumerate(folds):
    train = [p for p in patients if p not in held]
    for j, d in enumerate(D):
        # training sample restricted to region d, capped per patient
        Zs, Ys = [], []
        for p in train:
            m = np.flatnonzero(data[p]["lvl"] <= j)
            if m.size > a.fit_per_case:
                m = rng.choice(m, size=a.fit_per_case, replace=False)
            Zs.append(data[p]["z"][m].astype(np.float32)); Ys.append(data[p]["y"][m])
        Ztr, Ytr = np.concatenate(Zs), np.concatenate(Ys)
        for meth in a.methods:
            params = fit(meth, Ztr, Ytr)
            num, n_all, num_r, n_r = 0.0, 0, 0.0, 0
            for p in held:
                m = data[p]["lvl"] <= j
                Zh, Yh = data[p]["z"][m].astype(np.float32), data[p]["y"][m]
                if len(Yh):
                    diff = float(np.mean(method_nll(meth, params, Zh, Yh) - nll_vec(Zh, Yh, 1.0)))
                    num += diff * data[p]["sizes"][j]
                    num_r += diff * data[p]["sizes"][j]
                    n_r += data[p]["sizes"][j]
                n_all += data[p]["N"]
            results[meth][d]["delta_nll_whole_volume"].append(num / n_all)
            results[meth][d]["delta_nll_region"].append(num_r / n_r if n_r else None)
            results[meth][d]["params"].append(summary(meth, params))
    print(f"fold {f + 1}/{a.folds} done  {time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- choose
print("\nmean change in whole-volume NLL on held-out calibration patients (more negative = better)")
print(f"{'method':<12}{'dilate':>7}{'delta NLL (whole)':>20}{'fold sd':>10}{'delta NLL (region)':>20}   fitted T (fold 1)")
best = None
for meth in a.methods:
    for d in D:
        r = results[meth][d]
        mu, sd = float(np.mean(r["delta_nll_whole_volume"])), float(np.std(r["delta_nll_whole_volume"]))
        mr = float(np.mean([x for x in r["delta_nll_region"] if x is not None]))
        r.update({"mean": mu, "sd": sd, "mean_region": mr})
        ps = r["params"][0]
        pstr = (f"{ps['T']:.3f}" if meth == "temperature" else
                "voxels by pred class " + ",".join(str(c) for c in ps["pred_class_counts"]) if meth == "isotonic" else
                ",".join(f"{t:.2f}" for t in ps["T_by_pred_class"]))
        print(f"{meth:<12}{d:>7}{mu:>+20.6f}{sd:>10.6f}{mr:>+20.5f}   {pstr}")
        if best is None or mu < best[2]:
            best = (meth, d, mu)

meth, d, mu = best
cmd = f"python Analyses/Calibration/fit_temperature.py --cache {a.cache} --region pred --dilate {d} --method {meth}"
json.dump({"criterion": "mean change in whole-volume NLL on held-out calibration patients",
           "folds": folds, "dilations": D, "methods": a.methods, "seed": a.seed,
           "results": {m: {str(d): v for d, v in r.items()} for m, r in results.items()},
           "selected": {"method": meth, "dilate": d, "cv_delta_nll": mu, "fit_command": cmd}},
          open(cache / a.out, "w"), indent=1)
print(f"\nselected: method={meth} dilate={d} (CV delta {mu:+.6f})")
print("fit it on all calibration patients with:\n  " + cmd)
print("wrote", cache / a.out)
