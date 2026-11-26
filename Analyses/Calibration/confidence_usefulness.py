"""Step 5 (exploratory): is the confidence USEFUL, and does calibration make it more useful?

Two uses of a voxel probability, each evaluated for the raw model and every calibrator, on the TEST patients:

A. Flag-for-review (selective prediction), on the voxels chosen by --population: by default the organ region
   (label != 0 OR raw argmax != 0, the same region as evaluate_calibration.py; it uses labels, so it includes
   missed organ far from the prediction), or pred<d>, the raw predicted organ dilated by d voxels, which needs
   no labels and is what a reviewer would actually be shown. Outputs get a _<population> suffix. Confidence = top-label probability. Voxels below a
   confidence cut-off tau are "flagged for review", the rest are kept.
   A1 risk-coverage: error rate of the kept voxels vs the fraction kept. This depends only on the RANKING
      of confidences; any calibrator that is monotone in the top-label confidence leaves it unchanged.
   A2 honesty of the cut-off: for every tau, the error rate the model PROMISES on the kept voxels
      (mean of 1 - confidence) against the error rate they actually have. This is where calibration matters:
      with calibrated confidences you can pick tau for a target error rate and get roughly that rate.

B. Case-level quality estimate. Per case and class k, a Dice the model expects from its own probabilities:
      expected Dice_k = 2 * sum_{v predicted k} p_k(v) / (#predicted k + sum_{all v} p_k(v))
   (expected true positives over the expected mask sizes). Compared with the actual argmax Dice of the
   same probabilities: mean absolute error (is the estimate honest?) and Spearman correlation (does it
   rank cases, i.e. can it flag the cases the model got wrong?).

Everything is from per-case sufficient statistics (confidence histograms, sums), so pooled curves are exact
up to the 1000-bin confidence resolution; patient-bootstrap intervals for the summary numbers.

Writes <cache>/usefulness.json and <cache>/usefulness.png.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.stats import spearmanr

import isotonic_map as iso

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--calibrators", nargs="+", default=["calib_pred8_predclass.json", "calib_pred4_isotonic.json"])
ap.add_argument("--bins", type=int, default=1000, help="confidence resolution")
ap.add_argument("--bootstrap", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--population", default="organ",
                help="voxels for part A: 'organ' (label != 0 OR raw argmax != 0; uses labels, evaluation only), "
                     "'pred<d>' (raw predicted organ dilated by d voxels; what a reviewer would see, no labels), or 'all'")
a = ap.parse_args()
cache = Path(a.cache)
assert a.population in ("organ", "all") or (a.population.startswith("pred") and a.population[4:].isdigit()), a.population
SUFFIX = "" if a.population == "organ" else f"_{a.population}"
POP_TEXT = {"organ": "organ region (label != 0 OR raw argmax != 0)", "all": "whole volume"}.get(
    a.population, f"predicted organ dilated by {a.population[4:]} voxels (no labels)")
split = json.load(open(cache / "split.json"))
run = json.load(open(cache / "run.json"))
CALS = {"T1": {"method": "temperature", "T": 1.0, "region": "all"}}
for fn in a.calibrators:
    d = json.load(open(cache / fn))
    d.setdefault("method", "temperature"); d.setdefault("region", "all")
    assert d["region"] in ("all", "pred"), "fg-region calibrators need labels at inference"
    CALS[Path(fn).stem] = d
NAMES = list(CALS)
CLS = {1: "kidney", 2: "tumor"}  # cyst is too rare per case for a case-level estimate
NB = a.bins
ST = generate_binary_structure(3, 1)


def softmax(z):
    z = z - z.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


def probs(L, cal, regions):
    """L (K, n) raw logits -> calibrated probabilities (K, n), same rules as evaluate_calibration.py."""
    if cal["method"] == "temperature" and cal["region"] == "all":
        return softmax(L / np.float32(cal["T"]))
    P = softmax(L)
    m = regions[cal["dilate"]] if cal["region"] == "pred" else np.ones(L.shape[1], bool)
    Lm = L[:, m]
    if cal["method"] == "isotonic":
        P[:, m] = iso.apply(softmax(Lm), cal, R=Lm.argmax(axis=0))
    elif cal["method"] == "predclass":
        P[:, m] = softmax(Lm / np.asarray(cal["T_by_pred_class"], dtype=np.float32)[Lm.argmax(axis=0)][None])
    else:
        P[:, m] = softmax(Lm / np.float32(cal["T"]))
    return P


cases = split["test"]
H = {c: np.zeros((len(cases), 2, NB)) for c in NAMES}         # per case: [count, correct] by confidence bin
SC = {c: np.zeros((len(cases), NB)) for c in NAMES}           # per case: sum of confidence by bin
CASE = {c: {k: {"expected": [], "actual": []} for k in CLS.values()} for c in NAMES}
t0 = time.time()
for i, cid in enumerate(cases):
    z = np.load(cache / f"{cid}.npz")
    L = z["logits"].astype(np.float32)
    Y = z["label"].reshape(-1)
    raw_pred = L.argmax(axis=0)
    regions = {}
    for c in CALS.values():
        if c["region"] == "pred" and c["dilate"] not in regions:
            regions[c["dilate"]] = binary_dilation(raw_pred != 0, structure=ST, iterations=c["dilate"]).reshape(-1) \
                if c["dilate"] else (raw_pred != 0).reshape(-1)
    L = L.reshape(L.shape[0], -1)
    if a.population == "organ":
        organ = (Y != 0) | (raw_pred.reshape(-1) != 0)
    elif a.population == "all":
        organ = np.ones(Y.shape, bool)
    else:
        d = int(a.population[4:])
        organ = (binary_dilation(raw_pred != 0, structure=ST, iterations=d) if d else raw_pred != 0).reshape(-1)
    for c in NAMES:
        P = probs(L, CALS[c], regions)
        pred = P.argmax(axis=0)
        conf = P.max(axis=0)[organ]
        b = np.minimum((conf * NB).astype(np.int64), NB - 1)
        H[c][i, 0] = np.bincount(b, minlength=NB)
        H[c][i, 1] = np.bincount(b, weights=(pred[organ] == Y[organ]).astype(float), minlength=NB)
        SC[c][i] = np.bincount(b, weights=conf.astype(float), minlength=NB)
        for k, name in CLS.items():
            pk = P[k]
            pm = pred == k
            gt = Y == k
            n_pred, s_all = pm.sum(), pk.sum(dtype=np.float64)
            if gt.sum() == 0 and n_pred == 0:
                continue
            exp_d = 2 * pk[pm].sum(dtype=np.float64) / (n_pred + s_all) if (n_pred + s_all) > 0 else np.nan
            act_d = 2 * (pm & gt).sum() / (n_pred + gt.sum())
            CASE[c][name]["expected"].append(float(exp_d)); CASE[c][name]["actual"].append(float(act_d))
    print(f"[{i + 1}/{len(cases)}] {cid}  {time.time() - t0:.0f}s", flush=True)
    for c in NAMES:
        for name in CLS.values():
            CASE[c][name].setdefault("case", [])
            if len(CASE[c][name]["case"]) < len(CASE[c][name]["actual"]):
                CASE[c][name]["case"].append(cid)


def curves(h, sc):
    """pooled bins -> kept fraction, actual error of kept, promised error of kept, for tau = bin edges."""
    cnt, cor, s = h[0], h[1], sc
    # keep voxels with confidence >= tau: cumulative from the top bin down
    n = cnt[::-1].cumsum()[::-1]
    err = (cnt - cor)[::-1].cumsum()[::-1]
    promised = (cnt - s)[::-1].cumsum()[::-1]  # sum of (1 - conf)
    with np.errstate(invalid="ignore", divide="ignore"):
        return n / n[0], err / n, promised / n


def aurc(h, sc):
    cov, risk, _ = curves(h, sc)
    ok = np.isfinite(risk)
    x, y = cov[ok][::-1], risk[ok][::-1]  # increasing coverage
    return float(np.trapz(y, x))


def honesty_gap(h, sc):
    """mean |actual - promised| error of the kept set over cut-offs keeping 50-100% of voxels (coverage-weighted grid)."""
    cov, risk, prom = curves(h, sc)
    grid = np.linspace(0.5, 1.0, 51)
    ok = np.isfinite(risk)
    order = np.argsort(cov[ok])
    r = np.interp(grid, cov[ok][order], risk[ok][order])
    p = np.interp(grid, cov[ok][order], prom[ok][order])
    return float(np.mean(np.abs(r - p)))


def case_stats(e, act):
    e, act = np.asarray(e), np.asarray(act)
    ok = np.isfinite(e)
    return {"mae": float(np.mean(np.abs(e[ok] - act[ok]))), "bias": float(np.mean(e[ok] - act[ok])),
            "spearman": float(spearmanr(e[ok], act[ok]).correlation), "n": int(ok.sum())}


rng = np.random.default_rng(a.seed)
idx = [rng.integers(0, len(cases), len(cases)) for _ in range(a.bootstrap)]
res = {"cache": str(cache), "checkpoint_sha256": run.get("checkpoint_sha256"), "test_cases": len(cases),
       "label": "exploratory", "region": POP_TEXT, "population": a.population, "calibrators": a.calibrators,
       "voxel": {}, "case": {}}
for c in NAMES:
    h, sc = H[c].sum(axis=0), SC[c].sum(axis=0)
    boot_aurc = [aurc(H[c][j].sum(0), SC[c][j].sum(0)) for j in idx]
    boot_gap = [honesty_gap(H[c][j].sum(0), SC[c][j].sum(0)) for j in idx]
    res["voxel"][c] = {"aurc": aurc(h, sc), "aurc_ci95": np.percentile(boot_aurc, [2.5, 97.5]).tolist(),
                       "honesty_gap": honesty_gap(h, sc), "honesty_gap_ci95": np.percentile(boot_gap, [2.5, 97.5]).tolist(),
                       "error_at_full_coverage": float(curves(h, sc)[1][0])}
    res["case"][c] = {name: {**case_stats(v["expected"], v["actual"]), "per_case": v} for name, v in CASE[c].items()}

print(f"\n{len(cases)} test patients (exploratory) · {POP_TEXT}")
print(f"{'':<24}{'AURC (lower=better ranking)':>30}{'honesty gap (lower=more honest)':>36}")
for c in NAMES:
    v = res["voxel"][c]
    print(f"{c:<24}{v['aurc']:>12.4f} [{v['aurc_ci95'][0]:.4f}, {v['aurc_ci95'][1]:.4f}]"
          f"{v['honesty_gap']:>16.4f} [{v['honesty_gap_ci95'][0]:.4f}, {v['honesty_gap_ci95'][1]:.4f}]")
print("\ncase-level expected vs actual Dice")
for name in CLS.values():
    for c in NAMES:
        s = res["case"][c][name]
        print(f"  {name:<7}{c:<24} MAE {s['mae']:.3f}  bias {s['bias']:+.3f}  Spearman {s['spearman']:.3f}  (n={s['n']})")
json.dump(res, open(cache / f"usefulness{SUFFIX}.json", "w"), indent=1)

# ------------------------------------------------------------------ figure
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

col = {"T1": "#1f77b4"}
palette = ["#2ca02c", "#d62728", "#9467bd", "#8c564b"]
for j, c in enumerate(NAMES[1:]):
    col[c] = palette[j % len(palette)]
fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
for c in NAMES:
    cov, risk, prom = curves(H[c].sum(0), SC[c].sum(0))
    ax[0].plot(cov, risk, c=col[c], label=f"{c} (AURC {res['voxel'][c]['aurc']:.4f})")
    ok = cov >= 0.05
    ax[1].plot(prom[ok], risk[ok], c=col[c], label=f"{c} (gap {res['voxel'][c]['honesty_gap']:.4f})")
ax[0].set(xlabel=f"fraction kept of: {POP_TEXT}", ylabel="error rate of kept voxels",
          title="A1 risk-coverage (ranking)", xlim=(0, 1))
lim = max(ax[1].get_xlim()[1], ax[1].get_ylim()[1])
ax[1].plot([0, lim], [0, lim], ls="--", c="0.5", lw=1)
ax[1].set(xlabel="error rate the model promises on kept voxels", ylabel="actual error rate of kept voxels",
          title="A2 honesty of a confidence cut-off\n(diagonal = you get the error rate you were promised)")
for a_, name in zip(ax[2:], CLS.values()):
    for c in NAMES:
        s = res["case"][c][name]
        a_.scatter(s["per_case"]["expected"], s["per_case"]["actual"], s=14, alpha=0.65, c=col[c],
                   label=f"{c}: MAE {s['mae']:.3f}, rho {s['spearman']:.2f}")
    a_.plot([0, 1], [0, 1], ls="--", c="0.5", lw=1)
    a_.set(xlim=(0, 1), ylim=(0, 1), xlabel=f"expected {name} Dice (from the probabilities)",
           ylabel=f"actual {name} Dice", title=f"B case-level {name} quality estimate")
for a_ in ax:
    a_.legend(fontsize=8)
fig.suptitle(f"Is the confidence useful?  ·  {len(cases)} test patients  ·  exploratory  ·  part A on {POP_TEXT}", y=1.03)
plt.tight_layout()
plt.savefig(cache / f"usefulness{SUFFIX}.png", dpi=130, bbox_inches="tight")
print("wrote", cache / f"usefulness{SUFFIX}.json", "and", cache / f"usefulness{SUFFIX}.png")
