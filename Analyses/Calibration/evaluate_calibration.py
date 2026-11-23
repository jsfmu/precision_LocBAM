"""Step 3: before/after calibration evaluation on the TEST patients only.

Reads <cache>/split.json and <cache>/temperature.json written by fit_temperature.py, and compares
T = 1 (raw softmax) against the fitted T on every test case, using all voxels (no subsampling).

Per case it accumulates sufficient statistics (histograms and sums), so pooled metrics are exact and
the patient-level bootstrap only re-sums per-case arrays instead of re-reading volumes.

Metrics, for both temperatures:
  - NLL and Brier (multiclass, mean over voxels), over all voxels and over the "foreground region"
    (voxels where ground truth or argmax prediction is not background)
  - top-label ECE (confidence = max_k p_k, correct = argmax == label)
  - class-wise ECE for kidney / tumor / cyst (one-vs-rest p_k against [label == k])
  - per-class threshold sweeps: precision / recall / Dice of the mask p_k >= t for t = 0.00 ... 1.00
  - argmax Dice per class (must be identical for both temperatures: T > 0 cannot change the argmax)
  - patient-level bootstrap 95% CIs for every pooled scalar and for the after - before difference

Writes <cache>/metrics.json and, with --plots, <cache>/reliability.png and <cache>/threshold_sweeps.png.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True, help="data/cache/<hash> folder from cache_logits.py")
ap.add_argument("--ece-bins", type=int, default=15)
ap.add_argument("--sweep-bins", type=int, default=100, help="threshold resolution: t = j / sweep_bins")
ap.add_argument("--bootstrap", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--slab", type=int, default=16, help="x-slices processed at once (memory vs speed)")
ap.add_argument("--plots", action="store_true")
a = ap.parse_args()
cache = Path(a.cache)

split = json.load(open(cache / "split.json"))
temp = json.load(open(cache / "temperature.json"))
run = json.load(open(cache / "run.json"))
assert not set(split["calibration"]) & set(split["test"]), "calibration and test patients overlap"
assert temp.get("split") == "split.json", "temperature.json was not fitted on this split.json"

LABELS = {int(k): v for k, v in run["label_map"].items()}
K = len(LABELS)
FG = [k for k in LABELS if k != 0]
TEMPS = {"T1": 1.0, "Tfit": float(temp["T"])}
EB, SB = a.ece_bins, a.sweep_bins
print(f"test cases: {len(split['test'])}   temperatures: {TEMPS}   label_map_confirmed: {run.get('label_map_confirmed')}")


def empty_stats():
    return {
        "n": 0, "nll": 0.0, "brier": 0.0, "n_fg": 0, "nll_fg": 0.0, "brier_fg": 0.0,
        "top_cnt": np.zeros(EB), "top_sp": np.zeros(EB), "top_sy": np.zeros(EB),
        "cls_cnt": np.zeros((K, EB)), "cls_sp": np.zeros((K, EB)), "cls_sy": np.zeros((K, EB)),
        "sw_pos": np.zeros((K, SB + 1)), "sw_neg": np.zeros((K, SB + 1)),
    }


def accumulate(s, logits, label, T):
    """logits (K, n) float32, label (n,) int. Adds this chunk's statistics into s."""
    z = logits / T
    z -= z.max(axis=0, keepdims=True)
    e = np.exp(z)
    se = e.sum(axis=0)
    p = e / se
    n = label.size
    ar = np.arange(n)
    logp_y = z[label, ar] - np.log(se)
    onehot = np.zeros_like(p)
    onehot[label, ar] = 1.0
    brier = ((p - onehot) ** 2).sum(axis=0)
    pred = p.argmax(axis=0)
    fg = (label != 0) | (pred != 0)

    s["n"] += n
    s["nll"] += float(-logp_y.sum())
    s["brier"] += float(brier.sum())
    s["n_fg"] += int(fg.sum())
    s["nll_fg"] += float(-logp_y[fg].sum())
    s["brier_fg"] += float(brier[fg].sum())

    conf = p[pred, ar]
    b = np.minimum((conf * EB).astype(np.int64), EB - 1)
    s["top_cnt"] += np.bincount(b, minlength=EB)
    s["top_sp"] += np.bincount(b, weights=conf, minlength=EB)
    s["top_sy"] += np.bincount(b, weights=(pred == label), minlength=EB)

    for k in range(K):
        pk = p[k]
        yk = label == k
        b = np.minimum((pk * EB).astype(np.int64), EB - 1)
        s["cls_cnt"][k] += np.bincount(b, minlength=EB)
        s["cls_sp"][k] += np.bincount(b, weights=pk, minlength=EB)
        s["cls_sy"][k] += np.bincount(b, weights=yk, minlength=EB)
        sb = np.minimum(np.floor(pk * SB + 1e-9).astype(np.int64), SB)
        s["sw_pos"][k] += np.bincount(sb[yk], minlength=SB + 1)
        s["sw_neg"][k] += np.bincount(sb[~yk], minlength=SB + 1)
    return pred


# ---------------------------------------------------------------- per-case pass (the slow part)
per_case = {}
t0 = time.time()
for i, cid in enumerate(split["test"]):
    z = np.load(cache / f"{cid}.npz")
    L, Y = z["logits"], z["label"]  # (K, X, Y, Z) fp16, (X, Y, Z) uint8
    stats = {t: empty_stats() for t in TEMPS}
    conf_mat = {t: np.zeros((K, K), dtype=np.int64) for t in TEMPS}  # [label, pred]
    for x0 in range(0, Y.shape[0], a.slab):
        lg = L[:, x0:x0 + a.slab].reshape(K, -1).astype(np.float32)
        lb = Y[x0:x0 + a.slab].reshape(-1).astype(np.int64)
        for t, T in TEMPS.items():
            pred = accumulate(stats[t], lg, lb, T)
            conf_mat[t] += np.bincount(lb * K + pred, minlength=K * K).reshape(K, K)
    assert np.array_equal(conf_mat["T1"], conf_mat["Tfit"]), f"{cid}: argmax changed with T (bug)"
    per_case[cid] = {"stats": stats, "conf": conf_mat["T1"]}
    print(f"[{i + 1}/{len(split['test'])}] {cid}  {Y.size:,} voxels  {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- metrics from summed statistics
def ece(cnt, sp, sy):
    n = cnt.sum()
    m = cnt > 0
    return float(np.sum(np.abs(sy[m] - sp[m])) / n) if n else None  # sum_b |sum y - sum p| / N


def summarize(cases, t):
    s = empty_stats()
    for c in cases:
        cs = per_case[c]["stats"][t]
        for k in s:
            s[k] = s[k] + cs[k]
    out = {
        "nll": s["nll"] / s["n"], "brier": s["brier"] / s["n"],
        "nll_fg": s["nll_fg"] / s["n_fg"] if s["n_fg"] else None,
        "brier_fg": s["brier_fg"] / s["n_fg"] if s["n_fg"] else None,
        "ece_top": ece(s["top_cnt"], s["top_sp"], s["top_sy"]),
    }
    for k in FG:
        out[f"ece_{LABELS[k]}"] = ece(s["cls_cnt"][k], s["cls_sp"][k], s["cls_sy"][k])
    return out, s


def reliability(cnt, sp, sy):
    return [{"lo": b / EB, "hi": (b + 1) / EB, "n": int(cnt[b]),
             "mean_p": float(sp[b] / cnt[b]) if cnt[b] else None,
             "freq": float(sy[b] / cnt[b]) if cnt[b] else None} for b in range(EB)]


def sweep(pos, neg):
    tp = np.cumsum(pos[::-1])[::-1]  # voxels with bin >= j, i.e. p >= j / SB
    fp = np.cumsum(neg[::-1])[::-1]
    fn = pos.sum() - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        rec = np.where(pos.sum() > 0, tp / pos.sum(), np.nan)
        dice = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    rows = [{"t": j / SB, "tp": int(tp[j]), "fp": int(fp[j]), "fn": int(fn[j]),
             "precision": None if np.isnan(prec[j]) else float(prec[j]),
             "recall": None if np.isnan(rec[j]) else float(rec[j]),
             "dice": None if np.isnan(dice[j]) else float(dice[j])} for j in range(SB + 1)]
    best = int(np.nanargmax(dice)) if np.any(~np.isnan(dice)) else None
    return rows, (rows[best] if best is not None else None)


def dice_from_conf(C):
    out = {}
    for k in FG:
        tp, fp, fn = C[k, k], C[:, k].sum() - C[k, k], C[k, :].sum() - C[k, k]
        out[LABELS[k]] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
    return out


test = split["test"]
pooled, sums = {}, {}
for t in TEMPS:
    pooled[t], sums[t] = summarize(test, t)

# patient-level bootstrap (resample test patients with replacement)
rng = np.random.default_rng(a.seed)
keys = list(pooled["T1"].keys())
boot = {t: {k: [] for k in keys} for t in TEMPS}
for _ in range(a.bootstrap):
    sample = list(rng.choice(test, size=len(test), replace=True))
    for t in TEMPS:
        r, _ = summarize(sample, t)
        for k in keys:
            boot[t][k].append(r[k])


def ci(v):
    v = np.array([x for x in v if x is not None], dtype=float)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else None


result = {
    "cache": str(cache), "checkpoint_sha256": run["checkpoint_sha256"], "temperatures": TEMPS,
    "label_map": LABELS, "label_map_confirmed": run.get("label_map_confirmed"),
    "n_test_cases": len(test), "test_cases": test, "ece_bins": EB, "sweep_bins": SB,
    "bootstrap": a.bootstrap, "seed": a.seed,
    "pooled": {t: {k: {"value": pooled[t][k], "ci95": ci(boot[t][k])} for k in keys} for t in TEMPS},
    "delta_Tfit_minus_T1": {k: {"value": None if pooled["T1"][k] is None else pooled["Tfit"][k] - pooled["T1"][k],
                                "ci95": ci([None if x is None or y is None else y - x
                                            for x, y in zip(boot["T1"][k], boot["Tfit"][k])])} for k in keys},
    "reliability": {t: {"top_label": reliability(sums[t]["top_cnt"], sums[t]["top_sp"], sums[t]["top_sy"]),
                        **{LABELS[k]: reliability(sums[t]["cls_cnt"][k], sums[t]["cls_sp"][k], sums[t]["cls_sy"][k])
                           for k in FG}} for t in TEMPS},
    "threshold_sweeps": {}, "best_threshold_on_test_DO_NOT_REPORT_AS_TUNED": {},
    "argmax_dice_pooled": dice_from_conf(sum(per_case[c]["conf"] for c in test)),
    "per_case": {},
}
for t in TEMPS:
    result["threshold_sweeps"][t], result["best_threshold_on_test_DO_NOT_REPORT_AS_TUNED"][t] = {}, {}
    for k in FG:
        rows, best = sweep(sums[t]["sw_pos"][k], sums[t]["sw_neg"][k])
        result["threshold_sweeps"][t][LABELS[k]] = rows
        result["best_threshold_on_test_DO_NOT_REPORT_AS_TUNED"][t][LABELS[k]] = best
for c in test:
    result["per_case"][c] = {"argmax_dice": dice_from_conf(per_case[c]["conf"]),
                             **{t: summarize([c], t)[0] for t in TEMPS}}

json.dump(result, open(cache / "metrics.json", "w"), indent=1)

# ---------------------------------------------------------------- console summary
print(f"\nT fitted = {TEMPS['Tfit']:.4f}   (pooled over {len(test)} test patients, 95% patient-bootstrap CI)")
print(f"{'metric':<12}{'T=1':>28}{'T=fit':>28}{'fit - 1':>31}")
for k in keys:
    row = [result["pooled"][t][k] for t in TEMPS]
    d = result["delta_Tfit_minus_T1"][k]
    fmt = lambda r: "—" if r["value"] is None else f"{r['value']:.5f} [{r['ci95'][0]:.5f},{r['ci95'][1]:.5f}]"
    dv = "—" if d["value"] is None else f"{d['value']:+.5f} [{d['ci95'][0]:+.5f},{d['ci95'][1]:+.5f}]"
    print(f"{k:<12}{fmt(row[0]):>28}{fmt(row[1]):>28}{dv:>31}")
print("argmax Dice (identical for both T):", {k: None if v is None else round(v, 4) for k, v in result["argmax_dice_pooled"].items()})
print("wrote", cache / "metrics.json")

# ---------------------------------------------------------------- optional figures
if a.plots:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = ["top_label"] + [LABELS[k] for k in FG]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
    for ax, name in zip(axes[0], panels):
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        for t, style in zip(TEMPS, ["o-", "s-"]):
            bins = [b for b in result["reliability"][t][name] if b["n"] > 0]
            ax.plot([b["mean_p"] for b in bins], [b["freq"] for b in bins], style, ms=4,
                    label=f"{t} (T={TEMPS[t]:.3f})")
        key = "ece_top" if name == "top_label" else f"ece_{name}"
        ax.set_title(f"{name}\nECE {pooled['T1'][key]:.4f} → {pooled['Tfit'][key]:.4f}")
        ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(cache / "reliability.png", dpi=150)

    fig, axes = plt.subplots(1, len(FG), figsize=(4 * len(FG), 4), squeeze=False)
    for ax, k in zip(axes[0], FG):
        for t, ls in zip(TEMPS, ["-", "--"]):
            rows = result["threshold_sweeps"][t][LABELS[k]]
            ts = [r["t"] for r in rows]
            for m, c in [("precision", "C0"), ("recall", "C1"), ("dice", "C2")]:
                ax.plot(ts, [np.nan if r[m] is None else r[m] for r in rows], ls, color=c,
                        label=f"{m} {t}" if k == FG[0] else None)
        ax.set_title(LABELS[k]); ax.set_xlabel("threshold t on p_k"); ax.set_ylim(0, 1)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(cache / "threshold_sweeps.png", dpi=150)
    print("wrote", cache / "reliability.png", "and", cache / "threshold_sweeps.png")
