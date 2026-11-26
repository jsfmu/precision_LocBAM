"""Step 3: before/after calibration evaluation on the TEST patients only.

Compares the raw softmax ("T1") against every calibrator fitted by fit_temperature.py on the
calibration patients: temperature.json plus every calib_*.json in the cache (or those given with
--calibrators). All test voxels are used, no subsampling.

Per case it accumulates sufficient statistics (histograms and sums), so pooled metrics are exact and
the patient-level bootstrap only re-sums per-case arrays instead of re-reading volumes. The bootstrap
is paired: every calibrator is scored on the same resampled patients, so differences are tight.

Two voxel sets are reported:
  all  every voxel of the cropped volume (>99% background; ECE here is near 0 for any model)
  fg   the organ region: voxels whose label OR raw (uncalibrated) argmax is not background. The region
       is fixed by the raw argmax, so every calibrator is scored on exactly the same voxels.

Metrics per calibrator: NLL, Brier, top-label ECE, class-wise ECE (each on all and fg), per-class
threshold sweeps (precision / recall / Dice of p_k >= t), and argmax Dice. Temperature scaling cannot
change the argmax, nor can predclass (checked); classwise, isotonic and vector scaling can, so Dice is reported
per calibrator.

Calibrators fitted with --region pred are two-region: applied only inside the predicted region (raw
argmax != background, dilated by the calibrator's own `dilate`), identity (T = 1) everywhere else.
That region needs no labels, so this is exactly what would run at inference time.

Writes <cache>/metrics.json and, with --plots, <cache>/reliability_fg.png and <cache>/threshold_sweeps.png.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

import isotonic_map as iso

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True, help="data/cache/<hash> folder from cache_logits.py")
ap.add_argument("--calibrators", nargs="*", default=None,
                help="calibrator JSON names inside --cache (default: temperature.json + all calib_*.json)")
ap.add_argument("--ece-bins", type=int, default=15)
ap.add_argument("--sweep-bins", type=int, default=100, help="threshold resolution: t = j / sweep_bins")
ap.add_argument("--bootstrap", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--slab", type=int, default=16, help="x-slices processed at once (memory vs speed)")
ap.add_argument("--plots", action="store_true")
a = ap.parse_args()
cache = Path(a.cache)

split = json.load(open(cache / "split.json"))
run = json.load(open(cache / "run.json"))
assert not set(split["calibration"]) & set(split["test"]), "calibration and test patients overlap"

files = a.calibrators
if files is None:
    files = (["temperature.json"] if (cache / "temperature.json").exists() else []) + \
            sorted(p.name for p in cache.glob("calib_*.json"))
CALS = {"T1": {"method": "temperature", "T": 1.0}}
for fn in files:
    d = json.load(open(cache / fn))
    d.setdefault("method", "temperature")
    d.setdefault("region", "all")
    assert d.get("split") == "split.json" and d.get("seed", split["seed"]) == split["seed"], \
        f"{fn} was not fitted on this split.json"
    CALS[Path(fn).stem] = d
NAMES = list(CALS)
OTHERS = NAMES[1:]

LABELS = {int(k): v for k, v in run["label_map"].items()}
K = len(LABELS)
FG = [k for k in LABELS if k != 0]
EB, SB = a.ece_bins, a.sweep_bins
def describe(c):
    if c["method"] == "temperature":
        s = f"T={c['T']:.4f}"
    elif c["method"] == "classwise":
        s = "T_k=" + ",".join(f"{t:.3f}" for t in c["T_per_class"])
    elif c["method"] == "predclass":
        s = "T_by_pred=" + ",".join(f"{t:.3f}" for t in c["T_by_pred_class"])
    elif c["method"] == "isotonic":
        s = "isotonic per (pred class, class)"
    else:
        s = "vector w,b"
    reg = c.get("region", "-")
    return s + (f", fit+applied in pred region (dilate {c['dilate']}), T=1 outside" if reg == "pred" else f", fit on {reg}")


print(f"test cases: {len(split['test'])}   label_map_confirmed: {run.get('label_map_confirmed')}")
for n in NAMES:
    print(f"  {n:<26} {describe(CALS[n])}")
DILATES = sorted({c["dilate"] for c in CALS.values() if c.get("region") == "pred"})


def transform(z, cal):
    """z (K, n) float32 raw logits -> calibrated scores whose softmax is the calibrated probability."""
    if cal["method"] == "temperature":
        return z / np.float32(cal["T"])
    if cal["method"] == "classwise":
        return z / np.asarray(cal["T_per_class"], dtype=np.float32)[:, None]
    if cal["method"] == "predclass":  # z here is always the RAW logits, so argmax(z) is the raw prediction
        return z / np.asarray(cal["T_by_pred_class"], dtype=np.float32)[z.argmax(axis=0)][None, :]
    if cal["method"] == "isotonic":  # log of the calibrated probabilities; argmax(z) is the raw prediction
        return iso.calibrated_scores(z, cal)
    w = np.asarray(cal["w"], dtype=np.float32)[:, None]
    b = np.asarray(cal["b"], dtype=np.float32)[:, None]
    return z * w + b


def scores(z, cal, pred_regions):
    """Apply cal to the whole chunk, or (region 'pred') only inside its predicted region, identity outside."""
    if cal.get("region") != "pred":
        return transform(z, cal)
    m = pred_regions[cal["dilate"]]
    out = z.copy()
    if m.any():
        out[:, m] = transform(z[:, m], cal)
    return out


def empty_stats():
    s = {"n": 0.0, "nll": 0.0, "brier": 0.0, "n_fg": 0.0, "nll_fg": 0.0, "brier_fg": 0.0}
    for reg in ("", "f"):
        for q in ("cnt", "sp", "sy"):
            s[f"{reg}top_{q}"] = np.zeros(EB)
            s[f"{reg}cls_{q}"] = np.zeros((K, EB))
    s["sw_pos"] = np.zeros((K, SB + 1))
    s["sw_neg"] = np.zeros((K, SB + 1))
    return s


def accumulate(s, z, label, fgmask):
    """z (K, n) calibrated scores, label (n,), fgmask (n,) bool. Adds this chunk's statistics into s."""
    z = z - z.max(axis=0, keepdims=True)
    e = np.exp(z)
    se = e.sum(axis=0)
    p = e / se
    n = label.size
    ar = np.arange(n)
    logp_y = z[label, ar] - np.log(se)
    p_true = p[label, ar]
    brier = (p ** 2).sum(axis=0) - 2 * p_true + 1.0
    pred = p.argmax(axis=0)

    s["n"] += n
    s["nll"] += float(-logp_y.sum())
    s["brier"] += float(brier.sum())
    s["n_fg"] += int(fgmask.sum())
    s["nll_fg"] += float(-logp_y[fgmask].sum())
    s["brier_fg"] += float(brier[fgmask].sum())

    # top-label
    conf = p[pred, ar]
    b = np.minimum((conf * EB).astype(np.int64), EB - 1)
    for reg, m in (("", slice(None)), ("f", fgmask)):
        s[f"{reg}top_cnt"] += np.bincount(b[m], minlength=EB)
        s[f"{reg}top_sp"] += np.bincount(b[m], weights=conf[m], minlength=EB)
        s[f"{reg}top_sy"] += np.bincount(b[m], weights=(pred == label)[m], minlength=EB)
    # class-wise and threshold sweeps
    for k in range(K):
        pk = p[k]
        yk = label == k
        b = np.minimum((pk * EB).astype(np.int64), EB - 1)
        for reg, m in (("", slice(None)), ("f", fgmask)):
            s[f"{reg}cls_cnt"][k] += np.bincount(b[m], minlength=EB)
            s[f"{reg}cls_sp"][k] += np.bincount(b[m], weights=pk[m], minlength=EB)
            s[f"{reg}cls_sy"][k] += np.bincount(b[m], weights=yk[m], minlength=EB)
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
    stats = {c: empty_stats() for c in NAMES}
    conf_mat = {c: np.zeros((K, K), dtype=np.int64) for c in NAMES}  # [label, pred]
    # predicted regions need the whole volume (dilation crosses slab borders), computed once per dilation
    regions3d = {}
    if DILATES:
        raw_fg = L.argmax(axis=0) != 0
        st = generate_binary_structure(3, 1)
        for d in DILATES:
            regions3d[d] = binary_dilation(raw_fg, structure=st, iterations=d) if d > 0 else raw_fg
    for x0 in range(0, Y.shape[0], a.slab):
        lg = L[:, x0:x0 + a.slab].reshape(K, -1).astype(np.float32)
        lb = Y[x0:x0 + a.slab].reshape(-1).astype(np.int64)
        fgmask = (lb != 0) | (lg.argmax(axis=0) != 0)  # fixed by the RAW argmax
        pr = {d: r[x0:x0 + a.slab].reshape(-1) for d, r in regions3d.items()}
        for c in NAMES:
            pred = accumulate(stats[c], scores(lg, CALS[c], pr), lb, fgmask)
            conf_mat[c] += np.bincount(lb * K + pred, minlength=K * K).reshape(K, K)
    for c in NAMES:
        if CALS[c]["method"] in ("temperature", "predclass"):
            assert np.array_equal(conf_mat["T1"], conf_mat[c]), f"{cid}: argmax changed under {c} (bug)"
    per_case[cid] = {"stats": stats, "conf": conf_mat}
    print(f"[{i + 1}/{len(split['test'])}] {cid}  {Y.size:,} voxels  {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- metrics from summed statistics
def ece(cnt, sp, sy):
    n = cnt.sum()
    return float(np.sum(np.abs(sy - sp)) / n) if n else None  # sum_b |sum y - sum p| / N


def summarize(cases, c):
    s = empty_stats()
    for cid in cases:
        cs = per_case[cid]["stats"][c]
        for k in s:
            s[k] = s[k] + cs[k]
    out = {
        "nll": s["nll"] / s["n"], "brier": s["brier"] / s["n"],
        "nll_fg": s["nll_fg"] / s["n_fg"] if s["n_fg"] else None,
        "brier_fg": s["brier_fg"] / s["n_fg"] if s["n_fg"] else None,
        "ece_top": ece(s["top_cnt"], s["top_sp"], s["top_sy"]),
        "ece_top_fg": ece(s["ftop_cnt"], s["ftop_sp"], s["ftop_sy"]),
    }
    for k in FG:
        out[f"ece_{LABELS[k]}"] = ece(s["cls_cnt"][k], s["cls_sp"][k], s["cls_sy"][k])
        out[f"ece_{LABELS[k]}_fg"] = ece(s["fcls_cnt"][k], s["fcls_sp"][k], s["fcls_sy"][k])
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
    nn = lambda v: None if np.isnan(v) else float(v)
    rows = [{"t": j / SB, "tp": int(tp[j]), "fp": int(fp[j]), "fn": int(fn[j]),
             "precision": nn(prec[j]), "recall": nn(rec[j]), "dice": nn(dice[j])} for j in range(SB + 1)]
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
for c in NAMES:
    pooled[c], sums[c] = summarize(test, c)
keys = list(pooled["T1"].keys())

# paired patient-level bootstrap (resample test patients with replacement; same sample for every calibrator)
rng = np.random.default_rng(a.seed)
boot = {c: {k: [] for k in keys + ["dice_" + LABELS[j] for j in FG]} for c in NAMES}
for _ in range(a.bootstrap):
    sample = list(rng.choice(test, size=len(test), replace=True))
    for c in NAMES:
        r, _ = summarize(sample, c)
        r.update({"dice_" + k: v for k, v in dice_from_conf(sum(per_case[x]["conf"][c] for x in sample)).items()})
        for k in boot[c]:
            boot[c][k].append(r[k])


def ci(v):
    v = np.array([x for x in v if x is not None], dtype=float)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else None


def delta(c, k):
    return [None if x is None or y is None else y - x for x, y in zip(boot["T1"][k], boot[c][k])]


dice_pooled = {c: dice_from_conf(sum(per_case[x]["conf"][c] for x in test)) for c in NAMES}
result = {
    "cache": str(cache), "checkpoint_sha256": run["checkpoint_sha256"],
    "calibrators": CALS, "label_map": LABELS, "label_map_confirmed": run.get("label_map_confirmed"),
    "n_test_cases": len(test), "test_cases": test, "ece_bins": EB, "sweep_bins": SB,
    "bootstrap": a.bootstrap, "seed": a.seed,
    "fg_region": "label != background OR raw (T=1) argmax != background; identical voxels for every calibrator",
    "pooled": {c: {k: {"value": pooled[c][k], "ci95": ci(boot[c][k])} for k in keys} for c in NAMES},
    "delta_vs_T1": {c: {k: {"value": None if pooled["T1"][k] is None else pooled[c][k] - pooled["T1"][k],
                            "ci95": ci(delta(c, k))} for k in keys} for c in OTHERS},
    "argmax_dice_pooled": {c: {k: {"value": v, "ci95": ci(boot[c]["dice_" + k])} for k, v in dice_pooled[c].items()}
                           for c in NAMES},
    "delta_dice_vs_T1": {c: {k: {"value": None if dice_pooled["T1"][k] is None else dice_pooled[c][k] - dice_pooled["T1"][k],
                                 "ci95": ci(delta(c, "dice_" + k))} for k in dice_pooled[c]} for c in OTHERS},
    "reliability": {c: {reg: {"top_label": reliability(sums[c][f"{p}top_cnt"], sums[c][f"{p}top_sp"], sums[c][f"{p}top_sy"]),
                              **{LABELS[k]: reliability(sums[c][f"{p}cls_cnt"][k], sums[c][f"{p}cls_sp"][k], sums[c][f"{p}cls_sy"][k])
                                 for k in FG}}
                        for reg, p in (("all", ""), ("fg", "f"))} for c in NAMES},
    "threshold_sweeps": {}, "best_threshold_on_test_DO_NOT_REPORT_AS_TUNED": {},
    "per_case": {},
}
for c in NAMES:
    result["threshold_sweeps"][c], result["best_threshold_on_test_DO_NOT_REPORT_AS_TUNED"][c] = {}, {}
    for k in FG:
        rows, best = sweep(sums[c]["sw_pos"][k], sums[c]["sw_neg"][k])
        result["threshold_sweeps"][c][LABELS[k]] = rows
        result["best_threshold_on_test_DO_NOT_REPORT_AS_TUNED"][c][LABELS[k]] = best
for x in test:
    result["per_case"][x] = {c: {**summarize([x], c)[0], "argmax_dice": dice_from_conf(per_case[x]["conf"][c])}
                             for c in NAMES}

json.dump(result, open(cache / "metrics.json", "w"), indent=1)

# ---------------------------------------------------------------- console summary
print(f"\npooled over {len(test)} test patients; [95% paired patient-bootstrap CI]; * = CI of the change excludes 0")
w = max(len(c) for c in NAMES) + 2
for k in keys:
    print(f"\n{k}")
    print(f"  {'T1':<{w}}{pooled['T1'][k]:.5f}")
    for c in OTHERS:
        d = result["delta_vs_T1"][c][k]
        lo, hi = d["ci95"]
        sig = "*" if lo > 0 or hi < 0 else " "
        print(f"  {c:<{w}}{pooled[c][k]:.5f}   change {d['value']:+.5f} [{lo:+.5f}, {hi:+.5f}] {sig}")
print("\nargmax Dice")
for c in NAMES:
    print(f"  {c:<{w}}" + "  ".join(f"{k} {v['value']:.4f}" for k, v in result["argmax_dice_pooled"][c].items() if v["value"] is not None))
print("\nwrote", cache / "metrics.json")

# ---------------------------------------------------------------- optional figures
if a.plots:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = ["top_label"] + [LABELS[k] for k in FG]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4), squeeze=False)
    for ax, name in zip(axes[0], panels):
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        for c in NAMES:
            bins = [b for b in result["reliability"][c]["fg"][name] if b["n"] > 0]
            ax.plot([b["mean_p"] for b in bins], [b["freq"] for b in bins], "o-", ms=3, label=c)
        key = "ece_top_fg" if name == "top_label" else f"ece_{name}_fg"
        ax.set_title(f"{name} (organ region)\n" + "  ".join(f"{pooled[c][key]:.3f}" for c in NAMES) + "  ECE", fontsize=9)
        ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(cache / "reliability_fg.png", dpi=150)

    fig, axes = plt.subplots(1, len(FG), figsize=(4.2 * len(FG), 4.2), squeeze=False)
    for ax, k in zip(axes[0], FG):
        for c in NAMES:
            rows = result["threshold_sweeps"][c][LABELS[k]]
            ax.plot([r["t"] for r in rows], [np.nan if r["dice"] is None else r["dice"] for r in rows], label=f"Dice {c}")
            ax.plot([r["t"] for r in rows], [np.nan if r["precision"] is None else r["precision"] for r in rows], ":", lw=1)
            ax.plot([r["t"] for r in rows], [np.nan if r["recall"] is None else r["recall"] for r in rows], "--", lw=1)
        ax.set_title(f"{LABELS[k]}: Dice (solid), precision (dotted), recall (dashed)", fontsize=8)
        ax.set_xlabel("threshold t on p_k"); ax.set_ylim(0, 1)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(cache / "threshold_sweeps.png", dpi=150)
    print("wrote", cache / "reliability_fg.png", "and", cache / "threshold_sweeps.png")
