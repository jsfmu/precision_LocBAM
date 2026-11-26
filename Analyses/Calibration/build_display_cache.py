"""Step 4: write the small per-case files the calibration viewer notebook reads.

The viewer never runs the model and never reads the large logit files. For every test case of a
cache it needs, per case:
  ct        uint8 (x, y, z)     the preprocessed CT (same crop / 1.5 mm grid as the logits), for display
  label     uint8 (x, y, z)     ground truth on the same grid
  p_raw     uint8 (3, x, y, z)  raw softmax p(kidney), p(tumor), p(cyst), scaled 0..255
  p_cal     uint8 (3, x, y, z)  the same after the calibrator
  hist_raw / hist_cal  int64 (3, 2, B + 1)  FULL-resolution counts of p_k in bins j = floor(p_k * B),
            split by label == k (row 1) or not (row 0). Cumulative sums from the top give TP / FP / FN of
            the mask p_k >= j / B at every threshold, exactly as evaluate_calibration.py's sweeps.
  key       json string: case, checkpoint sha256, preprocessing version, calibrator file + sha256, downsampling

Only the display arrays are downsampled (every `--down`-th voxel per axis); Dice / precision / recall
shown in the viewer come from the full-resolution histograms, never from the display arrays.

The CT is not in the logit cache, so it is recomputed here with the cache's own validation transforms
(CPU only, no model), and the resulting label is checked voxel-for-voxel against the cached label so the
CT is guaranteed to line up with the probabilities.

Writes <cache>/display/<case>.npz and <cache>/display/index.json.

Example:
  python Analyses/Calibration/build_display_cache.py --cache data/cache/851c43ca5339
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_logits import PREPROCESSING  # noqa: E402  (also sets up the MONAI / LocBAM imports)

CLASSES = ["kidney", "tumor", "cyst"]  # label values 1, 2, 3

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--calibrator", default="calib_pred8_predclass.json", help="calibrator JSON inside --cache")
ap.add_argument("--cases", nargs="*", default=None, help="default: every test case in split.json")
ap.add_argument("--down", type=int, default=2, help="display downsampling per axis")
ap.add_argument("--bins", type=int, default=100, help="threshold resolution of the histograms")
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()
cache = Path(a.cache)
out = cache / "display"
out.mkdir(exist_ok=True)

run = json.load(open(cache / "run.json"))
split = json.load(open(cache / "split.json"))
cases = a.cases or split["test"]
cal_path = cache / a.calibrator
cal = json.load(open(cal_path))
cal.setdefault("method", "temperature")
cal.setdefault("region", "all")
cal_sha = hashlib.sha256(cal_path.read_bytes()).hexdigest()
assert cal["method"] in ("temperature", "predclass"), "viewer supports argmax-preserving calibrators only"
metrics = json.load(open(cache / "metrics.json")) if (cache / "metrics.json").exists() else None
by_version = {v: f for f, v in PREPROCESSING.values()}
ST = generate_binary_structure(3, 1)


def softmax(z):
    z = z - z.max(axis=0, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=0, keepdims=True)
    return z


def calibrated_logits(L):
    """Same rule as evaluate_calibration.py: inside the dilated predicted region for region 'pred'."""
    pred = L.argmax(axis=0)
    if cal["method"] == "predclass":
        T = np.asarray(cal["T_by_pred_class"], dtype=np.float32)[pred]
    else:
        T = np.full(pred.shape, float(cal["T"]), dtype=np.float32)
    if cal["region"] == "pred":
        region = binary_dilation(pred != 0, structure=ST, iterations=int(cal["dilate"])) if cal["dilate"] else pred != 0
        T = np.where(region, T, np.float32(1.0))
    elif cal["region"] != "all":
        raise ValueError("region 'fg' needs labels and cannot be used at inference")
    return L / T[None]


def hists(P, Y):
    H = np.zeros((3, 2, a.bins + 1), dtype=np.int64)
    for k in range(3):
        b = np.minimum((P[k + 1] * a.bins).astype(np.int64), a.bins).ravel()
        pos = (Y == k + 1).ravel()
        H[k, 1] = np.bincount(b[pos], minlength=a.bins + 1)
        H[k, 0] = np.bincount(b[~pos], minlength=a.bins + 1)
    return H


index = []
t0 = time.time()
for n, cid in enumerate(cases, 1):
    target = out / f"{cid}.npz"
    z = np.load(cache / f"{cid}.npz")
    meta = json.loads(str(z["meta"]))
    if not target.exists() or a.overwrite:
        L = z["logits"].astype(np.float32)
        Y = z["label"]
        data = by_version[meta["preprocessing_version"]]()({"image": meta["image_path"], "label": meta["label_path"]})
        lab = data["label"][0].numpy().astype(np.uint8)
        if lab.shape != Y.shape or not np.array_equal(lab, Y):
            raise RuntimeError(f"{cid}: recomputed label does not match the cached one; CT would be misaligned")
        ct = data["image"][0].numpy()  # channel 0 = scaled CT (0..1); the rest are location channels

        Lc = calibrated_logits(L)
        assert np.array_equal(Lc.argmax(axis=0), L.argmax(axis=0)), "calibrator changed the argmax"
        P_raw, P_cal = softmax(L), softmax(Lc)
        s = (slice(None, None, a.down),) * 3
        q = lambda P: np.round(P[1:][(slice(None),) + s] * 255).astype(np.uint8)  # noqa: E731
        key = {"case": cid, "checkpoint_sha256": meta["checkpoint_sha256"],
               "preprocessing_version": meta["preprocessing_version"], "calibrator": a.calibrator,
               "calibrator_sha256": cal_sha, "down": a.down, "full_shape": list(Y.shape), "bins": a.bins}
        tmp = target.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, ct=np.round(np.clip(ct[s], 0, 1) * 255).astype(np.uint8), label=Y[s],
                            p_raw=q(P_raw), p_cal=q(P_cal), hist_raw=hists(P_raw, Y), hist_cal=hists(P_cal, Y),
                            key=json.dumps(key))
        tmp.rename(target)
        status = f"{time.time() - t0:.0f}s"
    else:
        status = "exists"
    row = {"case": cid, "shape": meta["shape_xyz"]}
    if metrics and cid in metrics["per_case"]:
        row["metrics"] = metrics["per_case"][cid]
    index.append(row)
    print(f"[{n}/{len(cases)}] {cid}  {status}", flush=True)

json.dump({"cache": str(cache), "checkpoint_sha256": run.get("checkpoint_sha256"), "calibrator": a.calibrator,
           "calibrator_sha256": cal_sha, "calibrator_name_in_metrics": cal_path.stem, "classes": CLASSES,
           "down": a.down, "bins": a.bins, "cases": index}, open(out / "index.json", "w"), indent=1)
print("wrote", out / "index.json")
