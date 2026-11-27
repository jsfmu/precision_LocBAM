"""Interactive 3D viewer for the calibration display cache (complements calibration_viewer.ipynb).

Instead of scrolling axial / coronal / sagittal slices, the viewer shows each test case as 3D surfaces you can
rotate, zoom and pan in the browser, with CT planes that can be moved through the volume:

  Ground truth      kidney / tumor / cyst surfaces (label)
  Prediction        argmax surfaces (raw and calibrated argmax are identical for the per-class temperature)
  Errors            the outer surface of (truth OR prediction) per class, each face coloured by the voxel just under
                    it: TP (green) / FP (red) / FN (yellow), the same colours as the notebook
  Confidence        the ground-truth surface coloured by the model's probability for that class (raw or calibrated);
                    dark patches = organ the model was unsure about or missed
  Probability shell the surface p_k >= t for t in LEVELS (raw or calibrated); the gap between low and high shells
                    is where the model is uncertain
  CT planes         three orthogonal CT slices placed in 3D, each with its own slider

Everything comes from <cache>/display/<case>.npz (build_display_cache.py), i.e. the 2x-downsampled display grid
(3 mm voxels). Surfaces are for looking only; Dice shown in the page is the full-resolution value from metrics.json.

Output (local only, under the gitignored data folder):
  <cache>/viewer3d/index.html         the viewer; open it in a browser (works offline)
  <cache>/viewer3d/plotly.min.js      plotly.js, downloaded once at build time
  <cache>/viewer3d/data/<case>.js     per-case data, loaded when the case is selected
  <cache>/viewer3d/cases.js           list of built cases with their metrics

Example:
  python Analyses/Calibration/build_3d_viewer.py --cache data/cache/851c43ca5339 --cases case_00003 case_00032
  python Analyses/Calibration/build_3d_viewer.py --cache data/cache/851c43ca5339          # every case in display/
"""
import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.measure import marching_cubes

CLASSES = ["kidney", "tumor", "cyst"]
LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
SPACING_MM = 1.5  # preprocessing grid; multiplied by the display downsampling
PLOTLY_URL = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--cases", nargs="*", default=None, help="default: every case in <cache>/display/index.json")
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--open", action="store_true", help="open the viewer in the default browser when done")
a = ap.parse_args()
cache = Path(a.cache)
disp = cache / "display"
if not (disp / "index.json").exists():
    raise SystemExit(f"{disp / 'index.json'} not found: run build_display_cache.py --cache {cache} first")
index = json.load(open(disp / "index.json"))
rows = {r["case"]: r for r in index["cases"]}
cases = a.cases or list(rows)
out = cache / "viewer3d"
(out / "data").mkdir(parents=True, exist_ok=True)


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def mesh(vol, level, sp, values=None, cats=None):
    """Marching cubes on vol (x, y, z) at level -> dict of base64 arrays, or None if there is no surface.
    cats: uint8 (x, y, z) field for a binary vol; each vertex gets the value of the in-mask voxel it borders."""
    if vol.max() <= level or vol.min() >= level:
        return None
    v = np.pad(vol.astype(np.float32), 1, constant_values=float(vol.min()))  # closes surfaces at the border
    verts, faces, _, _ = marching_cubes(v, level=level, spacing=(sp, sp, sp))
    verts -= sp  # undo the padding offset
    m = {"x": b64(verts[:, 0].astype(np.float32)), "y": b64(verts[:, 1].astype(np.float32)),
         "z": b64(verts[:, 2].astype(np.float32)), "f": b64(faces.astype(np.uint32).ravel()), "n": int(len(verts))}
    if cats is not None:
        # on a binary mask every vertex sits halfway along one voxel edge: one end is in the mask, the other is not
        g = verts / sp
        lo, hi = np.floor(g + 1e-6).astype(int), np.ceil(g - 1e-6).astype(int)
        shp = np.array(vol.shape) - 1
        lo, hi = np.clip(lo, 0, shp), np.clip(hi, 0, shp)
        in_lo = vol[lo[:, 0], lo[:, 1], lo[:, 2]] > level
        pick = np.where(in_lo[:, None], lo, hi)
        m["c"] = b64(cats[pick[:, 0], pick[:, 1], pick[:, 2]].astype(np.uint8))
    if values is not None:
        m["v"] = {}
        for name, field in values.items():  # sample a (x, y, z) uint8 field at the vertices (nearest voxel)
            ix = np.clip(np.round(verts / sp).astype(int), 0, np.array(field.shape) - 1)
            m["v"][name] = b64(field[ix[:, 0], ix[:, 1], ix[:, 2]].astype(np.uint8))
    return m


def inside_mean(p, mask):
    """Mean of p over the in-mask voxels of each 3x3x3 neighbourhood: the probability just inside the surface,
    so a surface vertex shows what the model thought of the organ there (not the ~0.5 value at the edge)."""
    num = uniform_filter(p.astype(np.float32) * mask, size=3)
    den = uniform_filter(mask.astype(np.float32), size=3)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, 0)
    return np.round(r).astype(np.uint8)


built = []
t0 = time.time()
for n, cid in enumerate(cases, 1):
    target = out / "data" / f"{cid}.js"
    if target.exists() and not a.overwrite:
        print(f"[{n}/{len(cases)}] {cid}  exists", flush=True)
        built.append(cid)
        continue
    d = np.load(disp / f"{cid}.npz")
    key = json.loads(str(d["key"]))
    sp = SPACING_MM * key["down"]
    ct, lab, pr, pc = d["ct"], d["label"], d["p_raw"], d["p_cal"]
    bg = 255 - pr.astype(np.int16).sum(axis=0)
    pred = np.concatenate([bg[None], pr.astype(np.int16)]).argmax(axis=0)  # 0 bg, 1 kidney, 2 tumor, 3 cyst
    layers = {"gt": {}, "pred": {}, "err": {}, "conf": {}, "shell": {"raw": {}, "cal": {}}}
    for k, name in enumerate(CLASSES, start=1):
        g, p = lab == k, pred == k
        layers["gt"][name] = mesh(g, 0.5, sp)
        layers["pred"][name] = mesh(p, 0.5, sp)
        # one outer surface of (truth OR prediction), each vertex tagged 0 TP / 1 FP / 2 FN by the voxel under it
        cat = np.where(g & p, 0, np.where(p, 1, 2)).astype(np.uint8)
        layers["err"][name] = mesh(g | p, 0.5, sp, cats=cat)
        layers["conf"][name] = mesh(g, 0.5, sp, values={"raw": inside_mean(pr[k - 1], g), "cal": inside_mean(pc[k - 1], g)}) \
            if g.any() else None
        for which, P in (("raw", pr), ("cal", pc)):
            layers["shell"][which][name] = {str(t): mesh(P[k - 1], t * 255, sp) for t in LEVELS}
    m = rows.get(cid, {}).get("metrics", {})
    info = {"case": cid, "shape": list(ct.shape), "spacing_mm": sp, "key": key,
            "dice": (m.get("T1") or {}).get("argmax_dice"),
            "voxels": {name: {"gt": int((lab == k).sum()), "pred": int((pred == k).sum())} for k, name in enumerate(CLASSES, 1)}}
    payload = {"info": info, "ct": b64(ct), "layers": layers}
    tmp = target.with_suffix(".tmp")
    tmp.write_text("window.__case_loaded(" + json.dumps(payload) + ");\n")
    tmp.rename(target)
    built.append(cid)
    print(f"[{n}/{len(cases)}] {cid}  {target.stat().st_size / 1e6:.1f} MB  {time.time() - t0:.0f}s", flush=True)

# cases.js lists every case built so far (not only this run), so building more later just adds to the menu
listing = []
for f in sorted((out / "data").glob("*.js")):
    cid = f.stem
    m = rows.get(cid, {}).get("metrics", {})
    listing.append({"case": cid, "dice": (m.get("T1") or {}).get("argmax_dice")})
(out / "cases.js").write_text("window.CASES = " + json.dumps(listing) + ";\nwindow.CACHE = " +
                              json.dumps({"cache": str(cache), "calibrator": index.get("calibrator")}) + ";\n")
(out / "index.html").write_text((Path(__file__).resolve().parent / "viewer3d.html").read_text())
if not (out / "plotly.min.js").exists():  # saved once so the viewer works offline; the page falls back to the CDN
    try:
        import urllib.request
        urllib.request.urlretrieve(PLOTLY_URL, out / "plotly.min.js.tmp")
        (out / "plotly.min.js.tmp").rename(out / "plotly.min.js")
    except Exception as e:  # noqa: BLE001
        print(f"could not download plotly.js ({e}); the viewer will load it from the CDN instead")
print(f"wrote {out / 'index.html'}  ({len(listing)} cases available)")
if a.open:
    import webbrowser
    webbrowser.open((out / "index.html").resolve().as_uri())
