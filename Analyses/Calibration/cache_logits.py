# Copyright 2025 Donnate Bridget Hooft
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Calibration step 1: run a frozen KiTS23 LocBAM checkpoint and cache its volume logits.

Every later calibration step (patient split, temperature fit, ECE / NLL / Brier,
threshold sweeps, viewer) reads the files this script writes. The model is run
exactly once per case.

What this script does, per out-of-fold validation case of one fold:
  1. the same validation preprocessing as BasicUNet/kits_loc_context_cleaned_files.py
     (get_val_transforms, copied verbatim because it is nested under __main__ there),
  2. sliding_window_inference with overlap and blend mode passed explicitly
     (the training script relied on MONAI 1.3.0 defaults: 0.25, "constant"),
  3. NO argmax: the aggregated logits are kept and written as fp16,
  4. the resampled label, spacing, and run metadata are written next to them,
  5. per-class argmax Dice is reported as a sanity check that the checkpoint loaded
     correctly (it should be close to the validation Dice logged during training).

Output layout:
  <out_dir>/<checkpoint sha256[:12]>/<case_id>.npz
      logits  float16 (4, X, Y, Z)   aggregated, pre-softmax
      label   uint8   (X, Y, Z)      resampled ground truth (1.5 mm, RAS, cropped)
      meta    json string            see case_meta() below
  <out_dir>/<checkpoint sha256[:12]>/run.json    run-level settings + per-case Dice

Example:
  python Analyses/Calibration/cache_logits.py \
      --checkpoint data/checkpoints/cleaned_KITS23_UNet_final_LocBAM_large_0_0.85.pth \
      --fold-json  data/kits_bpr_json/dataset_0_hmcluster_bpr_new.json \
      --patch-size large \
      --remap-prefix /lustre/boost_ai/users/test/stefan.fischer/nnUNet_raw/Dataset012_KiTS23 data/kits_bpr
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

# The LocBAM gates use AdaptiveAvgPool3d, which Apple MPS does not implement; let that one op
# fall back to CPU. Must be set before torch is imported. No effect on CUDA or CPU runs.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
# BasicUNet/ for `from LocBAM.LocBAM_1D import ...`; BasicUNet/LocBAM for `from config import ...`
sys.path.insert(0, str(REPO_ROOT / "BasicUNet"))
sys.path.insert(0, str(REPO_ROOT / "BasicUNet" / "LocBAM"))

import monai  # noqa: E402
from monai.data import load_decathlon_datalist  # noqa: E402
from monai.inferers import sliding_window_inference  # noqa: E402
from monai.transforms import (  # noqa: E402
    AddCoordinateChannelsd,
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

from LocBAM.LocBAM_1D import BasicUnetLocBAMs2  # noqa: E402
from kits_loc_context_cleaned_files import (  # noqa: E402  (top-level classes only; its __main__ is guarded)
    BodyPartRegressorScoreTransform,
    ConcatChannelsd,
    OptimizeCoordinateChannels,
    ReconstructCoordinateChannels,
    SplitChannelsd,
)

NUM_CLASSES = 4
# KiTS23 convention. The notebook legend says "Kidney / Mass / Tumor": confirm before reporting class-wise results.
LABEL_MAP = {0: "background", 1: "kidney", 2: "tumor", 3: "cyst"}
LABEL_MAP_CONFIRMED = False
# Cases the training script drops as corrupted.
WRONG_FILE_IDS = ["00516", "00223", "00088", "00560"]
PATCH_SIZES = {"small": ((32, 32, 32), 128), "large": ((128, 128, 128), 2)}
# Bump whenever get_val_transforms() below changes, so stale caches are never mixed with new ones.
PREPROCESSING_VERSION = "kits_locbam_val_v1"


def get_val_transforms():
    """Copy of get_val_transforms() in BasicUNet/kits_loc_context_cleaned_files.py (L419-471).

    Differences, none of which change the values the model sees:
      - RandShiftIntensityd(prob=0.0, offsets=0.0) is dropped (it is a no-op).
      - The final EnsureTyped that moved tensors to the GPU is dropped; the loop moves them.
    """
    dtype = [np.float16, np.int8]  # same quantization as training
    return Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
            SplitChannelsd(keys=["image"], output_keys=("image_ct", "image_bpr")),
            ScaleIntensityRanged(keys=["image_ct"], a_min=-58, a_max=302, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["image_bpr"], a_min=0, a_max=100, b_min=-0.5, b_max=0.5, clip=False),
            ConcatChannelsd(keys=["image_ct", "image_bpr"], output_key="image"),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=False),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            AddCoordinateChannelsd(keys=["image"], spatial_dims=[0, 1]),
            OptimizeCoordinateChannels(keys=["image"]),
            BodyPartRegressorScoreTransform(keys=["image"]),
            ReconstructCoordinateChannels(keys=["image"]),
            EnsureTyped(keys=["image", "label"], track_meta=False, dtype=dtype),
        ]
    )


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_model(checkpoint, device):
    """Build BasicUnetLocBAMs2 as trained and load it strictly.

    The training script passes hanet_params=None, which the current constructor does not
    accept (it takes LocBAM_params). Defaults are used here; strict=True makes any
    architecture mismatch with the checkpoint fail loudly instead of silently.
    """
    model = BasicUnetLocBAMs2(
        spatial_dims=3,
        in_channels=1,
        out_channels=NUM_CLASSES,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.1,
    )
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()  # eval(): dropout inside the LocBAM blocks is off


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def case_id_of(path):
    m = re.search(r"(\d{5})", Path(path).name)
    return f"case_{m.group(1)}" if m else Path(path).name.split(".")[0]


def remap(path, prefixes):
    for old, new in prefixes:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def load_cases(fold_json, base_dir, prefixes, only):
    files = load_decathlon_datalist(fold_json, True, "validation", base_dir=base_dir)
    cases = []
    for item in files:
        if any(bad in item["image"] for bad in WRONG_FILE_IDS):
            continue
        item = {k: remap(v, prefixes) for k, v in item.items()}
        cid = case_id_of(item["image"])
        if only and cid not in only:
            continue
        cases.append((cid, item))
    return cases


@torch.no_grad()
def infer_logits(model, image, patch_size, sw_batch_size, overlap, mode, device, amp):
    """image: (1, 4, X, Y, Z) on CPU. Returns aggregated logits (1, 4, X, Y, Z) float32 on CPU."""
    x = image.to(device=device, dtype=torch.float32)
    use_amp = amp and device.type == "cuda"
    with torch.autocast(device_type="cuda", enabled=use_amp):
        logits = sliding_window_inference(
            x, patch_size, sw_batch_size, model,
            overlap=overlap, mode=mode,
            sw_device=device, device=torch.device("cpu"),
        )
    return logits.float()


def per_class_dice(pred, label):
    out = {}
    for c in range(1, NUM_CLASSES):
        p, g = pred == c, label == c
        denom = int(p.sum()) + int(g.sum())
        out[LABEL_MAP[c]] = None if denom == 0 else 2.0 * int((p & g).sum()) / denom
    return out


def case_meta(cid, item, label_shape, run):
    return {
        "case_id": cid,
        "image_path": item["image"],
        "label_path": item["label"],
        "shape_xyz": list(label_shape),
        "spacing_mm": [1.5, 1.5, 1.5],
        "orientation": "RAS",
        "note": "cropped to foreground of the 2-channel input, then resampled; not in original image space",
        **{k: run[k] for k in ("checkpoint_sha256", "preprocessing_version", "patch_size", "overlap", "mode", "amp")},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="trained BasicUnetLocBAMs2 KiTS23 .pth")
    ap.add_argument("--fold-json", required=True, help="dataset_{fold}_hmcluster_bpr_new.json used to train that checkpoint")
    ap.add_argument("--patch-size", choices=PATCH_SIZES, required=True, help="the -ps the checkpoint was trained with")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "cache"))
    ap.add_argument("--base-dir", default=None, help="base for relative paths in the fold JSON (default: its folder)")
    ap.add_argument("--remap-prefix", nargs=2, action="append", default=[], metavar=("OLD", "NEW"),
                    help="rewrite absolute cluster paths in the JSON; repeatable")
    ap.add_argument("--cases", nargs="*", default=None, help="only these case ids, e.g. case_00012")
    ap.add_argument("--limit", type=int, default=None, help="stop after N cases (for a first test)")
    ap.add_argument("--overlap", type=float, default=0.25, help="MONAI 1.3.0 default, used implicitly in training")
    ap.add_argument("--mode", choices=["constant", "gaussian"], default="constant",
                    help="'constant' keeps blending linear, so z/T commutes with aggregation")
    ap.add_argument("--sw-batch-size", type=int, default=None, help="default: the training effective batch size")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast on CUDA, as in training validation")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--overwrite", action="store_true", help="recompute cases already in the cache")
    args = ap.parse_args()

    device = resolve_device(args.device)
    patch_size, train_batch = PATCH_SIZES[args.patch_size]
    sw_batch = args.sw_batch_size or train_batch

    ckpt_hash = sha256_file(args.checkpoint)
    out_dir = Path(args.out_dir) / ckpt_hash[:12]
    out_dir.mkdir(parents=True, exist_ok=True)

    run = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": ckpt_hash,
        "fold_json": str(Path(args.fold_json).resolve()),
        "fold_json_sha256": sha256_file(args.fold_json),
        "preprocessing_version": PREPROCESSING_VERSION,
        "patch_size": list(patch_size),
        "sw_batch_size": sw_batch,
        "overlap": args.overlap,
        "mode": args.mode,
        "amp": bool(args.amp and device.type == "cuda"),
        "device": str(device),
        "label_map": LABEL_MAP,
        "label_map_confirmed": LABEL_MAP_CONFIRMED,
        "monai": monai.__version__,
        "torch": torch.__version__,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cases": {},
    }
    run_path = out_dir / "run.json"
    if run_path.exists():
        prev = json.loads(run_path.read_text())
        for key in ("fold_json_sha256", "preprocessing_version", "patch_size", "overlap", "mode", "amp"):
            if prev.get(key) != run[key] and not args.overwrite:
                sys.exit(f"{run_path} was written with {key}={prev.get(key)!r}, now {run[key]!r}. "
                         "Use another --out-dir or --overwrite.")
        run["cases"] = prev.get("cases", {})

    model = load_model(args.checkpoint, device)
    transforms = get_val_transforms()
    cases = load_cases(args.fold_json, args.base_dir, args.remap_prefix, set(args.cases or []))
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} validation cases · device {device} · patch {patch_size} · cache {out_dir}", flush=True)

    for n, (cid, item) in enumerate(cases, 1):
        target = out_dir / f"{cid}.npz"
        if target.exists() and not args.overwrite:
            print(f"[{n}/{len(cases)}] {cid}: cached, skipping", flush=True)
            continue
        t0 = time.time()
        data = transforms(dict(item))
        image = data["image"].unsqueeze(0)          # (1, 4, X, Y, Z) fp16
        label = data["label"][0].numpy().astype(np.uint8)  # (X, Y, Z)

        logits = infer_logits(model, image, patch_size, sw_batch, args.overlap, args.mode, device, args.amp)[0]
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"{cid}: non-finite logits")
        if logits.abs().max() > 6.0e4:
            raise RuntimeError(f"{cid}: logits exceed the fp16 range; store fp32 instead")

        pred = logits.argmax(0).numpy().astype(np.uint8)
        dice = per_class_dice(pred, label)
        meta = case_meta(cid, item, label.shape, run)

        tmp = target.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, logits=logits.numpy().astype(np.float16), label=label, meta=json.dumps(meta))
        os.replace(tmp, target)  # never leave a half-written case in the cache

        run["cases"][cid] = {"argmax_dice": dice, "shape_xyz": list(label.shape), "seconds": round(time.time() - t0, 1)}
        run_path.write_text(json.dumps(run, indent=2))
        dice_txt = "  ".join(f"{k} {'—' if v is None else f'{v:.3f}'}" for k, v in dice.items())
        print(f"[{n}/{len(cases)}] {cid}: {tuple(label.shape)}  {dice_txt}  ({time.time() - t0:.0f}s)", flush=True)

    run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_path.write_text(json.dumps(run, indent=2))
    done = [c["argmax_dice"] for c in run["cases"].values()]
    for cls in ("kidney", "tumor", "cyst"):
        vals = [d[cls] for d in done if d.get(cls) is not None]
        if vals:
            print(f"mean argmax Dice {cls}: {np.mean(vals):.3f} over {len(vals)} cases")
    print("Compare with the validation Dice logged for this checkpoint; a large gap means a loading or preprocessing mismatch.")


if __name__ == "__main__":
    main()
