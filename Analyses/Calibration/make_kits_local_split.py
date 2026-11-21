"""Patient-level KiTS23 split for training a LocBAM model locally on this Mac.

Writes two MONAI decathlon JSONs into data/kits_json/:
  kits_local_train_fold{F}.json  "training": N_TRAIN cases, "validation": N_MONITOR cases (model selection only)
  kits_fold{F}_heldout.json      "validation": the fold's held-out cases, never seen in training or selection
                                 (this is the --fold-json for Analyses/Calibration/cache_logits.py)

Why a subset: the trainer caches up to 96 training volumes in memory (~0.3 GB each at 1.5 mm),
which would not fit in 48 GB together with the validation cache.
"""
import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WRONG_FILE_IDS = ["00516", "00223", "00088", "00560"]  # same exclusions as the training script

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(REPO / "data" / "kits23" / "dataset"))
ap.add_argument("--out", default=str(REPO / "data" / "kits_json"))
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--n-train", type=int, default=64)
ap.add_argument("--n-monitor", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

cases = []
for d in sorted(Path(a.data).glob("case_*")):
    img, lab = d / "imaging.nii.gz", d / "segmentation.nii.gz"
    if any(b in d.name for b in WRONG_FILE_IDS) or not (img.exists() and lab.exists()):
        continue
    cases.append({"id": d.name, "image": str(img), "label": str(lab)})

ids = [c["id"] for c in cases]
rng = random.Random(a.seed)
rng.shuffle(ids)
heldout = sorted(ids[a.fold::a.k])
pool = [i for i in ids if i not in set(heldout)]  # already shuffled
train, monitor = sorted(pool[: a.n_train]), sorted(pool[a.n_train : a.n_train + a.n_monitor])
by_id = {c["id"]: {"image": c["image"], "label": c["label"]} for c in cases}

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
common = {"fold": a.fold, "k": a.k, "seed": a.seed, "excluded": WRONG_FILE_IDS}
json.dump({**common, "training": [by_id[i] for i in train], "validation": [by_id[i] for i in monitor]},
          open(out / f"kits_local_train_fold{a.fold}.json", "w"), indent=1)
json.dump({**common, "training": [by_id[i] for i in train + monitor], "validation": [by_id[i] for i in heldout]},
          open(out / f"kits_fold{a.fold}_heldout.json", "w"), indent=1)
assert not set(heldout) & set(train + monitor)
print(f"{len(cases)} cases -> train {len(train)}, monitor {len(monitor)}, held-out {len(heldout)} (fold {a.fold})")
print("wrote", out / f"kits_local_train_fold{a.fold}.json")
print("wrote", out / f"kits_fold{a.fold}_heldout.json")
