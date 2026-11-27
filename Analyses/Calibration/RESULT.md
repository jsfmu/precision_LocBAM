# Calibrating the confidence of the LocBAM kidney segmentation model (KiTS23)

## Summary
The segmentation model (BasicUnetLocBAMs2, trained on KiTS23) gives voxel probabilities that are **overconfident**: in the organ region, voxels it reports at ~85% confidence are correct ~74% of the time. We fitted a post-hoc calibrator that rescales the model's outputs without retraining and without changing the segmentation. We froze it, then tested it once on **316 KiTS23 cases that had never been used**.

- **All seven pre-registered hypotheses were confirmed.** Whole-volume NLL fell from 0.01247 to 0.01145 (change −0.00103, 95% CI [−0.00154, −0.00067]). Top-label calibration error in the organ region fell from 0.183 to 0.118. Dice is unchanged by construction.
- **The originally planned single global temperature did not help** and made the organ region slightly worse.
- **Exploratory results (49 patients):**
  - On the voxels a reviewer would see, a confidence cut-off becomes honest: the error rate you are promised is the error rate you get (gap 0.032 → 0.001).
  - The model's overestimate of its own tumor Dice disappears (+0.15 → +0.01).
- **Main limitation:** the model's worst errors are organ voxels it labels background with near-certainty, far from its prediction. No calibration can flag these.

## 1. Question
Can post-hoc calibration make the model's probabilities honest, without retraining and without changing the segmentation? And is the resulting confidence useful, for flagging voxels for review and for estimating per-patient segmentation quality?

## 2. Setup

**Model.** Frozen checkpoint `kits_locbam_local_fold0/best_metric_model.pth` (sha256 `851c43ca…0308d1`). Classes: background, kidney, tumor, cyst. Sliding-window inference with patch 32³, overlap 0.25 and constant blending, the same settings as in training. The raw outputs (logits) were cached once per case, so every analysis uses identical model outputs.

**Data (489 KiTS23 cases).**

| Use | Cases |
|---|---|
| Model training | 64 |
| Model selection during training | 8 |
| Calibrator fitting and selection ("calibration patients") | 48 |
| Exploratory test ("test patients") | 49 |
| Confirmation, untouched until the final test | 316 |
| Excluded (corrupted files) | 4 |

The split is made by `make_kits_local_split.py` (seed 0, fold 0) and is patient-level throughout.

**Calibrators compared.** All are applied to the cached logits.
- **Raw:** temperature T = 1, i.e. no change.
- **Global temperature:** one T for the whole volume, fitted on the calibration patients (T = 0.947). This was the original plan.
- **Per-predicted-class temperature (selected):** one temperature for each class the model predicts at a voxel, applied only within 8 voxels of the predicted organ, with T = 1 elsewhere. Fitted values: background 1.49, kidney 1.28, tumor 2.51, cyst 1.35. Because one temperature applies to all class scores at a given voxel, the predicted class never changes, so Dice is identical to the raw model.
- **Per-predicted-class isotonic (exploratory only):** a flexible, monotone map per predicted class and class, applied within 4 voxels. It can change the predicted class.

**How the calibrator was selected.** 5-fold patient-level cross-validation on the 48 calibration patients only. Method and region size (0–16 voxels) were chosen by held-out whole-volume NLL. The winner was then fitted on all 48.

**Metrics.**
- **NLL:** negative log-likelihood of the true class (lower is better). It is the primary metric.
- **Brier score.**
- **ECE:** expected calibration error, 15 bins, for the top label and for each class.
- **Argmax Dice.**
- **Two regions:**
  - **Whole volume.**
  - **Organ region:** voxels where the ground truth or the raw prediction is not background. This is where calibration matters, since most of the volume is trivially certain background. It uses labels, so it is for evaluation only.
- **Uncertainty:** 95% paired patient-bootstrap intervals (1000 resamples, seed 0).

## 3. Confirmation on 316 unused cases (main result)

**Procedure.** The calibrator, checkpoint and case list were frozen with their sha256 hashes in `CONFIRMATION_PLAN.md`, committed (57c308d) **before** the confirmation outputs were computed. The evaluation was run once. Nothing was fitted, chosen or tuned on these cases, and no case was dropped.

**Results.** Change vs the raw model, pooled over 316 patients.

| Hypothesis | Raw | Calibrated | Change [95% CI] | Result |
|---|---|---|---|---|
| H1 Whole-volume NLL (primary) | 0.01247 | 0.01145 | −0.00103 [−0.00154, −0.00067] | Confirmed |
| H2 Organ-region NLL | 1.013 | 0.869 | −0.144 [−0.190, −0.112] | Confirmed |
| H3 Whole-volume Brier | 0.00485 | 0.00471 | −0.00014 [−0.00019, −0.00009] | Confirmed |
| H4 Organ-region top-label ECE | 0.183 | 0.118 | −0.065 [−0.069, −0.060] | Confirmed |
| H5 Organ-region tumor ECE | 0.114 | 0.095 | −0.019 [−0.036, −0.004] | Confirmed |
| H6 Organ-region kidney ECE | 0.073 | 0.047 | −0.026 [−0.033, −0.019] | Confirmed |
| H7 Dice unchanged (kidney 0.888, tumor 0.562, cyst 0.106) | | | 0 exactly | Confirmed |

**The original global temperature (T = 0.947).**
- Whole-volume NLL: no change, −0.00001 [−0.00011, +0.00012].
- Organ-region NLL: worse, +0.041 [+0.033, +0.052].
- This matches what the exploratory data predicted.

**Side effect.** Whole-volume top-label ECE rose very slightly, 0.00099 → 0.00106 (+0.00007 [+0.00002, +0.00019]), and whole-volume kidney ECE rose 0.00077 → 0.00100. These values are tiny, because the whole volume is dominated by certain background. The organ-region improvements are 100–1000× larger.

Figures: `results/confirmation/reliability_fg.png` (reliability diagrams) and `results/confirmation/threshold_sweeps.png`.

## 4. Exploratory results (49 test patients)

These were analysed after looking at the data and are **not confirmed**. Read them as hypotheses for future work.

### 4.1 Isotonic vs per-class temperature
- **Whole-volume NLL:** isotonic −0.00156 [−0.00272, −0.00077], per-class temperature −0.00142 [−0.00241, −0.00074]. The paired difference, −0.00014 [−0.00043, +0.00011], is not significant.
- **Organ-region top-label ECE:** raw 0.239, per-class temperature 0.180, isotonic 0.173.
- **Dice:** isotonic can change the predicted class, and it raised Dice:
  - kidney +0.003 [+0.002, +0.004];
  - tumor +0.008 [+0.003, +0.011];
  - cyst +0.066 [+0.028, +0.104].
  The cyst result rests on few voxels: only ~18,000 calibration voxels were predicted cyst.
- **Remaining miscalibration:** top-label confidence near 0.9 is still overconfident (observed ≈ 0.77).
- **Conclusion:** isotonic is a candidate worth confirming. The confirmed calibrator stays the per-class temperature.

### 4.2 Is the confidence useful?
Results are from `confidence_usefulness.py`.

**Flag-for-review on the voxels a reviewer would see.** These are the predicted organ grown by 8 voxels; this needs no labels. Voxels below a confidence cut-off are flagged and the rest are kept.

| | Ranking: AURC (lower is better) | Honesty gap (lower is better) |
|---|---|---|
| Raw | 0.0282 [0.022, 0.035] | 0.0324 [0.023, 0.043] |
| Per-class temperature | 0.0269 [0.021, 0.034] | **0.0013** [0.001, 0.009] |
| Isotonic | 0.0273 [0.021, 0.035] | 0.0083 [0.002, 0.018] |

- **Honesty gap:** the mean difference between the error rate promised on the kept voxels (mean of 1 − confidence) and the actual error rate, over cut-offs keeping 50–100% of voxels.
  - Raw: the model delivers about twice the error rate it promises.
  - Per-class temperature: the promise is kept, and the curve lies on the diagonal.
  - This comparison favours the per-class temperature, which was fitted on exactly this 8-voxel region.
- **Ranking (AURC)** is essentially unchanged, as expected for a rescaling.
  - Error is ~12% when all voxels are kept and ~1–1.5% when the most confident half is kept.
  - The raw model puts ~25% of these voxels at confidence ≈ 1.0, so it cannot be thresholded below that.

**Per-patient quality estimate.**
- **Estimate:** expected Dice from the model's own probabilities, 2·Σ_{predicted k} p_k / (#predicted k + Σ p_k), compared with the actual Dice.

| | Tumor MAE | Tumor bias | Tumor Spearman ρ | Kidney MAE | Kidney bias |
|---|---|---|---|---|---|
| Raw | 0.226 | **+0.151** | 0.64 | **0.035** | −0.025 |
| Per-class temperature | 0.193 | **+0.010** | 0.64 | 0.068 | −0.066 |
| Isotonic | 0.187 | +0.032 | 0.67 | 0.057 | −0.054 |

- **Tumor:** the raw model overestimates its own tumor Dice by 0.15 on average, and calibration removes this bias.
- **Ranking patients** is moderate whichever calibrator is used (ρ ≈ 0.65), so the estimate cannot reliably pick out the worst cases.
- **Kidney:** calibration makes the estimate pessimistic.

### 4.3 The dominant failure: confident misses
- **Organ region (uses labels):** ranking got worse after calibration. AURC was 0.231 raw, 0.311 per-class temperature, 0.346 isotonic (wide CIs). In all three, keeping only the most confident voxels raised the error rate.
- **Diagnosis, first 20 test patients:**
  - 4.0% of organ-region voxels are wrong with confidence ≥ 0.99;
  - 3.5 of those percentage points are organ voxels predicted background, i.e. missed organ;
  - 3.2 points lie outside the calibrated region;
  - confident confusions between organ classes are rare (tumor called kidney: 0.1 points).
- **Why ranking got worse:** these missed voxels keep their raw, near-certain confidence while calibration lowers confidence near the prediction, so they rise to the top of the ranking.
- **Implication:** this is a limitation of the model, not of the calibration.

## 5. Limitations
- **Confident misses.** Calibration cannot flag organ, especially tumor, that the model labels background with near-certainty far from its prediction. Catching it would need a second source of uncertainty, such as disagreement under flipped inputs, dropout at inference or an ensemble. This is untested.
- **Kidney quality estimate** got worse with calibration.
- **Whole-volume ECE** rose very slightly (§3).
- **One model, one dataset, one fold.** External validity is untested.
- **Ground-truth-based region.** The organ-region metrics use labels to define the region. At inference the calibrator uses only the prediction.
- **No unused KiTS23 cases remain.** Any further KiTS23 result, including confirming isotonic, is exploratory. Confirmation needs a new dataset.

## 6. Reproduce
All scripts are in `Analyses/Calibration/`. Data and caches are under `data/` (not in git). Recorded results are in `Analyses/Calibration/results/`.

```bash
# 1. split (seed 0, fold 0)
python Analyses/Calibration/make_kits_local_split.py
# 2. cache raw logits (exploratory held-out fold; repeat with kits_fold0_unused.json and --out-dir data/cache_confirm for confirmation)
python Analyses/Calibration/cache_logits.py --checkpoint data/checkpoints/runs/kits_locbam_local_fold0/best_metric_model.pth \
  --fold-json data/kits_json/kits_fold0_heldout.json --patch-size small --preprocessing coords
# 3. select by 5-fold CV on the calibration patients, then fit on all of them
python Analyses/Calibration/select_calibrator.py --cache data/cache/851c43ca5339 --methods temperature predclass isotonic --dilations 0 1 2 4 8 12 16
python Analyses/Calibration/fit_temperature.py --cache data/cache/851c43ca5339 --region pred --dilate 8 --method predclass
# 4. evaluate (confirmation: run once with the frozen calibrator files)
python Analyses/Calibration/evaluate_calibration.py --cache data/cache_confirm/851c43ca5339 \
  --calibrators temperature.json calib_pred8_predclass.json --plots
# 5. usefulness (exploratory)
python Analyses/Calibration/confidence_usefulness.py --cache data/cache/851c43ca5339 --population pred8
# 6. viewer
python Analyses/Calibration/build_display_cache.py --cache data/cache/851c43ca5339   # then open calibration_viewer.ipynb
```

| Result | File |
|---|---|
| Pre-registration | `Analyses/Calibration/CONFIRMATION_PLAN.md` (commit 57c308d) |
| Confirmation | `results/confirmation/metrics.json`, `reliability_fg.png`, `threshold_sweeps.png` |
| Isotonic (exploratory) | `results/exploratory_isotonic/` |
| Usefulness (exploratory) | `results/exploratory_usefulness/usefulness.json`, `usefulness.png` (organ region), `usefulness_pred8.json`, `usefulness_pred8.png` (reviewer region) |
