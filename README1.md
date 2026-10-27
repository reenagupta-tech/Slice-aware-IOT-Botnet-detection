Get-Content .gitignore# Phase 2 — Baseline IoT Botnet Detection Model

This folder has everything you need to build and test your baseline autoencoder.

## Files

- `01_data_prep.py` — loads your N-BaIoT data (or makes realistic synthetic
  data if you haven't downloaded the real dataset yet) and splits it into
  train/val/test CSVs inside `prepared_data/`.
- `02_train_baseline.py` — builds the autoencoder, trains it on benign
  traffic only, picks a detection threshold, evaluates it, and saves the
  trained model into `models/`.

## How to run it

```bash
cd phase2
python3 01_data_prep.py
python3 02_train_baseline.py
```

That's it — two commands. You'll see training progress printed epoch by
epoch, then a final block of results like:

```
===== TEST SET RESULTS =====
Accuracy : 0.97xx
Precision: 0.97xx
Recall   : 0.96xx
F1 score : 0.97xx
=============================
```

## Switching from synthetic data to the real N-BaIoT dataset

1. Download the dataset from the UCI ML Repository or IEEE DataPort link
   in your literature review.
2. Unzip it so you get a folder per device, each containing a
   `benign_traffic.csv` and attack subfolders (`mirai_attacks/`,
   `gafgyt_attacks/`, etc. — these names may vary slightly by download
   source, check what you actually got).
3. Put that whole structure inside a folder named `nbaiot_data/` next to
   these scripts, e.g.:
   ```
   phase2/
     nbaiot_data/
       Danmini_Doorbell/
         benign_traffic.csv
         mirai_attacks/
           scan.csv
           ...
   ```
4. Re-run `python3 01_data_prep.py` — it automatically detects the real
   files and uses them instead of synthetic data. No code changes needed.

## What's saved in `models/` after training

- `baseline_model.pth` — the trained PyTorch model weights (Phase 3 needs this)
- `scaler.pkl` — the Min-Max scaler fitted on training data (needed to
  preprocess any new data the same way before feeding it to the model)
- `threshold.pkl` — the detection threshold
- `metrics.pkl` — accuracy/precision/recall/F1 from this run

## What's next (Phase 3)

Once this runs successfully and you have reasonable accuracy numbers,
we'll write the pruning + quantization scripts that load `baseline_model.pth`
and produce compressed versions of it.
