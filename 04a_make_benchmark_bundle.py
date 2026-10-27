"""
Phase 4 - Step A: Build a small "benchmark bundle" to copy into each slice VM
-----------------------------------------------------------------------------
WHY: each VM only needs the trained models and a SMALL sample of traffic, not
the 1+ GB of prepared data. This script creates a folder `benchmark_bundle/`
(about 30 MB) containing:

    benchmark_bundle/
        sample.csv            20,000 rows (10,000 normal + 10,000 attack),
                              randomly drawn from prepared_data/test.csv
        models/               the model variants we benchmark (Phase 3 + Phase 3b)
        scaler.pkl            preprocessing (needed by every model)
        thresholds.csv        detection threshold of each variant (from 03)
        04b_slice_benchmark.py  the script you run INSIDE each VM

RUN ON YOUR MAIN COMPUTER (after 01, 02, 03 have been run):
    python 04a_make_benchmark_bundle.py
Then copy the whole `benchmark_bundle` folder into each VM.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("prepared_data/test.csv")
MODELS = Path("models")
RESULTS = Path("results/compression_results.csv")
OUT = Path("benchmark_bundle")
N_PER_CLASS = 10_000

# variant name in results table -> model file saved by 03
VARIANTS = {
    "baseline_fp32": "baseline_model.pth",
    "pruned_50+int8": "pruned_50_int8.pth",   # best Phase 3 variant
}


def draw_balanced_sample():
    """Read test.csv in chunks and keep a random N_PER_CLASS rows of each class."""
    rng = np.random.default_rng(42)
    keep = {0: [], 1: []}
    total = {0: 0, 1: 0}
    for chunk in pd.read_csv(DATA, dtype=np.float32, chunksize=200_000):
        for cls in (0, 1):
            part = chunk[chunk["label"] == cls]
            total[cls] += len(part)
            keep[cls].append(part)
        # shrink what we hold so memory stays small
        for cls in (0, 1):
            merged = pd.concat(keep[cls])
            if len(merged) > N_PER_CLASS * 3:
                merged = merged.sample(n=N_PER_CLASS * 3, random_state=int(rng.integers(1e9)))
            keep[cls] = [merged]
    parts = []
    for cls in (0, 1):
        merged = keep[cls][0]
        parts.append(merged.sample(n=min(N_PER_CLASS, len(merged)), random_state=42))
    out = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    return out


def main():
    for need in (DATA, RESULTS, MODELS / "scaler.pkl"):
        if not need.exists():
            raise SystemExit(f"Missing {need}. Run 01, 02 and 03 first.")

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "models").mkdir(parents=True)

    print("Drawing a balanced sample from test.csv ...")
    sample = draw_balanced_sample()
    sample.to_csv(OUT / "sample.csv", index=False)
    print(f"  sample.csv: {len(sample)} rows "
          f"({int((sample['label'] == 0).sum())} normal, {int((sample['label'] == 1).sum())} attack)")

    res = pd.read_csv(RESULTS)
    rows = []
    for variant, fname in VARIANTS.items():
        src = MODELS / fname
        if not src.exists():
            raise SystemExit(f"Missing model file {src}. Run 03_compress.py first.")
        shutil.copy(src, OUT / "models" / fname)
        thr = float(res.loc[res["variant"] == variant, "threshold"].iloc[0])
        rows.append({"variant": variant, "file": fname, "threshold": thr})
    pd.DataFrame(rows).to_csv(OUT / "thresholds.csv", index=False)

    # Phase 3b variants (feature pruning + structured pruning), if 05 was run
    manifest = MODELS / "v2_manifest.csv"
    if manifest.exists():
        for _, r in pd.read_csv(manifest).iterrows():
            shutil.copy(MODELS / r["file"], OUT / "models" / r["file"])
            rows.append({"variant": r["variant"], "file": r["file"],
                         "threshold": float(r["threshold"])})
        pd.DataFrame(rows).to_csv(OUT / "thresholds.csv", index=False)
        print(f"  added {len(pd.read_csv(manifest))} Phase 3b variants from v2_manifest.csv")
    else:
        print("  (no models/v2_manifest.csv found: run 05_feature_and_structured_pruning.py "
              "to also benchmark the pruned-feature models)")

    shutil.copy(MODELS / "scaler.pkl", OUT / "scaler.pkl")
    shutil.copy("04b_slice_benchmark.py", OUT / "04b_slice_benchmark.py")

    size_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"\nBundle ready: '{OUT}/'  ({size_mb:.1f} MB)")
    print("Copy this whole folder into each slice VM, then run 04b_slice_benchmark.py there.")


if __name__ == "__main__":
    main()