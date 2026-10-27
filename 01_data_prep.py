"""
Phase 2 - Step 1: Load and prepare N-BaIoT data (or synthetic stand-in data)
-----------------------------------------------------------------------------
WHAT THIS SCRIPT DOES, IN PLAIN WORDS:
 1. Looks for real N-BaIoT CSV files ANYWHERE inside a folder called
    "nbaiot_data/" (any depth of subfolders).
 2. Classifies each CSV file purely by its FILENAME:
      - if the filename contains the word "benign" -> normal traffic
      - anything else (mirai, gafgyt, scan, udp, combo, junk, ack, syn, etc.)
        -> attack traffic
 3. Processes files ONE AT A TIME and writes results straight to disk,
    instead of loading everything into memory at once. The full N-BaIoT
    dataset across all 9 devices is several million rows (several GB once
    loaded as numbers) -- trying to hold it all in memory simultaneously
    is exactly what crashes a normal laptop with a MemoryError. Real
    research on lightweight/constrained-device ML commonly subsamples
    large datasets for exactly this reason, so this is a legitimate and
    defensible approach, not a shortcut.
 4. Caps how many rows are taken from any single CSV file (see
    MAX_ROWS_PER_FILE below). You can raise this number if your machine
    has a lot of RAM, or lower it if you still run into memory errors.
 5. If no real CSVs are found, falls back to generating realistic-looking
    SYNTHETIC data so you can still run the whole pipeline.
 6. Splits data into train/val/test, same convention as the papers.

HOW TO USE WITH YOUR REAL DATA:
 - Put the extracted N-BaIoT device folders inside "nbaiot_data/" next to
   this script, in whatever structure they came in.
 - Just run this script -- no code changes needed.
"""

import csv
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")   # hide harmless pandas performance notices
RNG = np.random.default_rng(42)       # fixed seed = reproducible results
N_FEATURES = 115                       # same as the real N-BaIoT feature count
DATA_DIR = Path("nbaiot_data")
OUT_DIR = Path("prepared_data")
OUT_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# MEMORY-SAFETY SETTING: maximum rows pulled from any ONE csv file.
# Why this matters: some N-BaIoT attack files have 900,000+ rows on their
# own. Loading every row from every one of the ~90 files at once can need
# more RAM than a typical laptop has. Capping per-file rows keeps every
# attack type and every device still represented, just with a manageable
# sample size instead of every single row.
#
# Raise this if your machine has plenty of RAM (16GB+) and you want more
# data; lower it if you still hit memory errors.
# ----------------------------------------------------------------------
MAX_ROWS_PER_FILE = 20000


def find_all_csvs():
    """Find every CSV file anywhere under DATA_DIR, at any nesting depth."""
    if not DATA_DIR.exists():
        return []
    return list(DATA_DIR.glob("**/*.csv"))


def read_and_sample(csv_path, max_rows):
    """
    Read one CSV, keep only numeric columns (defends against stray index/
    text columns causing dtype problems later), cast to float32 (uses half
    the memory of the pandas default float64), and randomly sample down to
    max_rows if the file is bigger than that.
    """
    df = pd.read_csv(csv_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols].astype(np.float32)

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    return df.reset_index(drop=True)


def stream_real_data_to_disk(all_csvs):
    """
    Process every CSV one at a time (never holding more than one file's
    worth of data in memory), and write rows straight into two combined
    files: all_benign.csv and all_attack.csv. This is what avoids the
    MemoryError -- we never build one giant in-memory table of everything.
    """
    benign_out = OUT_DIR / "_all_benign.csv"
    attack_out = OUT_DIR / "_all_attack.csv"

    benign_rows_total, attack_rows_total = 0, 0
    benign_file_count, attack_file_count = 0, 0
    reference_columns = None  # the column set we standardize every file to
    skipped = []

    benign_writer_file = open(benign_out, "w", newline="")
    attack_writer_file = open(attack_out, "w", newline="")
    benign_csv_writer = None
    attack_csv_writer = None

    for csv_path in all_csvs:
        name_lower = csv_path.name.lower()
        is_benign = "benign" in name_lower

        try:
            # benign files are small and scarce, so keep ALL their rows; only cap attacks
            df = read_and_sample(csv_path, None if is_benign else MAX_ROWS_PER_FILE)
        except Exception as e:
            skipped.append((csv_path, str(e)))
            continue

        if df.empty:
            continue

        # Standardize every file to the SAME set of columns (the first
        # file we see sets the reference). This prevents dtype/shape
        # mismatches between files from causing problems.
        if reference_columns is None:
            reference_columns = list(df.columns)

        missing = [c for c in reference_columns if c not in df.columns]
        if missing:
            skipped.append((csv_path, f"missing expected columns: {missing[:3]}..."))
            continue
        df = df[reference_columns]

        if is_benign:
            if benign_csv_writer is None:
                benign_csv_writer = csv.writer(benign_writer_file)
                benign_csv_writer.writerow(reference_columns)
            benign_csv_writer.writerows(df.values.tolist())
            benign_rows_total += len(df)
            benign_file_count += 1
        else:
            if attack_csv_writer is None:
                attack_csv_writer = csv.writer(attack_writer_file)
                attack_csv_writer.writerow(reference_columns)
            attack_csv_writer.writerows(df.values.tolist())
            attack_rows_total += len(df)
            attack_file_count += 1

        print(f"  processed {csv_path.name:40s} "
              f"({'benign' if is_benign else 'attack'}, {len(df)} rows kept)")

    benign_writer_file.close()
    attack_writer_file.close()

    print(f"\n--- Diagnostic summary ---")
    print(f"Benign files found : {benign_file_count}")
    print(f"Attack files found : {attack_file_count}")
    if skipped:
        print(f"Files skipped: {len(skipped)}")
        for path, err in skipped[:5]:
            print(f"  - {path.name}: {err}")
    print(f"Total benign rows kept (after per-file cap): {benign_rows_total}")
    print(f"Total attack rows kept (after per-file cap): {attack_rows_total}")
    print("--------------------------\n")

    if attack_rows_total == 0:
        print("!!! WARNING: No attack rows were kept. Check your 'nbaiot_data'")
        print("!!! folder structure and filenames.\n")

    return benign_rows_total, attack_rows_total


def generate_synthetic_data(n_benign=40000, n_attack=25000):
    """Creates fake-but-realistic traffic data for testing the pipeline."""
    print("No real CSV files found in 'nbaiot_data/' — generating SYNTHETIC "
          "data instead so you can run the full pipeline right now.")

    benign = RNG.normal(loc=0.0, scale=1.0, size=(n_benign, N_FEATURES))
    attack = RNG.normal(loc=2.5, scale=2.2, size=(n_attack, N_FEATURES))

    cols = [f"f{i}" for i in range(N_FEATURES)]
    benign_df = pd.DataFrame(benign, columns=cols).astype(np.float32)
    attack_df = pd.DataFrame(attack, columns=cols).astype(np.float32)

    benign_df.to_csv(OUT_DIR / "_all_benign.csv", index=False)
    attack_df.to_csv(OUT_DIR / "_all_attack.csv", index=False)
    return len(benign_df), len(attack_df)


def split_into_train_val_test():
    """
    Makes three files from _all_benign.csv / _all_attack.csv:

        train.csv -> benign only (70% of benign), for training the autoencoder
        tune.csv  -> 15% of benign + 15% of attack rows (labeled), used ONLY
                     to pick the detection threshold
        test.csv  -> 15% of benign + the other 85% of attack rows, used ONLY
                     for the final, unbiased evaluation

    MEMORY-SAFE: benign data is small and loaded whole (as float32). The
    attack file can be ~2 GB, so it is read in chunks and each row is
    randomly (fixed seed) sent to tune or test while streaming -- the
    whole attack file is never held in memory. (An earlier version loaded
    it all at once and ran out of memory on large datasets.)
    """
    FEATURE_DTYPE = np.float32

    benign_df = pd.read_csv(OUT_DIR / "_all_benign.csv", dtype=FEATURE_DTYPE)
    benign_df["label"] = np.float32(0)
    benign_df = benign_df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(benign_df)
    n_train = int(0.70 * n)
    n_tune = int(0.15 * n)
    train_df = benign_df.iloc[:n_train]
    tune_benign_df = benign_df.iloc[n_train:n_train + n_tune]
    test_benign_df = benign_df.iloc[n_train + n_tune:]

    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    tune_benign_df.to_csv(OUT_DIR / "tune.csv", index=False)
    test_benign_df.to_csv(OUT_DIR / "test.csv", index=False)
    n_train_rows, n_tune_benign, n_test_benign = len(train_df), len(tune_benign_df), len(test_benign_df)
    del benign_df, train_df, tune_benign_df, test_benign_df

    rng = np.random.default_rng(42)
    n_tune_attack, n_test_attack = 0, 0
    for chunk in pd.read_csv(OUT_DIR / "_all_attack.csv", dtype=FEATURE_DTYPE, chunksize=100_000):
        chunk["label"] = np.float32(1)
        to_tune = rng.random(len(chunk)) < 0.15
        chunk[to_tune].to_csv(OUT_DIR / "tune.csv", mode="a", header=False, index=False)
        chunk[~to_tune].to_csv(OUT_DIR / "test.csv", mode="a", header=False, index=False)
        n_tune_attack += int(to_tune.sum())
        n_test_attack += int((~to_tune).sum())

    # Clean up the intermediate combined files -- not needed any more
    (OUT_DIR / "_all_benign.csv").unlink()
    (OUT_DIR / "_all_attack.csv").unlink()

    print("Saved to 'prepared_data/':")
    print(f"  train.csv -> {n_train_rows} rows (benign only, for training the autoencoder)")
    print(f"  tune.csv  -> {n_tune_benign + n_tune_attack} rows "
          f"({n_tune_benign} benign + {n_tune_attack} attack, for picking the threshold)")
    print(f"  test.csv  -> {n_test_benign + n_test_attack} rows "
          f"({n_test_benign} benign + {n_test_attack} attack, for final evaluation)")


def main():
    all_csvs = find_all_csvs()
    if all_csvs:
        print(f"Found {len(all_csvs)} CSV file(s) under 'nbaiot_data/' — using REAL data.")
        print(f"(Capping each file at {MAX_ROWS_PER_FILE} rows to keep memory usage safe.)\n")
        benign_rows, attack_rows = stream_real_data_to_disk(all_csvs)
        if benign_rows == 0:
            print("No usable benign data found — falling back to synthetic data.")
            benign_rows, attack_rows = generate_synthetic_data()
    else:
        benign_rows, attack_rows = generate_synthetic_data()

    if attack_rows == 0:
        raise SystemExit(
            "\nStopping here: there is no attack data to evaluate against.\n"
            "Fix your 'nbaiot_data' folder and re-run this script."
        )

    split_into_train_val_test()


if __name__ == "__main__":
    main()