"""
Phase 3: Compress the baseline autoencoder (pruning + INT8 quantization)
-------------------------------------------------------------------------
WHAT THIS SCRIPT DOES, IN PLAIN WORDS:
  Your baseline model works well but a tiny IoT device or a small VM slice
  may not have much memory or CPU. Compression makes the model smaller and
  faster. We try two standard techniques, plus both together:

  1. PRUNING   - set the least important weights (the ones closest to zero)
                 to exactly zero. We try 30%, 50%, 70% and 90% of weights.
                 After pruning we fine-tune briefly so the model recovers.
  2. QUANTIZATION (INT8) - store weights as 8-bit integers instead of
                 32-bit decimals. That is roughly 4x smaller for the
                 weights.
  3. BOTH      - prune 50% then quantize.

  For every variant we measure:
     - size            (how much memory the model needs)
     - latency         (time to score ONE traffic sample, on CPU)
     - balanced accuracy, recall, specificity, ROC-AUC (plus accuracy/F1)
       on the same test set

  IMPORTANT: each compressed model gets its OWN detection threshold,
  re-tuned on tune.csv (same balanced-accuracy sweep as Phase 2). A compressed model's
  reconstruction errors shift a little, so reusing the old threshold
  would give unfair, worse numbers.

  Results are saved to results/compression_results.csv - this table is
  one of your main results for the report.

RUN:  python 03_compress.py      (needs Phase 2 to have been run first)
"""

import copy
import io
import pickle
import time
import warnings

# PyTorch prints harmless "deprecated" notices for its quantization tools.
# They do not affect results, so we hide them to keep the output readable.
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

torch.set_num_threads(1)  # single thread = fair, repeatable latency numbers
torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = Path("prepared_data")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

PRUNE_LEVELS = [0.30, 0.50, 0.70, 0.90]
FINETUNE_EPOCHS = 10
LATENCY_SAMPLES = 2000


# ---------------------------------------------------------------
# Same architecture as Phase 2 (must match to load the weights)
# ---------------------------------------------------------------
def signed_log(X):
    """Same sign-preserving log transform used in 02_train_baseline.py
    (applied before the Min-Max scaler). Must match exactly."""
    X = np.asarray(X, dtype=np.float32)
    return np.sign(X) * np.log1p(np.abs(X))


class Autoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 24), nn.ReLU(),
            nn.Linear(24, 6), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(6, 24), nn.ReLU(),
            nn.Linear(24, 64), nn.ReLU(),
            nn.Linear(64, n_features), nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------
# Data + helpers
# ---------------------------------------------------------------
def load_everything():
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    train_df = pd.read_csv(DATA_DIR / "train.csv", dtype=np.float32)
    tune_df = pd.read_csv(DATA_DIR / "tune.csv", dtype=np.float32)
    test_df = pd.read_csv(DATA_DIR / "test.csv", dtype=np.float32)
    cols = [c for c in train_df.columns if c != "label"]

    def prep(df):
        return scaler.transform(signed_log(df[cols].values)).astype(np.float32)

    X_train = prep(train_df)
    X_tune, y_tune = prep(tune_df), tune_df["label"].values.astype(np.int64)
    X_test, y_test = prep(test_df), test_df["label"].values.astype(np.int64)

    n_features = X_train.shape[1]
    model = Autoencoder(n_features)
    model.load_state_dict(torch.load(MODEL_DIR / "baseline_model.pth", map_location="cpu"))
    model.eval()
    return model, X_train, X_tune, y_tune, X_test, y_test


def reconstruction_error(model, X, batch=20000):
    """Per-row mean squared error. Works for float and quantized models."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch])
            out.append(torch.mean((model(xb) - xb) ** 2, dim=1).numpy())
    return np.concatenate(out)


def balanced_acc(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    return 0.5 * (tp / max(np.sum(y_true == 1), 1) + tn / max(np.sum(y_true == 0), 1))


def tune_threshold(errors, y):
    """
    Pick the threshold that maximizes BALANCED accuracy (not F1/accuracy).
    The data is ~98% attack, so plain F1/accuracy is maximized by flagging
    everything as an attack, which detects nothing useful. Candidates are
    quantiles of the errors because the errors are very skewed.
    """
    candidates = np.unique(np.quantile(errors, np.linspace(0.0, 1.0, 1001)))
    best_t, best = candidates[0], -1.0
    for t in candidates:
        sc = balanced_acc(y, (errors > t).astype(int))
        if sc > best:
            best_t, best = t, sc
    return best_t


def count_params(model):
    """Return (total weights+biases, non-zero weights+biases)."""
    total, nonzero = 0, 0
    for p in model.parameters():
        total += p.numel()
        nonzero += int((p != 0).sum().item())
    return total, nonzero


def file_size_kb(model):
    """Real size of the saved model file, in KB."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1024


def sparse_size_kb(model):
    """
    Pruning puts zeros inside a normal dense file, so the file does not
    shrink by itself. If the zeros were stored in a sparse format, you
    would only need to store the non-zero values (4 bytes each, ignoring
    the small index overhead). This gives that theoretical size.
    """
    _, nonzero = count_params(model)
    return nonzero * 4 / 1024


def measure_latency_ms(model, X):
    """Average time to score ONE sample (batch size 1), in milliseconds."""
    model.eval()
    n = min(LATENCY_SAMPLES, len(X))
    samples = [torch.from_numpy(X[i:i + 1]) for i in range(n)]
    with torch.no_grad():
        for s in samples[:100]:  # warm-up
            model(s)
        start = time.perf_counter()
        for s in samples:
            model(s)
        elapsed = time.perf_counter() - start
    return elapsed / n * 1000


def pruned_int8_size_kb(pruned_fp32_model):
    """
    Theoretical size of a pruned + INT8 model: each non-zero weight takes
    1 byte (INT8), each bias stays 4 bytes. Zeros cost nothing if stored
    in a sparse format.
    """
    weight_bytes, bias_bytes = 0, 0
    for m in linear_layers(pruned_fp32_model):
        weight_bytes += int((m.weight != 0).sum().item())
        bias_bytes += m.bias.numel() * 4
    return (weight_bytes + bias_bytes) / 1024


def evaluate_variant(name, model, X_tune, y_tune, X_test, y_test, kind,
                     size_override_kb=None):
    tune_err = reconstruction_error(model, X_tune)
    threshold = tune_threshold(tune_err, y_tune)

    test_err = reconstruction_error(model, X_test)
    pred = (test_err > threshold).astype(int)

    total, nonzero = count_params(model)
    is_quant = kind in ("quantized", "pruned+quantized")

    rec = recall_score(y_test, pred, zero_division=0)
    tn = np.sum((pred == 0) & (y_test == 0))
    fp = np.sum((pred == 1) & (y_test == 0))
    spec = tn / max(tn + fp, 1)

    row = {
        "variant": name,
        "balanced_accuracy": 0.5 * (rec + spec),
        "recall": rec,
        "specificity": spec,
        "roc_auc": roc_auc_score(y_test, test_err),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "latency_ms_per_sample": measure_latency_ms(model, X_test),
        "dense_file_kb": file_size_kb(model),
        # for quantized models the dense file is already the compact one
        "effective_size_kb": (size_override_kb if size_override_kb is not None
                              else file_size_kb(model) if is_quant
                              else sparse_size_kb(model)),
        "weights_total": total if not is_quant else np.nan,
        "weights_nonzero": nonzero if not is_quant else np.nan,
        "threshold": threshold,
    }
    return row


# ---------------------------------------------------------------
# Compression techniques
# ---------------------------------------------------------------
def linear_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def prune_model(base_model, amount, X_train):
    """
    Global L1 unstructured pruning: across ALL layers together, remove the
    `amount` fraction of weights with the smallest absolute value. Then
    fine-tune for a few epochs (masks stay active, so pruned weights stay
    zero) and make the pruning permanent.
    """
    model = copy.deepcopy(base_model)
    params = [(m, "weight") for m in linear_layers(model)]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)

    # brief fine-tune so the surviving weights compensate
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_train)),
        batch_size=256, shuffle=True)
    for _ in range(FINETUNE_EPOCHS):
        for (b,) in loader:
            opt.zero_grad()
            loss_fn(model(b), b).backward()
            opt.step()

    for m, name in params:
        prune.remove(m, name)  # bake the zeros in permanently
    model.eval()
    return model


def quantize_model(fp32_model):
    """Post-training dynamic INT8 quantization of all Linear layers."""
    m = copy.deepcopy(fp32_model).eval()
    return torch.ao.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)


# ---------------------------------------------------------------
def main():
    print("Loading baseline model and data...")
    base, X_train, X_tune, y_tune, X_test, y_test = load_everything()
    print(f"  {X_train.shape[1]} features | {len(X_train)} train | "
          f"{len(X_tune)} tune | {len(X_test)} test rows\n")

    rows = []

    print("Evaluating baseline (FP32, no compression)...")
    rows.append(evaluate_variant("baseline_fp32", base, X_tune, y_tune, X_test, y_test, "baseline"))

    pruned_models = {}
    for amount in PRUNE_LEVELS:
        name = f"pruned_{int(amount * 100)}"
        print(f"Pruning {int(amount * 100)}% + fine-tuning...")
        pm = prune_model(base, amount, X_train)
        pruned_models[amount] = pm
        rows.append(evaluate_variant(name, pm, X_tune, y_tune, X_test, y_test, "pruned"))
        torch.save(pm.state_dict(), MODEL_DIR / f"{name}.pth")

    print("Quantizing to INT8...")
    qm = quantize_model(base)
    rows.append(evaluate_variant("quantized_int8", qm, X_tune, y_tune, X_test, y_test, "quantized"))
    torch.save(qm.state_dict(), MODEL_DIR / "quantized_int8.pth")

    print("Pruning 50% + quantizing to INT8...")
    pq = quantize_model(pruned_models[0.50])
    rows.append(evaluate_variant("pruned_50+int8", pq, X_tune, y_tune, X_test, y_test,
                                 "pruned+quantized",
                                 size_override_kb=pruned_int8_size_kb(pruned_models[0.50])))
    torch.save(pq.state_dict(), MODEL_DIR / "pruned_50_int8.pth")

    df = pd.DataFrame(rows)
    base_size = df.loc[0, "effective_size_kb"]
    base_lat = df.loc[0, "latency_ms_per_sample"]
    df["size_reduction_x"] = base_size / df["effective_size_kb"]
    df["speedup_x"] = base_lat / df["latency_ms_per_sample"]
    df.to_csv(RESULT_DIR / "compression_results.csv", index=False)

    show = df[["variant", "balanced_accuracy", "recall", "specificity", "roc_auc",
               "effective_size_kb", "size_reduction_x",
               "latency_ms_per_sample", "speedup_x"]].copy()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n================ COMPRESSION RESULTS ================")
    print(show.round(4).to_string(index=False))
    print("======================================================")
    print(f"\nSaved table to {RESULT_DIR / 'compression_results.csv'}")
    print("Saved compressed models into 'models/'.")
    print("\nHow to read it: compare balanced_accuracy and roc_auc to baseline_fp32. "
          "'specificity' = share of NORMAL traffic correctly left alone.\n"
          "Pick the smallest variant whose balanced_accuracy stays close to baseline. "
          "(Plain accuracy/F1 are in the CSV but are inflated by the ~98% attack share.)")


if __name__ == "__main__":
    main()