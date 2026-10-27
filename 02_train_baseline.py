"""
Phase 2 - Step 2: Train the baseline autoencoder detection model
-------------------------------------------------------------------
WHAT AN AUTOENCODER IS, IN PLAIN WORDS:
  It's a small neural network shaped like an hourglass. You feed it a row
  of 115 numbers (one network traffic "snapshot"). It squeezes that down
  to a few numbers in the middle (the "bottleneck"), then tries to expand
  it back out to 115 numbers that match the original as closely as possible.

  We train it ONLY on benign (normal) traffic. So it becomes very good at
  recreating normal patterns. When we later feed it ATTACK traffic (which
  it has never seen), it will do a BAD job of recreating it — the
  difference between input and output (called "reconstruction error")
  will be much higher. That gap is how we detect attacks.

STEPS THIS SCRIPT DOES:
  1. Load train.csv / val.csv / test.csv made by 01_data_prep.py
  2. Scale all features to the [0, 1] range (Min-Max scaling, same as the
     papers do) so the network trains smoothly.
  3. Build the autoencoder in PyTorch.
  4. Train it for a number of "epochs" (full passes over the training data).
  5. Use the validation set to pick a detection threshold.
  6. Evaluate on the test set and print accuracy / precision / recall / F1.
  7. Save the trained model to disk so Phase 3 (compression) can load it.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from pathlib import Path
import pickle
import time

DATA_DIR = Path("prepared_data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def signed_log(X):
    """
    Sign-preserving log transform, applied BEFORE Min-Max scaling.
    N-BaIoT features are extremely heavy-tailed (a few huge values, most
    values tiny). Plain Min-Max scaling squashes normal traffic into a
    sliver near zero, which made the model flag 20% of normal traffic
    (65% for the Ecobee device). Compressing the range with a log first
    fixes that (balanced accuracy 0.90 -> ~0.996 in our tests).
    Sign is kept because some features (covariance, correlation) can be
    negative.
    """
    X = np.asarray(X, dtype=np.float32)
    return np.sign(X) * np.log1p(np.abs(X))
print(f"Training on: {DEVICE}")


# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
def load_data():
    # float32 halves memory use (important: test.csv has >1 million rows)
    train_df = pd.read_csv(DATA_DIR / "train.csv", dtype=np.float32)
    tune_df = pd.read_csv(DATA_DIR / "tune.csv", dtype=np.float32)
    test_df = pd.read_csv(DATA_DIR / "test.csv", dtype=np.float32)

    feature_cols = [c for c in train_df.columns if c != "label"]

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_tune = tune_df[feature_cols].values.astype(np.float32)
    y_tune = tune_df["label"].values.astype(np.int64)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.int64)

    return X_train, X_tune, y_tune, X_test, y_test, feature_cols


# ---------------------------------------------------------------
# 2. The autoencoder model itself
#    (mirrors the "all-in-one" style design from Catillo et al., 2023 —
#     a small, symmetric hourglass network)
# ---------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 24), nn.ReLU(),
            nn.Linear(24, 6), nn.ReLU(),       # bottleneck: squeezed to 6 numbers
        )
        self.decoder = nn.Sequential(
            nn.Linear(6, 24), nn.ReLU(),
            nn.Linear(24, 64), nn.ReLU(),
            nn.Linear(64, n_features), nn.Tanh(),  # Tanh output, same as the papers
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# ---------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------
def train_model(X_train, n_features, epochs=20, batch_size=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)  # different seed = different random start
    model = Autoencoder(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    start = time.time()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
        avg_loss = running_loss / len(dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{epochs} — avg reconstruction loss: {avg_loss:.6f}")

    print(f"Training took {time.time() - start:.1f} seconds")
    return model


# ---------------------------------------------------------------
# 4. Compute reconstruction error for a batch of rows
# ---------------------------------------------------------------
def reconstruction_error(model, X, batch=100_000):
    """Per-row mean squared error, computed in batches to save memory."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32).to(DEVICE)
            out.append(torch.mean((model(xb) - xb) ** 2, dim=1).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------
# 5. Pick a detection threshold using the TUNE set (benign + a small,
#    labeled slice of attack traffic -- never the final test set).
#
#    IMPORTANT LESSON (class imbalance): N-BaIoT has roughly 50 attack
#    rows for every 1 benign row here. If you pick the threshold by
#    maximizing plain F1 or accuracy, the "best" answer is to flag
#    EVERYTHING as an attack (you would score ~98% while detecting
#    nothing useful -- every normal packet becomes a false alarm).
#    So we maximize BALANCED ACCURACY instead:
#         balanced accuracy = (attack detection rate + normal-traffic
#                              correct rate) / 2
#    It treats both classes as equally important regardless of how many
#    rows each has, so flag-everything scores only 0.5 (= coin flip).
#
#    Candidate thresholds come from QUANTILES of the observed errors
#    (not an even spread from min to max), because reconstruction errors
#    are very skewed: a few huge outliers would make an even grid far
#    too coarse where it matters.
# ---------------------------------------------------------------
def balanced_accuracy_from(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    p = max(np.sum(y_true == 1), 1)
    n = max(np.sum(y_true == 0), 1)
    return 0.5 * (tp / p + tn / n)


def pick_threshold(model, X_tune, y_tune):
    errors = reconstruction_error(model, X_tune)

    candidates = np.unique(np.quantile(errors, np.linspace(0.0, 1.0, 1001)))
    best_threshold, best_score = candidates[0], -1.0
    for t in candidates:
        score = balanced_accuracy_from(y_tune, (errors > t).astype(int))
        if score > best_score:
            best_score, best_threshold = score, t

    naive_threshold = errors[y_tune == 0].mean() + errors[y_tune == 0].std()
    print(f"\nOld-style guess (benign mean + std) would have been: {naive_threshold:.6f}")
    print(f"Threshold chosen by maximizing balanced accuracy on tune data: "
          f"{best_threshold:.6f} (tune-set balanced accuracy: {best_score:.4f})")
    return best_threshold, best_score


# ---------------------------------------------------------------
# 6. Evaluate on the test set
#    We report the usual four metrics AND the ones that are not fooled
#    by class imbalance (normal-traffic correct rate, balanced accuracy,
#    ROC-AUC). Always look at "Normal traffic correct" -- it tells you how
#    many normal packets are wrongly flagged.
# ---------------------------------------------------------------
def evaluate(model, X_test, y_test, threshold):
    errors = reconstruction_error(model, X_test)
    y_pred = (errors > threshold).astype(int)  # 1 = flagged as attack

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn = np.sum((y_pred == 0) & (y_test == 0))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    specificity = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (rec + specificity)
    auc = roc_auc_score(y_test, errors)

    print("\n===== TEST SET RESULTS =====")
    print(f"Test set: {int((y_test == 0).sum())} normal rows, {int((y_test == 1).sum())} attack rows "
          f"({(y_test == 1).mean() * 100:.1f}% attack)")
    print(f"Attacks detected (recall)     : {rec:.4f}")
    print(f"Normal traffic correct        : {specificity:.4f}   <- false alarm rate = {1 - specificity:.4f}")
    print(f"Balanced accuracy             : {bal_acc:.4f}   <- the honest headline number")
    print(f"ROC-AUC (threshold-free)      : {auc:.4f}")
    print(f"Accuracy / Precision / F1     : {acc:.4f} / {prec:.4f} / {f1:.4f}"
          f"   (inflated by the imbalance, report with care)")
    print("=============================")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "specificity": specificity, "balanced_accuracy": bal_acc, "roc_auc": auc}


def main():
    X_train, X_tune, y_tune, X_test, y_test, feature_cols = load_data()
    n_features = X_train.shape[1]
    print(f"Loaded data — {n_features} features, "
          f"{len(X_train)} train rows, {len(X_tune)} tune rows, {len(X_test)} test rows")

    # Scale features to [0, 1] using ONLY the training data statistics
    # (this avoids "peeking" at test data, which would be a data-leakage bug)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(signed_log(X_train)).astype(np.float32)
    X_tune_scaled = scaler.transform(signed_log(X_tune)).astype(np.float32)
    X_test_scaled = scaler.transform(signed_log(X_test)).astype(np.float32)

    # ---------------------------------------------------------------
    # MODEL SELECTION ACROSS RANDOM SEEDS.
    # Finding from our experiments: with the exact same data and settings,
    # some random starting points give a model that detects ~every attack
    # (balanced accuracy ~0.997), while others reconstruct certain attack
    # types (gafgyt TCP/UDP floods) as well as normal traffic and miss them
    # (balanced accuracy ~0.90). So we train up to MAX_SEEDS models and keep
    # the one with the best score on the TUNE set. The test set is never
    # used for this choice, so the final numbers stay honest.
    # ---------------------------------------------------------------
    MAX_SEEDS = 5
    GOOD_ENOUGH = 0.98   # stop early once a model reaches this tune score
    best = None
    for seed in range(MAX_SEEDS):
        print(f"\n=== Training autoencoder on BENIGN traffic only (seed {seed}) ===")
        cand = train_model(X_train_scaled, n_features, seed=seed)
        cand_thr, cand_score = pick_threshold(cand, X_tune_scaled, y_tune)
        if best is None or cand_score > best[0]:
            best = (cand_score, cand, cand_thr, seed)
        if cand_score >= GOOD_ENOUGH:
            break
    _, model, threshold, chosen_seed = best
    print(f"\nKept the model from seed {chosen_seed} "
          f"(best tune-set balanced accuracy {best[0]:.4f}).")
    metrics = evaluate(model, X_test_scaled, y_test, threshold)
    metrics["chosen_seed"] = chosen_seed

    # ---- Save everything Phase 3 (compression) will need ----
    torch.save(model.state_dict(), MODEL_DIR / "baseline_model.pth")
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "threshold.pkl", "wb") as f:
        pickle.dump(threshold, f)
    with open(MODEL_DIR / "metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    print(f"\nSaved model weights to {MODEL_DIR / 'baseline_model.pth'}")
    print(f"Saved scaler to       {MODEL_DIR / 'scaler.pkl'}")
    print(f"Saved threshold to    {MODEL_DIR / 'threshold.pkl'}")
    print("\nPhase 2 complete. You now have a trained baseline model ready for Phase 3 (compression).")


if __name__ == "__main__":
    main()