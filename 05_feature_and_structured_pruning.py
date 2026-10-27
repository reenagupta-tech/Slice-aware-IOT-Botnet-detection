"""
Phase 3b: Feature-importance pruning + structured/unstructured weight pruning
-----------------------------------------------------------------------------
WHAT THIS SCRIPT DOES, IN PLAIN WORDS
  The detector currently reads 115 numbers for every traffic sample. Many of
  them are redundant (the same statistic measured over 5 time windows, etc.).
  Fewer inputs = a smaller, faster model that needs less data per sample.

  PART 1 - FEATURE-IMPORTANCE PRUNING
    1. Rank all 115 features by importance (a Random Forest looks at the
       labeled TUNE set and measures which features best separate normal
       from attack traffic). The ranking is saved to a CSV and summarized
       by stream type, time window and statistic (good material for the report).
    2. Retrain the autoencoder (on normal traffic only, as always) using only
       the top-k features, for k = 115, 80, 60, 40, 30, 20, 10.
    3. Measure accuracy for each k and automatically pick the "knee": the
       smallest k that stays within 0.005 balanced accuracy of using all 115
       (decided on the TUNE set, never the test set).

  PART 2 - WEIGHT PRUNING ON THE CHOSEN MODEL
    - Unstructured pruning: set the smallest weights to zero (50/70/90%).
    - Structured pruning: physically REMOVE whole neurons (keep 75% / 50%),
      so the network really becomes smaller, then fine-tune.
    - INT8 quantization, alone and combined with pruning.
    Every variant gets its own re-tuned threshold and is evaluated on the same
    untouched test set. Results: results/compression_v2_results.csv

RUN (after 01 and 02 have been run):
    python 05_feature_and_structured_pruning.py            # does everything
  Optional:
    --stage rank | k | compress | all      (default all; the script resumes
                                            and skips finished work)
    --k 40                                 (with --stage k: train one k only)
    --chosen-k 40                          (override the automatic knee pick)
    --epochs 20  --max-seeds 3  --threads N
"""
import argparse
import copy
import io
import json
import pickle
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

DATA_DIR = Path("prepared_data")
MODEL_DIR = Path("models")
V2_DIR = MODEL_DIR / "v2"
RESULT_DIR = Path("results")
V2_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

K_LIST = [115, 80, 60, 40, 30, 20, 10]
KNEE_TOLERANCE = 0.005     # allowed drop in tune-set balanced accuracy
GOOD_ENOUGH = 0.99         # stop trying more seeds once a model reaches this (tune set)


# =====================================================================
# Shared helpers (same transform / threshold method as scripts 02 and 03)
# =====================================================================
def signed_log(X):
    X = np.asarray(X, dtype=np.float32)
    return np.sign(X) * np.log1p(np.abs(X))


def load_prepped(path, scaler, names=None, chunk=200_000):
    """
    Read a big CSV in chunks, apply the log transform + scaler to each chunk,
    and store the result in one float32 array. Never holds the raw table and
    its transformed copy at the same time, so peak memory stays low.
    """
    n_rows = sum(len(c) for c in pd.read_csv(path, usecols=[0], chunksize=chunk))
    X, y, pos = None, np.empty(n_rows, dtype=np.int8), 0
    for c in pd.read_csv(path, dtype=np.float32, chunksize=chunk):
        names = names or [col for col in c.columns if col != "label"]
        part = scaler.transform(signed_log(c[names].values)).astype(np.float32)
        if X is None:
            X = np.empty((n_rows, len(names)), dtype=np.float32)
        X[pos:pos + len(part)] = part
        y[pos:pos + len(part)] = c["label"].values.astype(np.int8)
        pos += len(part)
    return names, X, y.astype(int)


def load_data():
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    names, Xtr, _ = load_prepped(DATA_DIR / "train.csv", scaler)
    _, Xtu, ytu = load_prepped(DATA_DIR / "tune.csv", scaler, names)
    _, Xte, yte = load_prepped(DATA_DIR / "test.csv", scaler, names)
    return names, Xtr, Xtu, ytu, Xte, yte


def balanced_acc(y, pred):
    tp = np.sum((pred == 1) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0))
    return 0.5 * (tp / max(np.sum(y == 1), 1) + tn / max(np.sum(y == 0), 1))


def tune_threshold(errors, y):
    cands = np.unique(np.quantile(errors, np.linspace(0, 1, 1001)))
    best_t, best = cands[0], -1.0
    for t in cands:
        sc = balanced_acc(y, (errors > t).astype(int))
        if sc > best:
            best_t, best = t, sc
    return float(best_t), float(best)


def reconstruction_error(model, X, batch=100_000):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch])
            out.append(torch.mean((model(xb) - xb) ** 2, dim=1).numpy())
    return np.concatenate(out)


# =====================================================================
# Autoencoder with configurable layer sizes
# =====================================================================
class AE(nn.Module):
    """enc = (a, b, bottleneck), dec = (c, d). Default is the Phase 2 shape."""

    def __init__(self, n_in, enc=(64, 24, 6), dec=(24, 64)):
        super().__init__()
        self.enc, self.dec = tuple(enc), tuple(dec)
        self.encoder = nn.Sequential(
            nn.Linear(n_in, enc[0]), nn.ReLU(),
            nn.Linear(enc[0], enc[1]), nn.ReLU(),
            nn.Linear(enc[1], enc[2]), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(enc[2], dec[0]), nn.ReLU(),
            nn.Linear(dec[0], dec[1]), nn.ReLU(),
            nn.Linear(dec[1], n_in), nn.Tanh())

    def forward(self, x):
        return self.decoder(self.encoder(x))


def linears(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(int((p != 0).sum()) for p in model.parameters())
    return total, nonzero


def train_ae(X, n_in, seed, epochs, enc=(64, 24, 6), dec=(24, 64), model=None,
             lr=1e-3):
    torch.manual_seed(seed)
    model = model or AE(n_in, enc, dec)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X)), batch_size=256, shuffle=True)
    model.train()
    for _ in range(epochs):
        for (b,) in loader:
            opt.zero_grad()
            nn.functional.mse_loss(model(b), b).backward()
            opt.step()
    return model.eval()


def save_ckpt(path, model, features, quantized=False):
    torch.save({"features": [int(i) for i in features], "n_in": len(features),
                "enc": model.enc, "dec": model.dec, "quantized": quantized,
                "state_dict": model.state_dict()}, path)


def quantize(model):
    return torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).eval(), {nn.Linear}, dtype=torch.qint8)


def file_kb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1024


def latency_ms(model, X, n=1500):
    model.eval()
    xs = [torch.from_numpy(X[i:i + 1]) for i in range(min(n, len(X)))]
    with torch.no_grad():
        for s in xs[:100]:
            model(s)
        t0 = time.perf_counter()
        for s in xs:
            model(s)
    return (time.perf_counter() - t0) / len(xs) * 1000


def evaluate(model, Xtune, ytune, Xtest, ytest):
    thr, tune_bal = tune_threshold(reconstruction_error(model, Xtune), ytune)
    err = reconstruction_error(model, Xtest)
    pred = (err > thr).astype(int)
    rec = float(((pred == 1) & (ytest == 1)).sum() / (ytest == 1).sum())
    spec = float(((pred == 0) & (ytest == 0)).sum() / (ytest == 0).sum())
    return {"threshold": thr, "tune_balanced_accuracy": tune_bal,
            "balanced_accuracy": 0.5 * (rec + spec), "recall": rec,
            "specificity": spec, "roc_auc": float(roc_auc_score(ytest, err))}


# =====================================================================
# PART 1a - rank features
# =====================================================================
def parse_feature(name):
    m = re.match(r"^(.*)_L([0-9.]+)_(.*)$", name)
    return (m.group(1), "L" + m.group(2), m.group(3)) if m else (name, "?", "?")


def stage_rank(names, Xtune, ytune, threads):
    out = V2_DIR / "feature_ranking.csv"
    if out.exists():
        print(f"[rank] using existing {out}")
        return pd.read_csv(out)
    print("[rank] ranking features with a Random Forest on the labeled TUNE set ...")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xtune), size=min(120_000, len(Xtune)), replace=False)
    rf = RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=5,
                                class_weight="balanced", n_jobs=threads or 2,
                                random_state=0)
    rf.fit(Xtune[idx], ytune[idx])
    df = pd.DataFrame({"feature_index": range(len(names)), "feature": names,
                       "importance": rf.feature_importances_})
    parts = df["feature"].apply(parse_feature)
    df["stream"] = [p[0] for p in parts]
    df["time_window"] = [p[1] for p in parts]
    df["statistic"] = [p[2] for p in parts]
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df.to_csv(out, index=False)
    print(f"[rank] saved {out}")
    return df


def print_rank_summary(df):
    print("\nTop 10 features:")
    print(df.head(10)[["rank", "feature", "importance"]].round(4).to_string(index=False))
    for col in ("stream", "time_window", "statistic"):
        s = df.groupby(col)["importance"].sum().sort_values(ascending=False)
        print(f"\nShare of total importance by {col}:")
        print((s / s.sum()).round(3).to_string())


# =====================================================================
# PART 1b - train an autoencoder for each k
# =====================================================================
def stage_k(k, ranking, Xtr, Xtune, ytune, Xtest, ytest, epochs, max_seeds, extend_from=0):
    """
    Train an autoencoder on the top-k features, trying several random seeds and
    keeping the best by TUNE-set score (the test set is never used for this).
    extend_from=N: a result already exists -- keep it as the incumbent and only
    try the NEW seeds N, N+1, ... (saves redoing seeds that were already tried).
    """
    res_path = V2_DIR / f"k{k}_result.json"
    feats = ranking["feature_index"].values[:k]
    Xk, Xtuk, Xtek = Xtr[:, feats], Xtune[:, feats], Xtest[:, feats]
    best = None
    if res_path.exists():
        if not extend_from:
            print(f"[k={k}] already done, skipping")
            return json.load(open(res_path))
        prev = json.load(open(res_path))
        ck = torch.load(V2_DIR / f"k{k}_model.pt", weights_only=False)
        pm = AE(k, ck["enc"], ck["dec"]); pm.load_state_dict(ck["state_dict"]); pm.eval()
        best = (prev["tune_balanced_accuracy"], pm, prev["seed"])
        print(f"[k={k}] keeping earlier best (seed {prev['seed']}, tune "
              f"{prev['tune_balanced_accuracy']:.4f}); trying seeds {extend_from}..{max_seeds - 1}")
    for seed in range(extend_from, max_seeds):
        if best is not None and best[0] >= GOOD_ENOUGH:
            break
        t0 = time.time()
        m = train_ae(Xk, k, seed, epochs)
        thr, tune_bal = tune_threshold(reconstruction_error(m, Xtuk), ytune)
        print(f"[k={k}] seed {seed}: tune balanced accuracy {tune_bal:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if best is None or tune_bal > best[0]:
            best = (tune_bal, m, seed)
    tune_bal, model, seed = best
    ev = evaluate(model, Xtuk, ytune, Xtek, ytest)
    total, _ = count_params(model)
    res = {"k": int(k), "seed": int(seed), "params": int(total),
           "size_kb": total * 4 / 1024, **ev}
    save_ckpt(V2_DIR / f"k{k}_model.pt", model, feats)
    json.dump(res, open(res_path, "w"))
    print(f"[k={k}] kept seed {seed}: test balanced accuracy {ev['balanced_accuracy']:.4f}, "
          f"recall {ev['recall']:.4f}, specificity {ev['specificity']:.4f}, "
          f"AUC {ev['roc_auc']:.4f}, {total} params", flush=True)
    return res


def collect_k_results():
    rows = [json.load(open(V2_DIR / f"k{k}_result.json")) for k in K_LIST
            if (V2_DIR / f"k{k}_result.json").exists()]
    return pd.DataFrame(rows).sort_values("k", ascending=False).reset_index(drop=True)


def pick_knee(df):
    ref = df.loc[df["k"] == 115, "tune_balanced_accuracy"].iloc[0]
    ok = df[df["tune_balanced_accuracy"] >= ref - KNEE_TOLERANCE]
    return int(ok["k"].min())


# =====================================================================
# PART 2 - weight pruning on the chosen model
# =====================================================================
def unstructured_prune(base, amount, X, epochs=10, seed=0, lr=5e-4):
    model = copy.deepcopy(base)
    params = [(m, "weight") for m in linears(model)]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    model = train_ae(X, X.shape[1], seed, epochs, model=model, lr=lr)
    for m, name in params:
        prune.remove(m, name)
    return model.eval()


def structured_prune(base, keep_frac, X, epochs=10, seed=0, lr=5e-4):
    """
    REALLY remove neurons. For each hidden layer (not the input, output or the
    6-unit bottleneck) keep the `keep_frac` of neurons with the largest
    contribution, = norm of incoming weights x norm of outgoing weights.
    Then build a smaller network, copy the surviving weights and fine-tune.
    """
    L = linears(base)               # e0, e1, e2, d0, d1, d2
    def keep_idx(layer_in, layer_out):
        score = layer_in.weight.detach().norm(dim=1) * layer_out.weight.detach().norm(dim=0)
        n = max(int(np.ceil(keep_frac * len(score))), 2)
        return torch.sort(torch.topk(score, n).indices).values

    k_e0, k_e1 = keep_idx(L[0], L[1]), keep_idx(L[1], L[2])
    k_d0, k_d1 = keep_idx(L[3], L[4]), keep_idx(L[4], L[5])
    n_in = L[0].in_features
    new = AE(n_in, enc=(len(k_e0), len(k_e1), L[2].out_features),
             dec=(len(k_d0), len(k_d1)))
    N = linears(new)
    with torch.no_grad():
        N[0].weight.copy_(L[0].weight[k_e0]);               N[0].bias.copy_(L[0].bias[k_e0])
        N[1].weight.copy_(L[1].weight[k_e1][:, k_e0]);      N[1].bias.copy_(L[1].bias[k_e1])
        N[2].weight.copy_(L[2].weight[:, k_e1]);            N[2].bias.copy_(L[2].bias)
        N[3].weight.copy_(L[3].weight[k_d0]);               N[3].bias.copy_(L[3].bias[k_d0])
        N[4].weight.copy_(L[4].weight[k_d1][:, k_d0]);      N[4].bias.copy_(L[4].bias[k_d1])
        N[5].weight.copy_(L[5].weight[:, k_d1]);            N[5].bias.copy_(L[5].bias)
    return train_ae(X, n_in, seed, epochs, model=new, lr=lr)


def int8_theory_kb(fp_model, sparse=False):
    """
    Size convention used for EVERY variant in the results table (so sizes are
    comparable): FP32 weight = 4 bytes, INT8 weight = 1 byte, bias = 4 bytes,
    plus 8 bytes per layer for the INT8 scale/zero-point. Sparse (pruned)
    models count only their non-zero weights, as if stored in a sparse format.
    The real saved-file size is reported separately as `file_kb`.
    """
    layers = linears(fp_model)
    w = sum(int((m.weight != 0).sum()) if sparse else m.weight.numel() for m in layers)
    b = sum(m.bias.numel() for m in layers) * 4
    return (w + b + 8 * len(layers)) / 1024


# Recovery strategies tried IN ORDER after pruning: (epochs, learning rate, seed).
# We found that long retraining can DAMAGE a good model (it sometimes drifts and
# starts missing attack types, differently on different machines), while pruning
# with no retraining, or only a couple of gentle epochs, keeps it intact. So the
# cheap, stable options come first and the long retraining is the last resort.
RECOVERY_STRATEGIES = [(0, 1e-4, 0),      # no retraining (deterministic)
                       (2, 1e-4, 0), (2, 1e-4, 1),
                       (3, 2e-4, 0),
                       (10, 5e-4, 0), (10, 5e-4, 1)]


def prune_with_retries(label, make_model, base_tune_score, Xtuk, ytune, tolerance=0.003):
    """
    Try each recovery strategy and keep the first whose TUNE-set score is within
    `tolerance` of the unpruned model; if none is, keep the best one. The test
    set is never used for this choice. `make_model(epochs, lr, seed)` builds
    the pruned model with that strategy.
    """
    best = None
    for epochs, lr, seed in RECOVERY_STRATEGIES:
        model = make_model(epochs, lr, seed)
        score = tune_threshold(reconstruction_error(model, Xtuk), ytune)[1]
        desc = "no retraining" if epochs == 0 else f"{epochs} epoch(s), lr {lr:g}, seed {seed}"
        if best is None or score > best[0]:
            best = (score, model, desc)
        if score >= base_tune_score - tolerance:
            break
        print(f"    {label}: {desc} scored {score:.4f} "
              f"(unpruned {base_tune_score:.4f}), trying next strategy", flush=True)
    note = "" if best[0] >= base_tune_score - tolerance else "  (below target; kept best)"
    print(f"    {label}: kept [{best[2]}] (tune {best[0]:.4f}){note}", flush=True)
    return best[1]


def stage_compress(k, ranking, Xtr, Xtune, ytune, Xtest, ytest, epochs):
    feats = ranking["feature_index"].values[:k]
    Xk, Xtuk, Xtek = Xtr[:, feats], Xtune[:, feats], Xtest[:, feats]

    # reference: the all-115-feature model, retrained by this script (same recipe)
    ck115 = torch.load(V2_DIR / "k115_model.pt", weights_only=False)
    ref = AE(115, ck115["enc"], ck115["dec"]); ref.load_state_dict(ck115["state_dict"]); ref.eval()
    ckk = torch.load(V2_DIR / f"k{k}_model.pt", weights_only=False)
    base = AE(k, ckk["enc"], ckk["dec"]); base.load_state_dict(ckk["state_dict"]); base.eval()

    base_tune = tune_threshold(reconstruction_error(base, Xtuk), ytune)[1]
    variants = []   # (name, model, features, X_tune, X_test, size_kb, quantized, pruned_note)
    def add(name, model, F, Xt, Xe, size_kb, quantized=False, kind=""):
        variants.append((name, model, F, Xt, Xe, size_kb, quantized, kind))

    # The reference was trained on the features in RANKED order (all 115 of
    # them), so it must be evaluated in that same column order.
    f115 = np.array(ck115["features"])
    add("all115_fp32 (reference)", ref, f115, Xtune[:, f115], Xtest[:, f115],
        count_params(ref)[0] * 4 / 1024)
    add(f"top{k}_fp32", base, feats, Xtuk, Xtek, count_params(base)[0] * 4 / 1024)

    for amt in (0.5, 0.7, 0.9):
        print(f"[compress] unstructured {int(amt * 100)}% ...", flush=True)
        p = prune_with_retries(f"unstructured {int(amt * 100)}%",
                               lambda ep, lr, sd, amt=amt: unstructured_prune(base, amt, Xk, epochs=ep, seed=sd, lr=lr),
                               base_tune, Xtuk, ytune)
        add(f"top{k}_unstructured{int(amt * 100)}", p, feats, Xtuk, Xtek,
            count_params(p)[1] * 4 / 1024, kind="sparse")
        if amt == 0.5:
            add(f"top{k}_unstructured50+int8", quantize(p), feats, Xtuk, Xtek,
                int8_theory_kb(p, sparse=True), True, "sparse")

    for keep in (0.75, 0.5):
        print(f"[compress] structured keep {int(keep * 100)}% of neurons ...", flush=True)
        s = prune_with_retries(f"structured {int(keep * 100)}%",
                               lambda ep, lr, sd, keep=keep: structured_prune(base, keep, Xk, epochs=ep, seed=sd, lr=lr),
                               base_tune, Xtuk, ytune)
        add(f"top{k}_structured{int(keep * 100)}", s, feats, Xtuk, Xtek,
            count_params(s)[0] * 4 / 1024)
        if keep == 0.5:
            add(f"top{k}_structured50+int8", quantize(s), feats, Xtuk, Xtek,
                int8_theory_kb(s), True)

    add(f"top{k}_int8", quantize(base), feats, Xtuk, Xtek, int8_theory_kb(base), True)

    rows, manifest = [], []
    ref_size = variants[0][5]
    for name, model, F, Xt, Xe, size_kb, quantized, kind in variants:
        print(f"[compress] evaluating {name} ...", flush=True)
        ev = evaluate(model, Xt, ytune, Xe, ytest)
        total, nonzero = count_params(model) if not quantized else (np.nan, np.nan)
        rows.append({"variant": name, "n_features": len(F), **ev,
                     "effective_size_kb": size_kb, "size_reduction_x": ref_size / size_kb,
                     "file_kb": file_kb(model),
                     "params": total, "params_nonzero": nonzero,
                     "latency_ms_per_sample": latency_ms(model, Xe)})
        fname = "v2_" + re.sub(r"[^A-Za-z0-9_+]", "_", name).replace("+", "_") + ".pt"
        save_ckpt(MODEL_DIR / fname, model, F, quantized)
        manifest.append({"variant": name, "file": fname, "threshold": ev["threshold"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / "compression_v2_results.csv", index=False)
    pd.DataFrame(manifest).to_csv(MODEL_DIR / "v2_manifest.csv", index=False)
    return df


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["rank", "k", "compress", "all"])
    ap.add_argument("--k", type=int)
    ap.add_argument("--k-list", default=None,
                    help="feature counts to test, comma separated (default 115,80,60,40,30,20,10). "
                         "Fewer values = much faster, e.g. 115,60,40,20. 115 is always included.")
    ap.add_argument("--chosen-k", type=int)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-seeds", type=int, default=6)
    ap.add_argument("--extend-from", type=int, default=0,
                    help="resume: keep the existing result and try only seeds from N upward")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    global K_LIST
    if a.k_list:
        K_LIST = sorted({115, *[int(x) for x in a.k_list.split(",")]}, reverse=True)

    print("Loading data ...")
    names, Xtr, Xtune, ytune, Xtest, ytest = load_data()
    print(f"  {len(names)} features | {len(Xtr)} train | {len(Xtune)} tune | {len(Xtest)} test rows\n")

    ranking = stage_rank(names, Xtune, ytune, a.threads)
    if a.stage == "rank":
        print_rank_summary(ranking)
        return

    if a.stage in ("k", "all"):
        for k in ([a.k] if a.k else K_LIST):
            stage_k(k, ranking, Xtr, Xtune, ytune, Xtest, ytest, a.epochs, a.max_seeds,
                    a.extend_from)

    if a.stage in ("compress", "all"):
        kdf = collect_k_results()
        missing = set(K_LIST) - set(kdf["k"])
        if missing:
            raise SystemExit(f"Missing results for k = {sorted(missing)}. Run --stage k first.")
        print_rank_summary(ranking)
        kdf.to_csv(RESULT_DIR / "feature_pruning_by_k.csv", index=False)
        show = kdf[["k", "params", "size_kb", "balanced_accuracy", "recall",
                    "specificity", "roc_auc", "tune_balanced_accuracy"]].round(4)
        print("\n============ FEATURE PRUNING: ACCURACY vs NUMBER OF FEATURES ============")
        print(show.to_string(index=False))
        chosen = a.chosen_k or pick_knee(kdf)
        print(f"\nChosen k = {chosen}  (smallest k within {KNEE_TOLERANCE} tune-set "
              f"balanced accuracy of using all 115 features)")
        df = stage_compress(chosen, ranking, Xtr, Xtune, ytune, Xtest, ytest, a.epochs)
        cols = ["variant", "n_features", "balanced_accuracy", "recall", "specificity",
                "roc_auc", "effective_size_kb", "size_reduction_x"]
        pd.set_option("display.width", 220)
        print("\n============ COMBINED COMPRESSION RESULTS ============")
        print(df[cols].round(4).to_string(index=False))
        print("\nSaved: results/feature_pruning_by_k.csv, results/compression_v2_results.csv,")
        print("       models/v2/feature_ranking.csv, models/v2_manifest.csv")


if __name__ == "__main__":
    main()