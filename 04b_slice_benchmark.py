"""
Phase 4 - Step B: Benchmark every model variant on ONE slice (run inside a VM)
-------------------------------------------------------------------------------
Run this script inside each slice VM (eMBB, URLLC, mMTC). It measures how each
model variant behaves under THAT VM's resource limits:

    - latency for ONE traffic sample (mean, median, 95th, 99th percentile)
    - throughput (samples per second, processing a stream in batches)
    - memory (peak RAM the model process needed)
    - CPU time actually used per 1,000 samples
    - model load (start-up) time
    - model file size
    - detection quality (balanced accuracy, recall, specificity) so we can
      confirm accuracy does not change between slices

Each model is measured in its OWN fresh process, so memory numbers are clean
(one model's memory never pollutes another's).

USAGE (inside the VM, from the benchmark_bundle folder):
    python3 04b_slice_benchmark.py --slice mMTC
Optional:
    --samples 3000      number of single-sample latency measurements (default 3000)
    --repeats 3         repeat every model this many times, keep the median
                        (default 3; more repeats = steadier numbers)

Output: results_<slice>.csv  (copy it back to your main computer for Phase 6)
"""
import argparse
import json
import os
import pickle
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------
# Same architecture as Phase 2 (needed to load the saved weights)
# ---------------------------------------------------------------
class AE(nn.Module):
    """Autoencoder with configurable layer sizes (enc = a,b,bottleneck; dec = c,d).
    The default sizes are the original Phase 2 model."""

    def __init__(self, n_in=115, enc=(64, 24, 6), dec=(24, 64)):
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


def signed_log(X):
    """Same transform used in Phase 2/3 (applied before the scaler)."""
    X = np.asarray(X, dtype=np.float32)
    return np.sign(X) * np.log1p(np.abs(X))


def load_model(path, variant):
    """
    Returns (model, feature_indices_or_None).
      .pt  files (Phase 3b): a checkpoint that stores its own layer sizes,
            whether it is INT8, and WHICH input features it uses.
      .pth files (Phase 3):  plain weights for the original 115-feature model.
    """
    if str(path).endswith(".pt"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = AE(ck["n_in"], ck["enc"], ck["dec"])
        if ck["quantized"]:
            model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(ck["state_dict"])
        return model.eval(), ck["features"]
    model = AE()
    if "int8" in variant:
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval(), None


def available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return psutil.cpu_count(logical=True) or 1


def peak_rss_mb():
    try:
        import resource  # Linux / macOS
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1024 if sys.platform != "darwin" else peak / 1e6
    except ImportError:  # Windows fallback
        return psutil.Process().memory_info().rss / 1e6


# ---------------------------------------------------------------
# WORKER: measures ONE model in a fresh process, prints JSON
# ---------------------------------------------------------------
def worker(variant, file, threshold, n_latency, threads=None):
    torch.set_num_threads(threads or available_cpus())
    proc = psutil.Process()
    rss_start = proc.memory_info().rss / 1e6

    # load data and preprocess (not part of the model's own cost)
    sample = pd.read_csv(HERE / "sample.csv", dtype=np.float32)
    y = sample["label"].values.astype(int)
    X = sample.drop(columns=["label"]).values
    with open(HERE / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    X = scaler.transform(signed_log(X)).astype(np.float32)
    rss_before_model = proc.memory_info().rss / 1e6

    # --- model load / start-up time ---
    t0 = time.perf_counter()
    model, feats = load_model(HERE / "models" / file, variant)
    load_ms = (time.perf_counter() - t0) * 1000
    if feats is not None:               # feature-pruned model reads fewer inputs
        X = np.ascontiguousarray(X[:, feats])
    rss_after_model = proc.memory_info().rss / 1e6

    # --- latency: one sample at a time ---
    singles = [torch.from_numpy(X[i:i + 1]) for i in range(n_latency)]
    times = []
    with torch.no_grad():
        for s in singles[:200]:               # warm-up (not recorded)
            model(s)
        for s in singles:
            t = time.perf_counter()
            r = model(s)
            _ = torch.mean((r - s) ** 2).item()  # include the error computation
            times.append((time.perf_counter() - t) * 1000)
    times = np.array(times)

    # --- throughput + CPU use: stream the whole sample in batches ---
    BATCH = 256
    with torch.no_grad():                      # warm-up pass (not timed)
        for i in range(0, min(len(X), BATCH * 8), BATCH):
            model(torch.from_numpy(X[i:i + BATCH]))
    cpu0, wall0 = time.process_time(), time.perf_counter()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb = torch.from_numpy(X[i:i + BATCH])
            errs.append(torch.mean((model(xb) - xb) ** 2, dim=1).numpy())
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    errs = np.concatenate(errs)

    # --- detection quality on this slice ---
    pred = (errs > threshold).astype(int)
    rec = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
    spec = float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))

    out = {
        "variant": variant,
        "latency_mean_ms": float(times.mean()),
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_p99_ms": float(np.percentile(times, 99)),
        "throughput_samples_per_s": len(X) / wall,
        "cpu_ms_per_1000_samples": cpu / len(X) * 1e6,
        "load_time_ms": load_ms,
        # RAM, in three honest parts:
        #   runtime_ram_mb = memory used just by Python + PyTorch (no data, no model)
        #   model_ram_mb   = extra memory taken by loading THIS model (tiny)
        #   peak_ram_mb    = highest total memory of the whole process
        # For models this small, the PyTorch runtime dominates total RAM.
        "runtime_ram_mb": rss_start,
        "model_ram_mb": max(rss_after_model - rss_before_model, 0.0),
        "peak_ram_mb": peak_rss_mb(),
        "model_file_kb": (HERE / "models" / file).stat().st_size / 1024,
        "balanced_accuracy": 0.5 * (rec + spec),
        "recall": rec,
        "specificity": spec,
    }
    print("RESULT_JSON:" + json.dumps(out))


# ---------------------------------------------------------------
# MAIN: describes this VM, runs a worker per model, saves a CSV
# ---------------------------------------------------------------
def describe_machine():
    vm = psutil.virtual_memory()
    return {
        "cpus_available": available_cpus(),
        "ram_total_mb": round(vm.total / 1e6),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "os": platform.platform(),
    }


def run_worker_subprocess(variant, file, threshold, n_latency, threads=None):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", variant,
           "--file", file, "--threshold", str(threshold), "--samples", str(n_latency)]
    if threads:
        cmd += ["--threads", str(threads)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise RuntimeError(f"Worker failed for {variant}:\n{p.stdout}\n{p.stderr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", help="slice name, e.g. eMBB, URLLC, mMTC")
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None,
                    help="PyTorch threads (default: all CPUs the machine gives us). "
                         "Set this to the slice's CPU count when using Docker.")
    ap.add_argument("--worker")
    ap.add_argument("--file")
    ap.add_argument("--threshold", type=float)
    args = ap.parse_args()

    if args.worker:
        worker(args.worker, args.file, args.threshold, args.samples, args.threads)
        return
    if not args.slice:
        ap.error("--slice is required (e.g. --slice mMTC)")

    machine = describe_machine()
    print(f"Slice: {args.slice}")
    print(f"This machine: {machine['cpus_available']} CPU(s), {machine['ram_total_mb']} MB RAM, "
          f"torch {machine['torch']}\n")

    table = pd.read_csv(HERE / "thresholds.csv")
    rows = []
    for _, r in table.iterrows():
        runs = []
        for k in range(args.repeats):
            print(f"  measuring {r['variant']:16s} (run {k + 1}/{args.repeats}) ...", flush=True)
            runs.append(run_worker_subprocess(r["variant"], r["file"], r["threshold"],
                                              args.samples, args.threads))
        df = pd.DataFrame(runs)
        med = df.drop(columns=["variant"]).median()           # median of repeats = robust to noise
        row = {"slice": args.slice, "variant": r["variant"], **med.to_dict()}
        rows.append(row)

    out = pd.DataFrame(rows)
    for k, v in machine.items():
        out[k] = v
    csv_path = Path(f"results_{args.slice}.csv")
    out.to_csv(csv_path, index=False)

    show = out[["variant", "latency_mean_ms", "latency_p99_ms", "throughput_samples_per_s",
                "peak_ram_mb", "model_file_kb", "cpu_ms_per_1000_samples",
                "balanced_accuracy"]].round(3)
    pd.set_option("display.width", 200)
    print(f"\n=========== RESULTS FOR SLICE: {args.slice} ===========")
    print(show.to_string(index=False))
    print("=================================================")
    print(f"Saved: {csv_path}  (copy this file back to your main computer)")


if __name__ == "__main__":
    main()