# Slice-Aware IoT Botnet Detection

**Lightweight Model Design and Performance Evaluation Across Virtualized 5G Network Slices**

B.Tech project, Delhi Technological University (DTU), Department of Information Technology.

| | |
|---|---|
| Team | Simran (2K23/IT/160), Reena Gupta (2K23/IT/202) |
| Supervisor | Dr. Anamika Chauhan |
| Group | Group No. 160 |
## What this project does

IoT devices are attacked by botnets such as Mirai and BASHLITE, and the detector that catches them has to run on very different networks. A 5G network is split into virtual slices (eMBB, URLLC, mMTC) with different CPU, memory and latency limits. This project asks: **how small and fast can a botnet detector be made, and how does it behave inside each slice?**

The pipeline:

1. Train an **autoencoder on benign traffic only** (N-BaIoT dataset, 9 IoT devices, 115 traffic features).
2. **Compress** it: weight pruning, structured pruning (real neuron removal), feature-importance pruning (115 to 20 features) and INT8 quantization.
3. **Benchmark** every variant inside resource-limited virtual machines that stand in for the three slice types.

## Repository contents

| File | Purpose |
|---|---|
| `01_data_prep.py` | Reads the N-BaIoT CSVs of all devices and writes `train.csv`, `tune.csv`, `test.csv` (chunked, memory-safe). |
| `02_train_baseline.py` | Trains the baseline autoencoder (115-64-24-6-24-64-115), picks the detection threshold on the tune set, evaluates on the test set. |
| `03_compress.py` | Phase 3: global weight pruning (30/50/70/90%), INT8 quantization, pruning + INT8. |
| `05_feature_and_structured_pruning.py` | Phase 3b: Random Forest feature ranking, retraining on the top-k features, structured and unstructured pruning, INT8. |
| `04a_make_benchmark_bundle.py` | Builds a small `benchmark_bundle/` (models, scaler, thresholds, 20,000 sample rows) to copy into a VM. |
| `04b_slice_benchmark.py` | Runs inside a slice VM or on any device: latency, throughput, CPU time, RAM, model size, accuracy for each variant. |
| `PHASE4_VM_SETUP_GUIDE.md` | Step-by-step guide to build the VirtualBox slice VM (or the Docker alternative). |
| `results/` | Result tables from the executed runs. |

## Setup

Tested with Python 3.12 on Windows 11 (PyTorch 2.2.0, CPU only). Install the libraries:

```
pip install -r requirements.txt
```

### Dataset (not included)

The N-BaIoT dataset is too large for GitHub. Download it from the [UCI repository](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot) and unpack each device so the CSVs sit under `nbaiot_data/` next to the scripts, for example:

```
nbaiot_data/
  Danmini_Doorbell/
    benign_traffic.csv
    gafgyt_attacks/...
    mirai_attacks/...
  Ecobee_Thermostat/...
```

Plan for about 8 to 10 GB of free disk space for the prepared data.

## How to run

```
python 01_data_prep.py                      # about 10-20 min, creates prepared_data/
python 02_train_baseline.py                 # trains the baseline, saves models/
python 03_compress.py                       # pruning + INT8, writes results/compression_results.csv
python 05_feature_and_structured_pruning.py --k-list 115,60,40,20
python 04a_make_benchmark_bundle.py         # creates benchmark_bundle/
```

Then copy `benchmark_bundle/` into a slice VM (see `PHASE4_VM_SETUP_GUIDE.md`) and run:

```
python 04b_slice_benchmark.py --slice mMTC --threads 1 --repeats 2
```

## Evaluation method (important)

N-BaIoT as used here is about 94 to 98% attack traffic. A detector that flags everything as an attack gets about 98% accuracy and an F1 near 0.99 while being useless. So:

- The decision threshold is chosen on a labelled **tune** set by maximising **balanced accuracy**.
- Results are reported as balanced accuracy, recall, specificity and ROC-AUC, evaluated once on an untouched **test** set (1,443,732 rows: 83,391 benign, 1,360,341 attack).
- Feature ranking, k selection and pruning choices use the tune set only, never the test set.
- Training is seed-sensitive (some seeds miss flood attacks), so the scripts train several seeds and keep the best by tune-set score.

## Results so far

Baseline autoencoder (all 115 features): recall 0.9998, specificity 0.9791, balanced accuracy 0.9895, ROC-AUC 0.9959.

Feature and structured pruning (test set):

| Variant | Balanced accuracy | Size reduction (theoretical) |
|---|---|---|
| All 115 features (reference) | 0.9968 | 1x |
| Top 20 features, FP32 | 0.9948 | 3.0x |
| Top 20, unstructured 90% | 0.9948 | 23x |
| Top 20, structured 50% | 0.9948 | 8.0x |
| Top 20, structured 50% + INT8 | 0.9948 | 27x (2.6 KB) |

Sizes for pruned models are theoretical sparse or INT8 sizes. The real file shrinks 5.5x (77 KB to 14 KB) for structured 50%, because unstructured zeros are still stored densely.

mMTC-style VM (1 CPU, 1 GB RAM, 40% execution cap):

| Variant | Throughput (samples/s) | p50 latency (ms) | File (KB) |
|---|---|---|---|
| 115 features (reference) | 62.9k | 0.29 | 77.2 |
| Top 20, FP32 | 126k | 0.30 | 28.9 |
| Top 20, structured 50% | 137k | 0.32 | 14.0 |
| Top 20, structured 50% + INT8 | 102k | 0.69 | 12.0 |

Findings so far:

- Reducing the input to the top 20 features roughly doubles throughput and costs about 0.2 points of balanced accuracy.
- INT8 dynamic quantization makes the file slightly smaller but is slower on a CPU for a model this small.
- Peak memory is about 310 MB for every variant; the PyTorch runtime dominates, not the model.

## Status and limitations

Completed: literature review, data preparation, baseline, compression, feature and structured pruning, one slice (mMTC) benchmarked.

Remaining: URLLC and eMBB slices, real-hardware test and energy estimate, comparison with the original per-device N-BaIoT autoencoders, end-to-end latency analysis (link delay plus inference), robustness checks, final report.

Known limitations: the feature ranking is supervised on the tune set, so it may be optimistic for unseen attack families; timings on a laptop are noisy, so differences under about 25% should be treated as ties; slice profiles are illustrative assumptions, not measurements of a real 5G core.

## References

- Y. Meidan et al., "N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders," IEEE Pervasive Computing, 2018.
- Y. Mirsky et al., "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection," NDSS, 2018.
- H. Wasswa, H. Abbass, T. Lynar, "A Quantized VAE-MLP Botnet Detection Model," arXiv:2511.03201, 2025.
- R. Dangi et al., "ML-Based 5G Network Slicing Security: A Comprehensive Survey," Future Internet 14(4):116, 2022.
