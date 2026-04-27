# PQ-FL-IDS: Post-Quantum Safe Federated Intrusion Detection for 6G Edge Networks

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)
[![Flower](https://img.shields.io/badge/Flower-v1.7.0-green.svg)](https://flower.dev/)
[![NIST FIPS 203](https://img.shields.io/badge/PQC-ML--KEM--512%20(FIPS%20203)-purple.svg)](https://csrc.nist.gov/pubs/fips/203/final)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Submitted to IEEE GLOBECOM 2026 — CISS Symposium**  
> *PQ-FL-IDS: A Post-Quantum Safe Federated Edge Intrusion Detection Scheme underlying 6G*  
> Husain Lucky, Pronaya Bhattacharya, Bharat Bhushan, Nishat Mahdiya Khan, Thippa Reddy Gadekallu

---

## Overview

PQ-FL-IDS is the **first federated IDS to combine real (non-simulated) ML-KEM-512 post-quantum cryptography with Differential Privacy-Stochastic Gradient Descent (DP-SGD)**, validated across two IoT security benchmarks targeting 6G edge deployment.

In 6G edge networks, federated learning allows intrusion detection models to train across distributed edge nodes without centralising raw traffic. However, the model updates exchanged during federation remain vulnerable to **harvest-now-decrypt-later** attacks from future quantum-capable adversaries. PQ-FL-IDS addresses this with:

- **ML-KEM-512** (NIST FIPS 203) hybrid encryption of every model update
- **Per-round ephemeral keypairs** providing Perfect Forward Secrecy (PFS)
- **DP-SGD** with formal Rényi privacy accounting to bound sample-level leakage
- **<0.003% cryptographic overhead** relative to training time — negligible deployment cost

---

## Key Results

All results are averaged over **5 independent runs × 40 FL rounds × 3 conditions**:

| Condition | Description |
|-----------|-------------|
| **A** | Baseline FL (no encryption, no DP) |
| **B** | ML-KEM-512 + AES-256-GCM |
| **C** | ML-KEM-512 + AES-256-GCM + DP-SGD (σ=1.1, ε≤3) |

### ToN-IoT — 1D-CNN + FedAvg (4-class)

| Metric | A | B | C |
|--------|---|---|---|
| Accuracy (R40) | 0.9755 ± 0.0005 | 0.9755 ± 0.0005 | 0.9670 ± 0.0010 |
| PQC overhead | — | 0.460 ± 0.040 ms | 0.500 ± 0.058 ms |
| PQC overhead (%) | — | 0.002% | 0.002% |
| ε (R40) | — | — | 1.263 |

### CICIoT2023 — MobileNetV2-1D + FedProx (7-class)

| Metric | A | B | C |
|--------|---|---|---|
| Accuracy (R40) | 0.7612 ± 0.0071 | 0.7699 ± 0.0024 | 0.4199 ± 0.0369 |
| PQC overhead | — | 1.497 ± 1.107 ms | 0.989 ± 0.194 ms |
| PQC overhead (%) | — | 0.001% | 0.001% |
| ε (R40) | — | — | 0.912 |

### Scalability (5-client, ToN-IoT)

PQC overhead remains at **0.002%** at 5 clients, confirming the cryptographic layer scales with federation size. ε grows from 1.263 (2-client) to 3.071 (5-client) due to increased subsampling rate q — expected behaviour under subsampling amplification, not a deployment limitation.

### Privacy–Utility Tradeoff (CICIoT2023)

| σ | ε (R40) | Accuracy | Privacy Regime |
|---|---------|----------|----------------|
| 1.1 | 0.912 | 0.420 ± 0.037 | Strong (ε < 3) ✅ |
| 0.8 | 2.412 | 0.490 ± 0.035 | Strong (ε < 3) ✅ |
| 0.5 | 12.554 | 0.530 ± 0.034 | Weak |
| 0.3 | 139.484 | 0.586 ± 0.016 | Nominal only |

**Recommended operating range: σ = 0.8–1.1, ε = 0.9–2.4.**

---

## Architecture

Each FL round follows four stages:

```
Server                              Client k
──────                              ────────
KeyGen() → (pk_t, sk_t)
Broadcast pk_t, w(t)        ──────►
                                    Train with DP-SGD
                                    Encaps(pk_t) → (ct_k, ss_k)
                                    HKDF-SHA256(ss_k) → sym_key
                                    AES-256-GCM.Encrypt(sym_key, w_k)
                            ◄──────  Send (ct_k, enc_k)
Decaps(sk_t, ct_k) → ss_k
Decrypt → w_k
Aggregate (FedAvg / FedProx)
DELETE sk_t  ← PFS
Update RDP accountant
```

DP-SGD is applied **before** encryption — the payload is already privacy-bounded before entering the communication channel.

---

## Tech Stack

| Component | Choice | Version |
|-----------|--------|---------|
| Language | Python | 3.10 (strict — liboqs incompatible with 3.12) |
| Federated Learning | Flower / gRPC | v1.7.0+ |
| ML Framework | TensorFlow / Keras | 2.15.0 |
| Post-Quantum Crypto | liboqs (C library) | v0.15.0-rc1 |
| PQC Python bindings | liboqs-python | v0.14.0 |
| PQC Algorithm | ML-KEM-512 (NIST FIPS 203) | — |
| Symmetric cipher | AES-256-GCM (cryptography.hazmat) | — |
| Differential Privacy | tensorflow-privacy | 0.9.0 |
| Privacy Accounting | `dp_accounting.rdp.RdpAccountant` | — |
| Hardware | AMD Ryzen 9 6900HX, 16 GB DDR5 | CPU-only† |

†GPU present but intentionally unused — CPU-only configuration simulates resource-constrained edge IoT deployment.

> **Note:** liboqs C v0.15.0-rc1 and Python bindings v0.14.0 have a known version mismatch. Both implement NIST FIPS 203 final standard and the mismatch is non-blocking.

---

## Datasets

Neither dataset is included in this repository. Download and preprocess separately.

### ToN-IoT
- Source: [UNSW Sydney ToN-IoT](https://research.unsw.edu.au/projects/toniot-datasets)
- 36,714 rows, 19 features, 4 classes
- Balanced to 10,000 samples/class
- Preprocess: `python preprocess.py`

### CICIoT2023
- Source: [CIC IoT Dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- 70,000 rows, 39 features, 7 classes
- Balanced to 10,000 samples/class
- Preprocess: `python preprocess_ciciot2023.py`

Both preprocessing scripts use **Dask streaming** to respect RAM constraints — never load the full dataset into memory.

---

## Installation

### Prerequisites

- Python 3.10 (strictly — not 3.11 or 3.12)
- liboqs C library v0.15.0-rc1 built and installed
- WSL2 Ubuntu 22.04 recommended (development environment)

### liboqs C Library

```bash
sudo apt install cmake gcc ninja-build libssl-dev
git clone --branch 0.15.0-rc1 https://github.com/open-quantum-safe/liboqs.git
cd liboqs && mkdir build && cd build
cmake -GNinja -DOQS_MINIMAL_BUILD=ON ..
ninja && sudo ninja install
```

### Python Environment

```bash
python3.10 -m venv venv_pq
source venv_pq/bin/activate
pip install -r requirements.txt
pip install liboqs-python==0.14.0
```

### Verify PQC Setup

```bash
cd src
export PYTHONPATH=.
python -c "import oqs; kem = oqs.KeyEncapsulation('ML-KEM-512'); print('ML-KEM-512 OK')"
```

---

## Usage

All commands run from the `src/` directory with `PYTHONPATH=.`:

```bash
cd src
export PYTHONPATH=.
```

### Single Benchmark Run

```bash
# ToN-IoT, 2 clients, 40 rounds
python benchmark_runner.py \
    --dataset toniot \
    --num-clients 2 \
    --num-rounds 40 \
    --noise-multiplier 1.1

# CICIoT2023, 2 clients, 40 rounds
python benchmark_runner.py \
    --dataset ciciot2023 \
    --num-clients 2 \
    --num-rounds 40 \
    --noise-multiplier 1.1
```

### Full 5-Run Benchmark (paper results)

```bash
nohup python -u benchmark_runner.py \
    --dataset toniot \
    --num-clients 2 \
    --num-rounds 40 \
    --num-runs 5 \
    --noise-multiplier 1.1 \
    > benchmark_toniot.log 2>&1 &
```

### Privacy–Utility Tradeoff Sweep

```bash
for sigma in 1.1 0.8 0.5 0.3; do
    python benchmark_runner.py \
        --dataset ciciot2023 \
        --num-clients 2 \
        --num-rounds 40 \
        --noise-multiplier $sigma
done
```

### Custom Data Path

```bash
python benchmark_runner.py \
    --dataset toniot \
    --data-path /path/to/custom/toniot_balanced.parquet \
    --noise-multiplier 1.1
```

---

## Project Structure

```
quantum-safe-ids/
├── src/
│   ├── benchmark_runner.py       # Orchestrates multi-run experiments
│   ├── server_node.py            # Flower server: keypair gen, aggregation, PFS
│   ├── fl_core/
│   │   └── edge_client.py        # Flower client: DP-SGD + ML-KEM encapsulation
│   ├── pqc_utils.py              # ML-KEM-512 + AES-256-GCM hybrid encryption
│   ├── kyber_manager.py          # Per-round ephemeral keypair management
│   ├── ids_cnn.py                # 1D-CNN for ToN-IoT
│   ├── mobilenet_ids.py          # MobileNetV2-1D (LayerNorm) for CICIoT2023
│   └── data_streamer.py          # Dask-based streaming data loader
├── notebooks/
│   └── 2_0-Feature-Engineering.ipynb
├── models/
│   └── v1_baselines/
├── preprocess.py                 # ToN-IoT preprocessing
├── preprocess_ciciot2023.py      # CICIoT2023 preprocessing
├── requirements.txt
└── .gitignore
```

---

## Reproducing Paper Results

The exact results in the paper require:

- 5 independent runs per condition
- 40 FL rounds per run
- 3 conditions: A (baseline), B (PQC only), C (PQC + DP-SGD)
- DP params: σ=1.1, clip C=1.0, δ=1e-5
- Privacy accounting: `dp_accounting.rdp.RdpAccountant` with `SelfComposedDpEvent`

> **Important:** Do not use `compute_dp_sgd_privacy` — it is broken in tensorflow-privacy 0.9.x. All privacy accounting uses `RdpAccountant` directly.

Expected runtimes on AMD Ryzen 9 6900HX (CPU-only):
- ToN-IoT: ~25s/round → ~17 hours for full 5×40×3 run
- CICIoT2023: ~150s/round → ~100 hours for full 5×40×3 run

Run with `nohup` and log output for overnight execution.

---

## Implementation Notes

### Why LayerNorm instead of BatchNorm

MobileNetV2-1D uses `LayerNormalization` throughout instead of `BatchNorm`. BatchNorm statistics couple per-sample gradients, which invalidates the per-sample clipping guarantee required by DP-SGD. In preliminary experiments, BatchNorm produced a **31% accuracy ceiling** under federated aggregation; LayerNorm resolved this and reached 74.7% by round 15.

### Why ε scales with client count

With fixed DP hyperparameters, ε grows faster at 5 clients (ε=3.071) than at 2 clients (ε=1.263). This is caused by the larger subsampling rate q = batch_size / per_client_data_size — more clients means smaller per-client partitions, which increases q and weakens subsampling amplification. In physical 6G deployments where each device generates independent local traffic, per-device q remains stable regardless of federation size.

### Perfect Forward Secrecy

The server generates a **fresh ML-KEM-512 keypair every round** and deletes the private key immediately after aggregation. Compromise of one round's key reveals nothing about past or future rounds.

---

## Comparison with Related Work

| Framework | Real PQC | DP | PFS | PQC Overhead |
|-----------|----------|-----|-----|--------------|
| **PQ-FL-IDS (ours)** | ✅ ML-KEM-512 | ✅ ε=0.9–2.4 | ✅ | **0.002%** |
| Nayak et al. (SPCSJ 2025) | ✅ | ✅ | ❌ | 18.7% |
| Beskar / Zhang et al. (2024) | ❌ theoretical | ✅ | ❌ | not reported |
| Edge-QSFL (IEEE TCE 2026) | ❌ Qiskit-simulated | ✅ | ❌ | not reported |

PQ-FL-IDS achieves **9,350× lower PQC overhead** than Nayak et al. (0.002% vs 18.7%).

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{lucky2026pqflids,
  title     = {{PQ-FL-IDS}: A Post-Quantum Safe Federated Edge Intrusion Detection
               Scheme underlying 6G},
  author    = {Lucky, Husain and Bhattacharya, Pronaya and Bhushan, Bharat and
               Khan, Nishat Mahdiya and Gadekallu, Thippa Reddy},
  booktitle = {Proc. IEEE Global Communications Conference (GLOBECOM)},
  year      = {2026},
  note      = {CISS Symposium}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Open Quantum Safe (liboqs)](https://openquantumsafe.org/) for ML-KEM-512 implementation
- [Flower Federated Learning Framework](https://flower.dev/)
- [TensorFlow Privacy](https://github.com/tensorflow/privacy) for DP-SGD
- Supervisor: Dr. Pronaya Bhattacharya, Amity University Kolkata
