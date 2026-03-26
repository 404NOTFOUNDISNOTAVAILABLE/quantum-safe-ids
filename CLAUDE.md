# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Quantum-Safe Federated Intrusion Detection System (IDS) for 6G Edge Networks** — a proof-of-concept implementing privacy-preserving, quantum-safe network intrusion detection using Federated Learning (FL) with post-quantum cryptography (PQC). Detects DoS, DDoS, C2, and Port Scanning attacks (~91.8% accuracy). Dataset: ToN-IoT (Parquet format).

## Commands

### Environment Setup
```bash
pip install -r requirements.txt
# liboqs-python must be installed from the bundled submodule for ML-KEM-512 support
```

### Data Preprocessing
```bash
python preprocess.py   # Converts raw ToN-IoT data → data/sampled_data.parquet
```

### Tests
```bash
cd src/
PYTHONPATH=. python test_pipeline.py   # Data streamer + CNN integration
PYTHONPATH=. python test_crypto.py     # ML-KEM-512 + AES-GCM encryption
PYTHONPATH=. python test_pqc.py        # PQC utility functions
```

### Run a Single FL Session (3 terminals)
```bash
cd src/

# Terminal 1: Server
PYTHONPATH=. python fl_core/server_node.py \
  --rounds 20 --csv-log ../results/session.csv \
  --condition "test_run" --run 1

# Terminals 2 & 3: Clients (add --dp --noise 1.1 --clip 1.0 for differential privacy)
PYTHONPATH=. python fl_core/edge_client.py --id 0 --clients 2 --rounds 20
PYTHONPATH=. python fl_core/edge_client.py --id 1 --clients 2 --rounds 20
```

### Run Automated Benchmark
```bash
cd src/
PYTHONPATH=. python -u benchmark_runner.py                      # 20 runs × 3 conditions (default)
PYTHONPATH=. python -u benchmark_runner.py --runs 5 --rounds 20 # Custom
PYTHONPATH=. python -u benchmark_runner.py --start-session 10 --csv ../results/existing.csv  # Resume
```

## Architecture

The system has four layers that work together across an FL training round:

### 1. Data Layer (`src/data/data_streamer.py`)
`ParquetDataStreamer` streams ToN-IoT Parquet data without loading into RAM (edge-friendly). Partitions data across clients deterministically via modulo on row index. Returns `tf.data.Dataset` with batch_size=32. Feature columns are auto-inferred from schema; target column is `label` (integer-encoded via `data/label_map.json`).

### 2. Model Layer (`src/model/ids_cnn.py`)
`IntrusionDetectionCNN`: 1D-CNN with Conv1D(64→128) → BatchNorm → MaxPool → Dropout → Dense → Softmax (4 classes). Optionally replaces Adam with `DPKerasAdamOptimizer` for DP-SGD (gradient clipping + Gaussian noise injection).

### 3. Cryptography Layer (`src/crypto/kyber_manager.py`)
`KyberManager` implements hybrid PQC encryption:
- **Server**: generates an ephemeral ML-KEM-512 keypair each round (perfect forward secrecy)
- **Client**: encapsulates shared secret with server's public key, then encrypts serialized weights with AES-GCM (key derived via SHA-256 of shared secret)
- **Server**: decapsulates with secret key, decrypts weights

### 4. Federated Learning Layer (`src/fl_core/`)
- **`server_node.py`** — `SecureQuantumStrategy` extends Flower `FedAvg`:
  1. `configure_fit`: broadcasts fresh ML-KEM-512 public key to all clients
  2. `aggregate_fit`: decrypts each client's payload, runs FedAvg, logs per-round metrics to CSV
  3. `aggregate_evaluate`: computes sample-weighted global accuracy/loss
- **`edge_client.py`** — `QuantumEdgeClient` extends Flower `NumPyClient`:
  1. Receives global weights, trains locally for 1 epoch
  2. Encrypts updated weights using server's public key
  3. Tracks cumulative DP epsilon via RDP accountant, returns metrics

### Benchmarking (`src/benchmark_runner.py`)
Automates 60 sessions across 3 conditions (A: no PQC/DP; B: PQC only; C: PQC+DP) by spawning server + 2 client subprocesses, capturing logs, and appending metrics to a CSV. Measures `train_time_ms`, `pqc_total_overhead_ms`, `encrypted_payload_bytes`, `epsilon_consumed`, and accuracy per round per client.

## Key Configuration Details

- **Clients**: hardcoded to 2 in data partitioning logic; all clients participate each round (`fraction_fit=1.0`)
- **Local training**: 1 epoch per FL round
- **DP parameters**: `l2_norm_clip=1.0`, `noise_multiplier=1.1`, `delta=1e-5`, `num_microbatches=1`
- **PQC algorithm**: ML-KEM-512 (NIST-standardized Kyber-512) via `liboqs-python` submodule
- **`src/pqc_utils.py`**: alternative/standalone PQC utilities, separate from `kyber_manager.py`
- **`src/v1_archive/`**: deprecated v1 implementation, kept for reference only
