"""
benchmark_runner.py — Automated 60-session benchmark for PQ-FLIDS.

Conditions:
  A_baseline  — No PQC, No DP
  B_pqc_only  — ML-KEM-512 + AES-GCM, no DP
  C_pqc_dp    — ML-KEM-512 + AES-GCM + DP-SGD

Runs RUNS_PER_CONDITION sessions per condition (default 20).
Each session runs ROUNDS FL communication rounds (default 40).
Metrics are logged to a single CSV by the server, then summarised here.

Usage (from src/):
    PYTHONPATH=. python -u benchmark_runner.py
    PYTHONPATH=. python -u benchmark_runner.py --runs 5 --rounds 20

Resume a partial run (skip sessions 1-9, redo from session 10):
    PYTHONPATH=. python -u benchmark_runner.py --runs 5 --rounds 40 \
        --start-session 10 --csv ../results/benchmark_TIMESTAMP.csv
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime

PYTHON = sys.executable
RESULTS_DIR = Path("../results")

CONDITIONS = [
    {
        "name":         "A_baseline",
        "label":        "No PQC, No DP",
        "server_flags": ["--no-pqc"],
        "client_flags": [],
    },
    {
        "name":         "B_pqc_only",
        "label":        "PQC only",
        "server_flags": [],
        "client_flags": [],
    },
    {
        "name":         "C_pqc_dp",
        "label":        "PQC + DP",
        "server_flags": [],
        "client_flags": ["--dp"],
    },
]


def wait_for_server(log_path: Path, timeout: int = 30) -> bool:
    """Poll server log until gRPC ready line appears or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if log_path.exists():
            text = log_path.read_text()
            if "gRPC server running" in text:
                return True
        time.sleep(0.5)
    return False


def run_session(condition: dict, run_idx: int, rounds: int,
                csv_path: Path, log_dir: Path, dataset: str = "toniot",
                model: str = "cnn", fedprox: bool = False,
                clients: int = 2, local_epochs: int = 2) -> bool:
    """
    Runs one complete FL session (server + N clients).
    Returns True if the server exited cleanly (returncode 0).
    """
    tag = f"{condition['name']}_run{run_idx:02d}"
    server_log = log_dir / f"{tag}_server.log"
    client_logs = [log_dir / f"{tag}_client{cid}.log" for cid in range(clients)]

    # --- Server ---
    server_cmd = [
        PYTHON, "-u", "fl_core/server_node.py",
        "--rounds",    str(rounds),
        "--csv-log",   str(csv_path),
        "--condition", condition["name"],
        "--run",       str(run_idx),
    ] + condition["server_flags"]
    if fedprox:
        server_cmd.append("--fedprox")
    server_cmd += ["--clients", str(clients)]
    server_cmd += ["--local-epochs", str(local_epochs)]

    with open(server_log, "w") as sf:
        server_proc = subprocess.Popen(
            server_cmd, stdout=sf, stderr=sf,
            env={**os.environ, "PYTHONPATH": "."},
        )

    if not wait_for_server(server_log):
        print(f"  [ERROR] Server did not start in time for {tag}.")
        server_proc.kill()
        return False

    # --- Clients ---
    client_procs = []
    for cid in range(clients):
        client_cmd = [
            PYTHON, "-u", "fl_core/edge_client.py",
            "--id",      str(cid),
            "--clients", str(clients),
            "--rounds",  str(rounds),
            "--dataset", dataset,
        ] + condition["client_flags"]
        client_cmd += ["--model", model]
        client_cmd += ["--local-epochs", str(local_epochs)]
        with open(client_logs[cid], "w") as cf:
            proc = subprocess.Popen(
                client_cmd, stdout=cf, stderr=cf,
                env={**os.environ, "PYTHONPATH": "."},
            )
        client_procs.append(proc)
        time.sleep(0.3)  # stagger starts slightly

    # --- Wait ---
    try:
        for proc in client_procs:
            proc.wait(timeout=14400)
        server_proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Timeout in {tag}.")
        for proc in client_procs:
            proc.kill()
        server_proc.kill()
        return False

    # Brief pause to ensure port is released before next session
    time.sleep(3)
    return server_proc.returncode == 0


def load_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def summarise(rows: list[dict]) -> None:
    """Compute and print mean ± std per condition for key metrics."""
    METRICS = [
        ("train_time_ms",          "Train time (ms/round)"),
        ("pqc_total_overhead_ms",  "PQC total overhead (ms)"),
        ("pqc_encap_ms",           "ML-KEM encap (ms)"),
        ("sym_enc_ms",             "AES-GCM enc (ms)"),
        ("encrypted_payload_bytes","Encrypted payload (bytes)"),
        ("plaintext_payload_bytes","Plaintext payload (bytes)"),
        ("epsilon_consumed",       "ε (cumulative, final round)"),
        ("global_accuracy",        "Global accuracy (final round)"),
        ("global_loss",            "Global loss (final round)"),
        ("eval_accuracy",          "Per-client eval accuracy"),
    ]

    # Final-round-only columns: take max server_round value per run
    FINAL_ROUND_COLS = {"epsilon_consumed", "global_accuracy", "global_loss"}

    # Bucket values per condition × metric
    data: dict = defaultdict(lambda: defaultdict(list))
    final_vals: dict = defaultdict(dict)  # (cond, run, col) → (max_round, val)

    for row in rows:
        cond = row["condition"]
        run  = row["run"]
        try:
            rnd = int(row["server_round"])
        except (ValueError, KeyError):
            continue

        for col, _ in METRICS:
            if col in FINAL_ROUND_COLS:
                continue
            try:
                val = float(row[col])
                data[cond][col].append(val)
            except (ValueError, KeyError):
                pass

        # Track final-round value per (condition, run, col)
        for col in FINAL_ROUND_COLS:
            try:
                val = float(row[col])
                key = (cond, run, col)
                prev_rnd, _ = final_vals.get(key, (-1, None))
                if rnd > prev_rnd:
                    final_vals[key] = (rnd, val)
            except (ValueError, KeyError):
                pass

    # Add final-round values to data
    for (cond, _run, col), (_, val) in final_vals.items():
        if col == "epsilon_consumed" and val < 0:
            continue
        data[cond][col].append(val)

    # Print table
    col_w = 32
    def _mean_std(vals):
        if not vals:
            return "n/a"
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = variance ** 0.5
            return f"{mean:.3f} ± {std:.3f}"
        return f"{mean:.3f}"

    cond_names = [c["name"] for c in CONDITIONS]
    cond_labels = {c["name"]: c["label"] for c in CONDITIONS}

    header = f"{'Metric':<{col_w}}" + "".join(f"{cond_labels[c]:>28}" for c in cond_names)
    print("\n" + "=" * (col_w + 28 * len(cond_names)))
    print("BENCHMARK RESULTS — mean ± std")
    print("=" * (col_w + 28 * len(cond_names)))
    print(header)
    print("-" * (col_w + 28 * len(cond_names)))
    for col, label in METRICS:
        row_str = f"{label:<{col_w}}"
        for cond in cond_names:
            row_str += f"{_mean_std(data[cond][col]):>28}"
        print(row_str)
    print("=" * (col_w + 28 * len(cond_names)))


def main() -> None:
    parser = argparse.ArgumentParser(description="PQ-FLIDS Benchmark Runner")
    parser.add_argument("--runs",          type=int,  default=20,   help="Repetitions per condition")
    parser.add_argument("--rounds",        type=int,  default=40,   help="FL rounds per session")
    parser.add_argument("--start-session", type=int,  default=1,    help="Skip sessions before this 1-based index (resume support)")
    parser.add_argument("--csv",           type=str,  default=None, help="Append to an existing CSV instead of creating a new timestamped one")
    parser.add_argument("--dataset",       type=str,  default="toniot",
                        choices=["toniot", "ciciot2023"],
                        help="Dataset to use for training (default: toniot)")
    parser.add_argument("--conditions", type=str, default="A,B,C",
                        help="Comma-separated conditions to run e.g. A,B or C")
    parser.add_argument("--model", type=str, default="cnn",
                        choices=["cnn", "mobilenet"],
                        help="Model architecture: cnn or mobilenet")
    parser.add_argument("--fedprox", action="store_true",
                        help="Use FedProx aggregation instead of FedAvg")
    parser.add_argument("--clients", type=int, default=2,
                        help="Number of FL clients (default: 2)")
    parser.add_argument("--local-epochs", type=int, default=2,
                        help="Local epochs per round (default: 2)")
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        log_dir  = csv_path.with_suffix("")          # e.g. results/benchmark_TIMESTAMP
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir   = RESULTS_DIR / f"benchmark_{args.dataset}_{timestamp}"
        csv_path  = RESULTS_DIR / f"benchmark_{args.dataset}_{timestamp}.csv"
    log_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    selected = [c.strip() for c in args.conditions.split(",")]
    conditions = [c for c in CONDITIONS if any(
        c["name"].startswith(s) for s in selected
    )]

    total_sessions = len(conditions) * args.runs
    session_idx = 0

    print(f"PQ-FLIDS Benchmark: {len(conditions)} conditions × {args.runs} runs "
          f"× {args.rounds} rounds = {total_sessions} sessions  [dataset: {args.dataset}]")
    print(f"  Model:       {args.model.upper()}")
    print(f"  Aggregation: {'FedProx' if args.fedprox else 'FedAvg'}")
    print(f"  Clients:     {args.clients}")
    print(f"  Local epochs:{args.local_epochs}")
    if args.start_session > 1:
        print(f"Resuming from session {args.start_session} (skipping {args.start_session - 1} already-completed sessions)")
    print(f"CSV: {csv_path}")
    print(f"Logs: {log_dir}\n")

    for condition in conditions:
        print(f"\n{'─'*60}")
        print(f"Condition: {condition['name']} — {condition['label']}")
        print(f"{'─'*60}")
        for run_idx in range(1, args.runs + 1):
            session_idx += 1
            if session_idx < args.start_session:
                print(f"  Run {run_idx:2d}/{args.runs}  [{session_idx}/{total_sessions}] → SKIP")
                continue
            print(f"  Run {run_idx:2d}/{args.runs}  [{session_idx}/{total_sessions}]", end=" ", flush=True)
            t0 = time.time()
            ok = run_session(condition, run_idx, args.rounds, csv_path, log_dir, args.dataset,
                             model=args.model, fedprox=args.fedprox, clients=args.clients,
                             local_epochs=args.local_epochs)
            elapsed = time.time() - t0
            status = "OK" if ok else "FAIL"
            print(f"→ {status}  ({elapsed:.1f}s)")

    print(f"\n{'═'*60}")
    print(f"All sessions complete. Parsing {csv_path} ...")
    if csv_path.exists():
        rows = load_csv(csv_path)
        print(f"Total rows: {len(rows)}")
        summarise(rows)
    else:
        print("ERROR: No CSV file found. Check session logs in", log_dir)


if __name__ == "__main__":
    main()
