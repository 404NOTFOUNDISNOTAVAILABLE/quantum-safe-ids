"""
preprocess_ciciot2023.py — Reproducible CICIoT2023 preprocessing for PQ-FLIDS.

Reads all 63 Merged*.csv files from data/ciciot2023/, maps fine-grained
attack labels to 7 coarse classes, applies stratified sampling, fits
StandardScaler on the sample, and writes ciciot2023_sampled.parquet.

Design decisions:
- Dask reads CSVs lazily — never loads the full dataset (~10M+ rows) into RAM.
- Label lookup is case-normalised (both map keys and data labels are uppercased)
  to survive the mixed-case inconsistency between the dataset and the map.
  Unmapped labels after normalisation trigger a printed warning.
- Stratified sampling: 10,000/class target, min 500/class.
  WebAttack (rare) takes all available rows without downsampling.
- StandardScaler is fit only on the final sample, not the full dataset.
- Zero-variance columns are detected on the sample and dropped.
- Label encoding: Benign = 0, rest alphabetically.

Usage (from ~/pq-flids/):
    python preprocess_ciciot2023.py
    python preprocess_ciciot2023.py --input-dir data/ciciot2023 \\
                                    --output data/ciciot2023_sampled.parquet \\
                                    --samples-per-class 10000 \\
                                    --min-samples 500
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Label mapping — fine-grained CICIoT2023 labels → 7 coarse classes
# Keys are stored UPPERCASE internally for case-insensitive lookup.
# ---------------------------------------------------------------------------

LABEL_MAP: Dict[str, str] = {
    # DDoS
    "DDOS-ACK_FRAGMENTATION": "DDoS",
    "DDOS-HTTP_FLOOD": "DDoS",
    "DDOS-ICMP_FLOOD": "DDoS",
    "DDOS-ICMP_FRAGMENTATION": "DDoS",
    "DDOS-PSHACK_FLOOD": "DDoS",
    "DDOS-RSTFINFLOOD": "DDoS",
    "DDOS-SLOWLORIS": "DDoS",
    "DDOS-SYN_FLOOD": "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDoS",
    "DDOS-TCP_FLOOD": "DDoS",
    "DDOS-UDP_FLOOD": "DDoS",
    "DDOS-UDP_FRAGMENTATION": "DDoS",
    # DoS
    "DOS-HTTP_FLOOD": "DoS",
    "DOS-SYN_FLOOD": "DoS",
    "DOS-TCP_FLOOD": "DoS",
    "DOS-UDP_FLOOD": "DoS",
    # Mirai
    "MIRAI-GREETH_FLOOD": "Mirai",
    "MIRAI-GREIP_FLOOD": "Mirai",
    "MIRAI-UDPPLAIN": "Mirai",
    # Recon
    "RECON-HOSTDISCOVERY": "Recon",
    "RECON-OSSCAN": "Recon",
    "RECON-PINGSWEEP": "Recon",
    "RECON-PORTSCAN": "Recon",
    # Spoofing
    "MITM-ARPSPOOFING": "Spoofing",
    "DNS_SPOOFING": "Spoofing",
    # WebAttack (rare classes grouped together)
    "SQLINJECTION": "WebAttack",
    "COMMANDINJECTION": "WebAttack",
    "XSS": "WebAttack",
    "UPLOADING_ATTACK": "WebAttack",
    "BROWSERHIJACKING": "WebAttack",
    "BACKDOOR_MALWARE": "WebAttack",
    "DICTIONARYBRUTEFORCE": "WebAttack",
    "VULNERABILITYSCAN": "WebAttack",
    # Benign
    "BENIGN": "Benign",
}


def map_label(raw) -> str:
    """Maps a raw CICIoT2023 label to a coarse class. Returns '' if unmapped or null."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    return LABEL_MAP.get(str(raw).strip().upper(), "")


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def build_label_encoder(labels: pd.Series) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Benign = 0, rest sorted alphabetically."""
    unique = sorted(labels.unique())
    if "Benign" in unique:
        unique.remove("Benign")
        ordered = ["Benign"] + unique
    else:
        ordered = unique
    str_to_int = {label: i for i, label in enumerate(ordered)}
    int_to_str = {i: label for label, i in str_to_int.items()}
    return str_to_int, int_to_str


# ---------------------------------------------------------------------------
# Feature utilities
# ---------------------------------------------------------------------------

def drop_zero_variance_columns(
    df: pd.DataFrame, exclude: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Drops columns where all values are identical."""
    feature_cols = [c for c in df.columns if c not in exclude]
    dropped = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    return df.drop(columns=dropped), dropped


# ---------------------------------------------------------------------------
# Stratified sampling via Dask (RAM-safe)
# ---------------------------------------------------------------------------

def sample_class(
    ddf: dd.DataFrame,
    coarse_label: str,
    n_target: int,
    take_all: bool,
    random_state: int,
) -> pd.DataFrame:
    """
    Filters `ddf` to rows matching `coarse_label`, then collects up to
    `n_target` rows (or all rows if `take_all` is True).

    Dask evaluates only the filtered partition subset, so the full dataset
    is never materialised in RAM simultaneously.
    """
    subset = ddf[ddf["label_str"] == coarse_label]
    if take_all:
        return subset.compute()
    # Compute just enough: use random_state seed on each partition, then trim.
    # Dask sample() is approximate — we over-sample slightly then trim.
    frac = min(1.0, (n_target * 2) / max(subset.shape[0].compute(), 1))
    approx = subset.sample(frac=frac, random_state=random_state).compute()
    if len(approx) >= n_target:
        return approx.sample(n=n_target, random_state=random_state)
    return approx  # under-target: return all available


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PQ-FLIDS CICIoT2023 Preprocessor")
    parser.add_argument("--input-dir",  type=str, default="data/ciciot2023",
                        help="Directory containing Merged*.csv files")
    parser.add_argument("--output", type=str, default="data/ciciot2023_sampled.parquet",
                        help="Output parquet path")
    parser.add_argument("--label-map", type=str, default="data/ciciot2023_label_map.json",
                        help="Output JSON path for label encoding")
    parser.add_argument("--samples-per-class", type=int, default=10_000,
                        help="Target rows per class (default: 10000)")
    parser.add_argument("--min-samples", type=int, default=500,
                        help="Minimum rows for any class present (default: 500)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_path = Path(args.output)
    label_map_path = Path(args.label_map)

    csv_files = sorted(input_dir.glob("Merged*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No Merged*.csv files found in {input_dir}")
    print(f"[preprocess] Found {len(csv_files)} CSV files in {input_dir}")

    # ------------------------------------------------------------------
    # Pass 1 — Peek at one file to get schema
    # ------------------------------------------------------------------
    sample_peek = pd.read_csv(csv_files[0], nrows=100)
    print(f"[preprocess] Columns ({len(sample_peek.columns)}): {sample_peek.columns.tolist()}")

    if "Label" not in sample_peek.columns:
        raise ValueError("Expected 'Label' column not found in CSV files.")

    # ------------------------------------------------------------------
    # Pass 2 — Load with Dask (lazy, no RAM spike)
    # ------------------------------------------------------------------
    print("[preprocess] Building Dask dataframe (lazy read) ...")
    ddf = dd.read_csv([str(f) for f in csv_files], assume_missing=True)

    # Apply label mapping — map unknown labels to empty string
    ddf["label_str"] = ddf["Label"].map(map_label, meta=("label_str", "str"))

    # ------------------------------------------------------------------
    # Pass 3 — Count unmapped labels and raw class distribution
    # ------------------------------------------------------------------
    print("[preprocess] Computing label distribution (triggers CSV scan) ...")
    label_counts_raw = ddf["Label"].value_counts().compute().sort_values(ascending=False)
    print("\n[preprocess] Raw label counts (all files):")
    for lbl, cnt in label_counts_raw.items():
        mapped = map_label(str(lbl))
        tag = f"→ {mapped}" if mapped else "  *** UNMAPPED ***"
        print(f"  {cnt:>10,}  {lbl:<40}  {tag}")

    unmapped = [lbl for lbl in label_counts_raw.index if not map_label(str(lbl))]
    if unmapped:
        print(f"\n[preprocess] WARNING: {len(unmapped)} unmapped label(s): {unmapped}")
        print("[preprocess]   These rows will be EXCLUDED from output.")
    else:
        print("\n[preprocess] ✓ All labels mapped successfully.")

    # Drop rows with unmapped labels
    ddf = ddf[ddf["label_str"] != ""]

    # ------------------------------------------------------------------
    # Pass 4 — Stratified sampling per coarse class
    # ------------------------------------------------------------------
    coarse_counts = ddf["label_str"].value_counts().compute()
    print(f"\n[preprocess] Coarse class counts before sampling:")
    for cls, cnt in coarse_counts.sort_values(ascending=False).items():
        print(f"  {cnt:>10,}  {cls}")

    print(f"\n[preprocess] Stratified sampling "
          f"(target: {args.samples_per_class:,}/class, min: {args.min_samples:,}/class) ...")

    parts: List[pd.DataFrame] = []
    for coarse_label, available in coarse_counts.items():
        n_target = min(args.samples_per_class, available)
        n_target = max(n_target, min(args.min_samples, available))

        print(f"  Sampling {coarse_label}: target={n_target:,}, available={available:,} ...",
              end=" ", flush=True)
        chunk = sample_class(ddf, coarse_label, n_target, take_all=False, random_state=args.seed)
        print(f"got {len(chunk):,}")
        parts.append(chunk)

    df = pd.concat(parts, ignore_index=True)
    print(f"\n[preprocess] Sample assembled: {len(df):,} rows")

    # ------------------------------------------------------------------
    # Feature engineering on sample
    # ------------------------------------------------------------------
    # Drop the original Label column (replaced by label_str → integer)
    df = df.drop(columns=["Label"], errors="ignore")

    # Encode remaining object/category columns (e.g. Protocol Type)
    exclude_cols = ["label_str"]
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col in exclude_cols:
            continue
        df[col] = pd.Categorical(df[col]).codes.astype(np.float32)

    # Cast features to float32
    feature_cols = [c for c in df.columns if c != "label_str"]
    df[feature_cols] = df[feature_cols].astype(np.float32)

    # Replace ±inf with NaN (common in network flow features, e.g. divide-by-zero rates)
    inf_total = np.isinf(df[feature_cols].values).sum()
    if inf_total > 0:
        print(f"[preprocess] Replacing {inf_total:,} ±inf values with NaN")
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Fill NaN
    nan_total = df[feature_cols].isna().sum().sum()
    if nan_total > 0:
        print(f"[preprocess] Filling {nan_total:,} NaN values with 0.0")
        df[feature_cols] = df[feature_cols].fillna(0.0)

    # Drop zero-variance columns (detected on the sample)
    df, dropped_zv = drop_zero_variance_columns(df, exclude=["label_str"])
    feature_cols = [c for c in df.columns if c != "label_str"]
    if dropped_zv:
        print(f"[preprocess] Dropped zero-variance columns ({len(dropped_zv)}): {dropped_zv}")
    else:
        print(f"[preprocess] No zero-variance columns found.")
    print(f"[preprocess] Remaining features: {len(feature_cols)}")

    # ------------------------------------------------------------------
    # StandardScaler — fit on sample only
    # ------------------------------------------------------------------
    print("[preprocess] Fitting StandardScaler on sample ...")
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols]).astype(np.float32)
    print(f"[preprocess] ✓ Features scaled (mean≈0, std≈1)")

    # ------------------------------------------------------------------
    # Label encoding
    # ------------------------------------------------------------------
    str_to_int, int_to_str = build_label_encoder(df["label_str"])
    df["label"] = df["label_str"].map(str_to_int).astype(np.int64)
    df = df.drop(columns=["label_str"])

    # Shuffle
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Final class distribution
    # ------------------------------------------------------------------
    print(f"\n[preprocess] Class distribution in output:")
    for label_int, count in df["label"].value_counts().sort_index().items():
        label_name = int_to_str[int(label_int)]
        print(f"  [{label_int}] {label_name:<15} {count:>7,}  ({count/len(df)*100:.1f}%)")

    # Sanity check: no zero-variance in output
    zv_check = [c for c in feature_cols if df[c].nunique() <= 1]
    if zv_check:
        print(f"[preprocess] WARNING: Zero-variance columns still present: {zv_check}")
    else:
        print(f"[preprocess] ✓ All {len(feature_cols)} feature columns have variance > 0")

    # ------------------------------------------------------------------
    # Save label map
    # ------------------------------------------------------------------
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w") as f:
        json.dump(
            {
                "str_to_int": str_to_int,
                "int_to_str": {str(k): v for k, v in int_to_str.items()},
                "coarse_label_map": LABEL_MAP,
            },
            f,
            indent=2,
        )
    print(f"\n[preprocess] Label encoding saved to {label_map_path}")
    print(f"  Mapping: {str_to_int}")

    # ------------------------------------------------------------------
    # Save parquet
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_path, compression="snappy")

    file_size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"\n[preprocess] ✓ Written to {output_path}")
    print(f"[preprocess]   Rows:    {len(df):,}")
    print(f"[preprocess]   Columns: {len(df.columns)} ({len(feature_cols)} features + 1 label)")
    print(f"[preprocess]   Size:    {file_size_mb:.1f} MB")

    num_classes = len(str_to_int)
    print(f"\n[preprocess] ⚠  num_classes = {num_classes}")
    print(f"[preprocess]    Update IntrusionDetectionCNN(num_classes={num_classes}) in edge_client.py")


if __name__ == "__main__":
    main()
