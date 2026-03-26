"""
preprocess.py — Reproducible ToN-IoT preprocessing for PQ-FLIDS.

Replaces the 2_0-Feature-Engineering.ipynb notebook.
Reads merged_data.parquet (raw Zeek flow records), parses labels,
drops zero-variance and identifier columns, applies stratified sampling,
and writes a clean sampled_data.parquet for FL training.

Design decisions (paper-relevant):
- Works on individual flow records, NOT time-window aggregations.
  Flow-level IDS is the standard approach in the literature and avoids
  the label-collapse bug caused by min() aggregation of string labels.
- Stratified sampling ensures all attack classes are represented even
  in the reduced dataset.
- Zero-variance features are dropped (14 constant columns identified
  in diagnosis — local_orig_*, missed_bytes_*, local_resp_*).
- Label encoding is deterministic and saved to label_map.json for
  reproducibility (reviewers can verify the class mapping).

Usage (from ~/pq-flids/):
    python preprocess.py
    python preprocess.py --input data/merged_data.parquet \\
                         --output data/sampled_data.parquet \\
                         --samples-per-class 10000 \\
                         --min-samples 500
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_zeek_label(raw_label: str) -> str:
    """
    Parses a raw Zeek/ToN-IoT label string into a clean attack-type string.

    ToN-IoT label format is one of:
      - "(empty)\\tBenign\\t-"  or  "Benign"
      - "(empty)\\tMalicious\\tPartOfAHorizontalPortScan"
      - "(empty)\\tMalicious\\tDDoS"
      - "(empty)\\tMalicious\\tC&C"
      etc.

    Returns the attack type string (e.g. "Benign", "PortScan", "DDoS", "C&C").

    Args:
        raw_label: Raw label string from the Zeek log.

    Returns:
        Cleaned attack type string.
    """
    if not isinstance(raw_label, str):
        return "Benign"

    raw = raw_label.strip()

    # Normalise delimiter: data uses 3 spaces, docs show tabs — handle both
    delimiter = "\t" if "\t" in raw else "   "

    if delimiter not in raw:
        if raw.lower() in ("", "-", "(empty)", "nan"):
            return "Benign"
        return raw

    # Delimited: "(empty)<delim>Benign<delim>-" or "(empty)<delim>Malicious<delim>AttackType"
    parts = [p.strip() for p in raw.split(delimiter)]
    # Find the meaningful parts (skip "(empty)" and "-")
    meaningful = [p for p in parts if p not in ("", "-", "(empty)")]

    if not meaningful:
        return "Benign"

    # If "Malicious" is present, return the attack type that follows it
    if "Malicious" in meaningful:
        idx = meaningful.index("Malicious")
        if idx + 1 < len(meaningful):
            return meaningful[idx + 1]
        return "Malicious_Unknown"

    # Normalise case on Benign variants
    meaningful = ["Benign" if p.lower() == "benign" else p for p in meaningful]

    # Otherwise return the last meaningful token
    return meaningful[-1]


def build_label_encoder(labels: pd.Series) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Builds a deterministic label encoder from the unique label strings.
    "Benign" is always encoded as 0. All other classes are sorted
    alphabetically and assigned 1, 2, 3, ...

    Args:
        labels: Series of cleaned label strings.

    Returns:
        Tuple of (str_to_int dict, int_to_str dict).
    """
    unique = sorted(labels.unique())
    # Ensure Benign = 0
    if "Benign" in unique:
        unique.remove("Benign")
        ordered = ["Benign"] + unique
    else:
        ordered = unique

    str_to_int = {label: i for i, label in enumerate(ordered)}
    int_to_str = {i: label for label, i in str_to_int.items()}
    return str_to_int, int_to_str


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def drop_zero_variance_columns(df: pd.DataFrame, exclude: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drops columns where all values are identical (zero variance).
    These provide no signal to the CNN and inflate the feature vector.

    Args:
        df: Input DataFrame.
        exclude: Column names to skip (labels, identifiers).

    Returns:
        Tuple of (cleaned DataFrame, list of dropped column names).
    """
    feature_cols = [c for c in df.columns if c not in exclude]
    dropped = []
    for col in feature_cols:
        if df[col].nunique(dropna=False) <= 1:
            dropped.append(col)
    df = df.drop(columns=dropped)
    return df, dropped


def drop_identifier_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drops columns that are identifiers or timestamps — not features.
    These would cause data leakage if included in training.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, list of dropped column names).
    """
    # Known identifier/timestamp columns in ToN-IoT Zeek logs
    identifier_patterns = [
        "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
        "ts", "time", "DatetimeIndex",
        # HTTP join columns that duplicate network-layer identifiers
        "id.orig_h_http", "id.orig_p_http", "id.resp_h_http", "id.resp_p_http",
        "ts_http",
    ]
    to_drop = [c for c in df.columns if c in identifier_patterns]
    df = df.drop(columns=to_drop, errors="ignore")
    return df, to_drop


def encode_categorical_columns(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    """
    Label-encodes any remaining string/object/categorical columns.
    This handles Zeek fields like 'proto', 'conn_state', 'service',
    'method', 'history', 'uri', 'host', etc.

    Args:
        df: Input DataFrame.
        exclude: Column names to skip.

    Returns:
        DataFrame with object columns replaced by integer codes.
    """
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col in exclude:
            continue
        df[col] = pd.Categorical(df[col]).codes.astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    df: pd.DataFrame,
    label_col: str,
    samples_per_class: int,
    min_samples: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Produces a stratified sample: up to `samples_per_class` rows per class,
    with a minimum of `min_samples` for minority classes (or all available
    rows if the class has fewer than `min_samples`).

    Args:
        df: Input DataFrame with integer label column.
        label_col: Name of the integer label column.
        samples_per_class: Target number of samples per class.
        min_samples: Minimum samples to include for any class present.
        random_state: Random seed for reproducibility.

    Returns:
        Stratified sample DataFrame, shuffled.
    """
    parts = []
    for cls, group in df.groupby(label_col):
        n = min(samples_per_class, len(group))
        n = max(n, min(min_samples, len(group)))
        parts.append(group.sample(n=n, random_state=random_state))
    result = pd.concat(parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PQ-FLIDS ToN-IoT Preprocessor")
    parser.add_argument("--input",  type=str, default="data/merged_data.parquet",
                        help="Path to merged_data.parquet")
    parser.add_argument("--output", type=str, default="data/sampled_data.parquet",
                        help="Output path for sampled_data.parquet")
    parser.add_argument("--label-map", type=str, default="data/label_map.json",
                        help="Output path for label encoding map (JSON)")
    parser.add_argument("--samples-per-class", type=int, default=10000,
                        help="Target rows per class in the output (default: 10000)")
    parser.add_argument("--min-samples", type=int, default=500,
                        help="Minimum rows for any class present (default: 500)")
    parser.add_argument("--chunk-size", type=int, default=100_000,
                        help="PyArrow read chunk size for RAM safety (default: 100000)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    label_map_path = Path(args.label_map)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"[preprocess] Reading {input_path} ...")
    pf = pq.ParquetFile(input_path)
    total_rows = pf.metadata.num_rows
    print(f"[preprocess] Total rows: {total_rows:,}")

    # --- Pass 1: Determine label column name and parse all labels ---
    # Read a sample to find the label column
    sample_df = next(pf.iter_batches(batch_size=5000)).to_pandas()
    print(f"[preprocess] Columns ({len(sample_df.columns)}): {sample_df.columns.tolist()}")

    # Identify the label column — try common names
    label_col_raw: str = ""
    for candidate in ["label", "Label", "attack_cat", "type", "class", "target"]:
        if candidate in sample_df.columns:
            label_col_raw = candidate
            break

    if not label_col_raw:
        raise ValueError(
            f"No label column found. Available columns: {sample_df.columns.tolist()}"
        )
    print(f"[preprocess] Label column: '{label_col_raw}'")
    print(f"[preprocess] Sample raw labels: {sample_df[label_col_raw].unique()[:10].tolist()}")

    # --- Pass 2: Stream full file, parse labels, collect clean data ---
    # RAM constraint: process in chunks, build list of small DataFrames
    # Full 1M row dataset at ~73 float32 cols = ~280MB — fits in 7.4GB WSL RAM
    # but we chunk anyway for safety
    print(f"[preprocess] Streaming and parsing labels (chunk_size={args.chunk_size:,}) ...")
    chunks: List[pd.DataFrame] = []

    for i, batch in enumerate(pf.iter_batches(batch_size=args.chunk_size)):
        df_chunk = batch.to_pandas()

        # Parse Zeek label strings → clean class names
        df_chunk["label_str"] = df_chunk[label_col_raw].astype(str).map(parse_zeek_label)

        # Drop the original raw label column (will be replaced by integer encoding)
        if label_col_raw != "label":
            df_chunk = df_chunk.drop(columns=[label_col_raw])

        chunks.append(df_chunk)
        rows_so_far = (i + 1) * args.chunk_size
        print(f"  chunk {i+1}: {min(rows_so_far, total_rows):,}/{total_rows:,} rows", end="\r")

    print()
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    print(f"[preprocess] Full dataset assembled: {len(df):,} rows")

    # --- Label distribution before sampling ---
    print("\n[preprocess] Raw label distribution:")
    label_counts = df["label_str"].value_counts()
    for label, count in label_counts.items():
        print(f"  {label:<45} {count:>8,}  ({count/len(df)*100:.1f}%)")

    # --- Build and save label encoder ---
    str_to_int, int_to_str = build_label_encoder(df["label_str"])
    df["label"] = df["label_str"].map(str_to_int).astype(np.int64)
    df = df.drop(columns=["label_str"])

    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w") as f:
        json.dump({"str_to_int": str_to_int, "int_to_str": {str(k): v for k, v in int_to_str.items()}}, f, indent=2)
    print(f"\n[preprocess] Label encoding saved to {label_map_path}")
    print(f"  Mapping: {str_to_int}")

    # --- Drop identifier columns ---
    df, dropped_ids = drop_identifier_columns(df)
    print(f"\n[preprocess] Dropped identifier columns ({len(dropped_ids)}): {dropped_ids}")

    # --- Encode remaining categorical columns ---
    df = encode_categorical_columns(df, exclude=["label"])
    print(f"[preprocess] Categorical columns encoded to integer codes.")

    # --- Convert to float32 for CNN (except label) ---
    feature_cols = [c for c in df.columns if c != "label"]
    df[feature_cols] = df[feature_cols].astype(np.float32)

    # --- Fill NaN with 0 (Zeek logs have many sparse optional fields) ---
    nan_counts = df[feature_cols].isna().sum()
    total_nan = nan_counts.sum()
    if total_nan > 0:
        print(f"[preprocess] Filling {total_nan:,} NaN values with 0.0")
        df[feature_cols] = df[feature_cols].fillna(0.0)

    # --- Drop zero-variance features ---
    df, dropped_zv = drop_zero_variance_columns(df, exclude=["label"])
    print(f"[preprocess] Dropped zero-variance columns ({len(dropped_zv)}): {dropped_zv}")
    print(f"[preprocess] Remaining features: {len([c for c in df.columns if c != 'label'])}")

    # --- Stratified sampling ---
    print(f"\n[preprocess] Stratified sampling "
          f"(target: {args.samples_per_class:,}/class, min: {args.min_samples:,}/class) ...")
    df_sampled = stratified_sample(
        df, "label",
        samples_per_class=args.samples_per_class,
        min_samples=args.min_samples,
    )
    del df

    print(f"\n[preprocess] Sampled dataset: {len(df_sampled):,} rows")
    print("[preprocess] Class distribution in output:")
    for label_int, count in df_sampled["label"].value_counts().sort_index().items():
        label_name = int_to_str[int(label_int)]
        print(f"  [{label_int}] {label_name:<45} {count:>7,}  ({count/len(df_sampled)*100:.1f}%)")

    # --- Verify no zero-variance in output (sanity check) ---
    feature_cols_final = [c for c in df_sampled.columns if c != "label"]
    zv_check = [c for c in feature_cols_final if df_sampled[c].nunique() <= 1]
    if zv_check:
        print(f"[preprocess] WARNING: Zero-variance columns still present: {zv_check}")
    else:
        print(f"[preprocess] ✓ All {len(feature_cols_final)} feature columns have variance > 0")

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df_sampled, preserve_index=False)
    pq.write_table(table, output_path, compression="snappy")
    print(f"\n[preprocess] ✓ Written to {output_path}")
    print(f"[preprocess]   Rows: {len(df_sampled):,}")
    print(f"[preprocess]   Columns: {len(df_sampled.columns)} "
          f"({len(feature_cols_final)} features + 1 label)")
    file_size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"[preprocess]   File size: {file_size_mb:.1f} MB")

    # --- Update num_classes hint ---
    num_classes = len(str_to_int)
    print(f"\n[preprocess] ⚠  num_classes = {num_classes}")
    print(f"[preprocess]    Update IntrusionDetectionCNN(num_classes={num_classes}) in edge_client.py")
    print(f"[preprocess]    if it differs from the current hardcoded value of 6.")


if __name__ == "__main__":
    main()