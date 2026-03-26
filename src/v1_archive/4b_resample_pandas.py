# src/4b_resample_pandas.py
import pandas as pd
import numpy as np
import sys
import os
import time

# --- Configuration ---
INPUT_FILE = 'data/processed_parquet/merged_data.parquet'
OUTPUT_FILE = 'data/processed_parquet/sampled_data.parquet'
RESAMPLE_RULE = '5s' # 5-second non-overlapping windows

# --- [P5] Aggregation Logic ---
# [P5] "multi-class labels, aggregation is performed using the 'most frequent' logic"
def most_frequent_logic(s):
    """The callable logic for our 'most_frequent' aggregation."""
    if s.empty:
        return None
    # Find the most frequent value (the index of the max count)
    return s.value_counts().idxmax()

def get_feature_aggregations(df):
    """
    Dynamically create aggregation dictionary for *features only*,
    replicating the strategy in [P5].
    """
    # [P5] "numeric features...: minimum, maximum, and mean."
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != 'label']
    
    # [P5] "categorical...: most frequent, second most frequent..."
    # We use 'count' and 'nunique' as robust proxies for the other two.
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if col != 'label']

    print(f"[PROGRESS] Found {len(numerical_cols)} numerical features for (min, max, mean).")
    print(f"[PROGRESS] Found {len(categorical_cols)} categorical features for (count, nunique).")
    
    feature_aggregations = {}
    for col in numerical_cols:
        feature_aggregations[col] = ['min', 'max', 'mean']
    for col in categorical_cols:
        feature_aggregations[col] = ['count', 'nunique']
    
    all_cols = set(df.columns)
    final_feature_aggs = {k: v for k, v in feature_aggregations.items() if k in all_cols}
    
    if 'label' not in all_cols:
        print("ERROR: 'label' column not found in Parquet file.", file=sys.stderr)
        sys.exit(1)
    if not final_feature_aggs:
        print("Warning: No numerical/categorical features found for aggregation.")

    return final_feature_aggs

# --- Main Script ---
def main():
    overall_start_time = time.time()
    print("--- [START] Resampling Script ---")

    # --- 1. Load Data with Pandas ---
    print(f"[1/6] Loading full dataset with Pandas from {INPUT_FILE} ...")
    step_start_time = time.time()
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}", file=sys.stderr)
        print("Please run '3_chunked_processing.py' first.", file=sys.stderr)
        return

    try:
        df = pd.read_parquet(INPUT_FILE)
    except Exception as e:
        print(f"Error loading Parquet file: {e}", file=sys.stderr)
        return
        
    print(f"      > Loaded {len(df)} rows in {time.time() - step_start_time:.2f}s")

    if 'ts' not in df.columns:
        print("Error: 'ts' column not found. It's required for resampling.", file=sys.stderr)
        return
        
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts').sort_index()

    # --- 2. Define Aggregations (Dynamically) ---
    print("[2/6] Defining Aggregations...")
    feature_aggs = get_feature_aggregations(df)

    # --- 3. Resample Features and Label Separately ---
    print(f"[3/6] Resampling features ({RESAMPLE_RULE} windows)... (This may take a minute)")
    step_start_time = time.time()
    # Call 1: Aggregate all features
    df_resampled_features = df.resample(RESAMPLE_RULE).agg(feature_aggs)
    print(f"      > Feature resampling complete in {time.time() - step_start_time:.2f}s")

    print("[4/6] Resampling label (using custom 'most_frequent' logic)...")
    step_start_time = time.time()
    # Call 2: Aggregate the label
    label_resampled_series = df['label'].resample(RESAMPLE_RULE).apply(most_frequent_logic)
    label_resampled_series = label_resampled_series.rename('label')
    print(f"      > Label resampling complete in {time.time() - step_start_time:.2f}s")

    # --- 4. Flatten Column Indexes ---
    print("[5/6] Flattening MultiIndex columns...")
    
    # Flatten features: ('orig_bytes', 'min') -> 'orig_bytes_min'
    if isinstance(df_resampled_features.columns, pd.MultiIndex):
        df_resampled_features.columns = ["_".join(col).strip() for col in df_resampled_features.columns.values]
    
    # --- 5. Combine Feature and Label DataFrames ---
    print("      Combining feature and label DataFrames...")
    
    # Use assign to add the label Series to the feature DataFrame
    df_final_agg = df_resampled_features.assign(label=label_resampled_series)
    
    # Clean up any potential NaN values from the resampling process
    df_final_agg = df_final_agg.fillna(0) 

    # --- 6. Compute and Save ---
    print(f"[6/6] Saving to {OUTPUT_FILE} ...")
    step_start_time = time.time()
    
    df_final_agg.to_parquet(OUTPUT_FILE, index=True)
    
    print(f"      > Save complete in {time.time() - step_start_time:.2f}s")
    
    overall_end_time = time.time()
    print("\n--- [COMPLETE] Resampling Finished. ---")
    print(f"Final aggregated data saved to {OUTPUT_FILE}")
    print(f"New dataset shape: {df_final_agg.shape}")
    print(f"Total time taken: {overall_end_time - overall_start_time:.2f} seconds")

if __name__ == "__main__":
    main()