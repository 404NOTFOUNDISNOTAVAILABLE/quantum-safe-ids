# src/5_resample_dask.py
import dask.dataframe as dd
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

    print(f"Found {len(numerical_cols)} numerical features for (min, max, mean).")
    print(f"Found {len(categorical_cols)} categorical features for (count, nunique).")
    
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
    start_time = time.time()
    
    # --- 1. Load Data ---
    print("--- Loading data with Dask ---")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}", file=sys.stderr)
        print("Please run '3_chunked_processing.py' first.", file=sys.stderr)
        return

    df_dask = dd.read_parquet(INPUT_FILE)
    
    if 'ts' not in df_dask.columns:
        print("Error: 'ts' column not found. It's required for resampling.", file=sys.stderr)
        return
        
    df_dask['ts'] = dd.to_datetime(df_dask['ts'])
    df_dask = df_dask.set_index('ts', sorted=True)

    # --- 2. Define Aggregations (Dynamically) ---
    print("--- Defining Aggregations ---")
    feature_aggs = get_feature_aggregations(df_dask)

    # --- 3. THE FIX: Resample Features and Label Separately ---

    print(f"--- Planning feature resampling ({RESAMPLE_RULE} windows)... ---")
    # Call 1: Resample the full DataFrame, then aggregate features
    df_resampled_features = df_dask.resample(RESAMPLE_RULE).agg(feature_aggs)

    print("--- Planning label resampling (using custom Python function)... ---")
    # Call 2: Select the 'label' Series, resample it, and pass the
    # raw Python function 'most_frequent_logic' directly to .agg()
    # We MUST provide 'meta' to tell Dask the output type (a string/object)
    label_resampled_series = df_dask['label'].resample(RESAMPLE_RULE).agg(most_frequent_logic, meta=('label', 'object'))
    
    # Rename the resulting Series for a clean merge
    label_resampled_series = label_resampled_series.rename('label')

    # --- 4. Flatten Column Indexes ---
    print("--- Flattening MultiIndex columns... ---")
    
    # Flatten features: ('orig_bytes', 'min') -> 'orig_bytes_min'
    if isinstance(df_resampled_features.columns, pd.MultiIndex):
        df_resampled_features.columns = ["_".join(col).strip() for col in df_resampled_features.columns.values]
    
    # --- 5. Combine Feature and Label DataFrames ---
    print("--- Combining feature and label DataFrames... ---")
    
    # Use assign to add the label Series to the feature DataFrame
    # This joins on their shared DatetimeIndex
    df_final_agg_dask = df_resampled_features.assign(label=label_resampled_series)
    
    # Clean up any potential NaN values from the resampling process
    df_final_agg_dask = df_final_agg_dask.fillna(0) 

    # --- 6. Compute and Save ---
    print(f"--- Computing and saving to {OUTPUT_FILE} ---")
    
    # This executes the Dask plan. This step will take some time.
    df_final_agg_dask.to_parquet(OUTPUT_FILE, write_index=True, schema='infer')
    
    end_time = time.time()
    print("\n--- Resampling Complete. ---")
    print(f"Final aggregated data saved to {OUTPUT_FILE}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()