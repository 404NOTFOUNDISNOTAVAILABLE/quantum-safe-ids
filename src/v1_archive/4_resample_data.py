import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import pandas as pd
import numpy as np
import os
import time

print("--- Loading data with Dask ---")
INPUT_FILE = './data/processed_parquet/merged_data.parquet'
OUTPUT_FILE = './data/processed_parquet/final_agg_data.parquet'

# 1. Load the data LAZILY with Dask
ddf = dd.read_parquet(INPUT_FILE)

# 2. Set the index to 'ts' (also lazy)
ddf['ts'] = dd.to_datetime(ddf['ts'])
ddf = ddf.set_index('ts', sorted=True)

print("--- Defining Aggregations (Dask-friendly) ---")

# Define aggregations. We'll use Dask-safe functions.
agg_dict = {}
numerical_cols = ddf.select_dtypes(include=np.number).columns

# --- THIS IS THE FIX ---
# We now include BOTH 'object' and 'string' types
categorical_cols = ddf.select_dtypes(include=['object', 'string']).columns.drop('label')

for col in numerical_cols:
    agg_dict[col] = ['min', 'max', 'mean']

for col in categorical_cols:
    agg_dict[col] = ['first', 'last', 'nunique'] # Dask-friendly

# 3. Resample the labels (lazy)
label_resampled = ddf['label'].resample('5S').apply(
    lambda x: 'Benign' if (x == 'Benign').all() else x[x != 'Benign'].iloc[0],
    meta=('label', 'object')
)

# 4. Resample the features (lazy)
df_resampled = ddf.resample('5S').agg(agg_dict)

# 5. Join labels back (lazy)
df_final_agg_dask = df_resampled.join(label_resampled)

# 6. Flatten column names (lazy)
df_final_agg_dask.columns = ['_'.join(col).strip() for col in df_final_agg_dask.columns.values]

# --- This is where the work happens! ---
print("\n--- Starting Dask computation (this has a progress bar) ---")
start_time = time.time()
with ProgressBar():
    # .compute() tells Dask to run the plan.
    df_final_agg = df_final_agg_dask.compute()

# 7. Final cleanup (in Pandas, now that it's small)
df_final_agg = df_final_agg.dropna(how='all')

# 8. Save the final aggregated file
print(f"\n--- Saving final aggregated data to {OUTPUT_FILE} ---")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df_final_agg.to_parquet(OUTPUT_FILE, engine='fastparquet')

end_time = time.time()
print("\n--- Success! ---")
print(f"Resampling complete in {end_time - start_time:.2f} seconds.")
print(f"New DataFrame shape: {df_final_agg.shape}")
print("\nFinal Label distribution:")
print(df_final_agg['label'].value_counts())

