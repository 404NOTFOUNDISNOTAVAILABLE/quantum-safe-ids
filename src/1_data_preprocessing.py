"""
Step 1: Data Preprocessing for FL-IDS
Replicates [P5] methodology: Group by 5-min intervals, aggregate
Input: merged_data.parquet (1M+ Zeek logs)
Output: sampled_data.parquet (ready for FL training)
SIMPLIFIED: Uses built-in pandas resample (no overlaps, memory-safe)
WITH PROGRESS BARS
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ CONFIG ============
INPUT_FILE = "data/processed_parquet/merged_data.parquet"
OUTPUT_FILE = "data/sampled_data.parquet"
RESAMPLE_WINDOW = "5min"  # Resample every 5 minutes (instead of 5s with overlap)
SAMPLE_RATIO = 0.1       # Use 10% of data

NUMERIC_FEATURES = [
    'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
    'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes',
    'local_orig', 'local_resp'
]

# ============ LOAD & SAMPLE ============

def load_and_sample():
    """Load with Dask and sample to 10% for speed"""
    logger.info(f"Loading {INPUT_FILE}...")
    start = time.time()
    
    ddf = dd.read_parquet(INPUT_FILE, engine='pyarrow')
    
    # Sample 10%
    def sample_partition(df):
        if len(df) > 0:
            return df.sample(frac=SAMPLE_RATIO, random_state=42)
        return df
    
    logger.info(f"Sampling {SAMPLE_RATIO*100:.0f}% of data...")
    with tqdm(total=100, desc="Sampling", unit="%", ncols=80) as pbar:
        ddf_sampled = ddf.map_partitions(sample_partition, meta=ddf)
        pbar.update(50)
        
        logger.info("Converting to Pandas...")
        df = ddf_sampled.compute()
        pbar.update(50)
    
    elapsed = time.time() - start
    logger.info(f"✓ Loaded {len(df):,} rows ({df.memory_usage(deep=True).sum() / (1024**2):.1f} MB) in {elapsed:.1f}s")
    return df


# ============ PREPROCESS ============

def preprocess(df):
    """Simple resample + aggregate [P5] methodology"""
    logger.info(f"\nResampling to {RESAMPLE_WINDOW} intervals...")
    start = time.time()
    
    # Ensure timestamp is datetime and set as index
    logger.info("Step 1: Converting timestamp and setting index...")
    with tqdm(total=1, desc="Timestamp", unit="op", ncols=80) as pbar:
        df['ts'] = pd.to_datetime(df['ts'])
        df = df.set_index('ts').sort_index()
        pbar.update(1)
    
    # Convert numeric columns to proper numeric type (drop non-numeric)
    logger.info("Step 2: Converting numeric columns...")
    with tqdm(total=len(NUMERIC_FEATURES), desc="Converting", unit="col", ncols=80) as pbar:
        for col in NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            pbar.update(1)
    
    # Build aggregation dict - ONLY for numeric columns
    logger.info("Step 3: Building aggregation dictionary...")
    agg_dict = {}
    
    # Only aggregate numeric columns that are actually numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col in NUMERIC_FEATURES]
    
    with tqdm(total=len(numeric_cols)+1, desc="Building agg", unit="col", ncols=80) as pbar:
        for col in numeric_cols:
            agg_dict[col] = ['min', 'max', 'mean']
            pbar.update(1)
        
        # Add uid count
        if 'uid' in df.columns:
            agg_dict['uid'] = 'nunique'
        pbar.update(1)
    
    # Resample and aggregate
    logger.info("Step 4: Applying resample and aggregation...")
    with tqdm(total=100, desc="Resampling", unit="%", ncols=80) as pbar:
        pbar.update(25)
        sampled = df.resample(RESAMPLE_WINDOW).agg(agg_dict)
        pbar.update(25)
        
        # Flatten column names
        sampled.columns = ['_'.join(col).strip('_') for col in sampled.columns.values]
        pbar.update(25)
        
        # Rename
        if 'uid_nunique' in sampled.columns:
            sampled = sampled.rename(columns={'uid_nunique': 'flow_count'})
        pbar.update(5)
        
        # Add packet count
        sampled['packet_count'] = df.resample(RESAMPLE_WINDOW).size()
        pbar.update(5)
    
    # Handle label
    logger.info("Step 5: Processing labels...")
    with tqdm(total=100, desc="Labels", unit="%", ncols=80) as pbar:
        if 'label' in df.columns:
            pbar.update(50)
            label_agg = df[['label']].resample(RESAMPLE_WINDOW).apply(
                lambda x: 1 if ((x == 'attack').any().any() or (x == 1).any().any()) else 0
            )
            sampled['label'] = label_agg
            pbar.update(50)
        else:
            pbar.update(100)
    
    # Cleanup
    logger.info("Step 6: Cleanup...")
    with tqdm(total=100, desc="Cleanup", unit="%", ncols=80) as pbar:
        sampled = sampled.dropna(subset=['label'])
        pbar.update(50)
        
        sampled = sampled.reset_index()
        pbar.update(50)
    
    elapsed = time.time() - start
    logger.info(f"✓ Created {len(sampled)} windows in {elapsed:.1f}s")
    
    # Label stats
    if 'label' in sampled.columns:
        n_attack = (sampled['label'] == 1).sum()
        n_normal = (sampled['label'] == 0).sum()
        pct = n_attack / (n_attack + n_normal) * 100 if (n_attack + n_normal) > 0 else 0
        logger.info(f"  Label distribution: {n_attack} attacks ({pct:.1f}%), {n_normal} normal")
    
    return sampled



# ============ SAVE ============

def save_data(df):
    """Save to parquet"""
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving to {OUTPUT_FILE}...")
    start = time.time()
    
    with tqdm(total=100, desc="Saving", unit="%", ncols=80) as pbar:
        pbar.update(25)
        df.to_parquet(OUTPUT_FILE, index=False, compression='snappy')
        pbar.update(75)
    
    elapsed = time.time() - start
    file_size_mb = output_path.stat().st_size / (1024**2)
    logger.info(f"✓ Saved in {elapsed:.1f}s! Size: {file_size_mb:.2f} MB")
    logger.info(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    return df


# ============ MAIN ============

def main():
    logger.info("="*80)
    logger.info("DATA PREPROCESSING PIPELINE [P5] - WITH PROGRESS BARS")
    logger.info("="*80)
    
    start_time = time.time()
    
    try:
        df = load_and_sample()
        df = preprocess(df)
        df = save_data(df)
        
        total_elapsed = time.time() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("✓✓✓ PREPROCESSING COMPLETE! ✓✓✓")
        logger.info("="*80)
        logger.info(f"Output: {OUTPUT_FILE}")
        logger.info(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.2f} minutes)")
        logger.info(f"Ready for FL training!")
        
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
