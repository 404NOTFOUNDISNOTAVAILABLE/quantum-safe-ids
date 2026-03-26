import pandas as pd
import numpy as np
import os
import time

# --- Configuration ---
LOG_DIR_BASE = './data/processed_logs'
RAW_DATA_DIR_BASE = './data/raw_pcaps/opt/Malware-Project/BigDataset/IoTScenarios'
OUTPUT_PATH = './data/processed_parquet' # A folder will be created

# Define our scenarios
SCENARIOS = {
    'scenario_1': {
        'log_dir': os.path.join(LOG_DIR_BASE, 'scenario_1'),
        'label_file': os.path.join(RAW_DATA_DIR_BASE, 'CTU-IoT-Malware-Capture-1-1/bro/conn.log.labeled')
    },
    'scenario_2': {
        'log_dir': os.path.join(LOG_DIR_BASE, 'scenario_2'),
        'label_file': os.path.join(RAW_DATA_DIR_BASE, 'CTU-IoT-Malware-Capture-34-1/bro/conn.log.labeled')
    },
    'scenario_3': {
        'log_dir': os.path.join(LOG_DIR_BASE, 'scenario_3'),
        'label_file': os.path.join(RAW_DATA_DIR_BASE, 'CTU-Honeypot-Capture-4-1/bro/conn.log.labeled')
    }
}

# --- PANDAS Helper Function (We know this works) ---
def load_zeek_log_pandas(log_path, default_col_prefix='col'):
    col_names = []
    data_rows = []
    
    if not os.path.exists(log_path):
        print(f"    [Info] Log file not found: {os.path.basename(log_path)}. Skipping.")
        return None

    try:
        with open(log_path, 'r', encoding='utf-8') as f: # Added encoding
            for line in f:
                if line.startswith('#'):
                    if line.startswith('#fields'):
                        if not col_names:
                            col_names = line.strip().replace('#fields\t', '').split('\t')
                    continue
                data_rows.append(line.strip().split('\t'))
    except Exception as e:
        print(f"    [Error] Could not read {log_path}: {e}")
        return None
    
    if not data_rows:
        print(f"    [Info] No data rows found in {os.path.basename(log_path)}. Skipping.")
        return None
    
    if not col_names:
        try:
            num_cols = len(data_rows[0])
            col_names = [f"{default_col_prefix}_{i}" for i in range(num_cols)]
        except IndexError:
            print(f"    [Error] File {log_path} appears to be empty.")
            return None

    try:
        df = pd.DataFrame(data_rows, columns=col_names)
        df.replace(to_replace=['(empty)', '-'], value=np.nan, inplace=True)
    except ValueError as e:
        print(f"    [Error] Mismatch in columns/data for {log_path}: {e}")
        return None
    return df

# --- Main Processing Function ---
def main():
    start_time = time.time()
    print(f"Starting chunked data processing... Output will be saved to {OUTPUT_PATH}")
    
    all_dataframes = [] # We will store the small, clean dataframes here

    for scenario_name, paths in SCENARIOS.items():
        print(f"\n--- Processing {scenario_name} ---")
        
        # === 1. Load LABELED conn.log as the BASE ===
        # THIS IS THE FIX: We start with the label file
        label_file_path = paths['label_file']
        main_df = load_zeek_log_pandas(label_file_path, 'conn')
        
        if main_df is None:
            print(f"  [Error] conn.log.labeled missing for {scenario_name}. Skipping this scenario.")
            continue
        
        # Rename the label column
        label_col_name = main_df.columns[-1] # The last column is the label
        main_df = main_df.rename(columns={label_col_name: 'label'})
        
        main_df['ts'] = pd.to_datetime(main_df['ts'], unit='s', errors='coerce')
        print(f"  > Loaded conn.log.labeled. Shape: {main_df.shape}")

        # === 2. Load and Merge Other Logs (http, dns, ssl) ===
        # These logs come from *our* Zeek run
        logs_to_merge = {
            'http': os.path.join(paths['log_dir'], 'http.log'),
            'dns': os.path.join(paths['log_dir'], 'dns.log'),
            'ssl': os.path.join(paths['log_dir'], 'ssl.log')
        }
        
        for log_name, log_path in logs_to_merge.items():
            log_df = load_zeek_log_pandas(log_path, log_name)
            if log_df is not None:
                # Merge our extra logs onto the base label file
                main_df = pd.merge(main_df, log_df, on='uid', how='left', suffixes=('', f'_{log_name}'))
        
        print(f"  > Merged other logs. New shape: {main_df.shape}")

        # === 3. Clean Data (Robust Version) ===
        print("  > Cleaning data (filling NaN values)...")
        
        # Fill numeric columns with 0
        num_cols = main_df.select_dtypes(include=np.number).columns
        main_df[num_cols] = main_df[num_cols].fillna(0)
        
        # Fill text/object columns with 'None'
        obj_cols = main_df.select_dtypes(include=['object']).columns
        main_df[obj_cols] = main_df[obj_cols].fillna('None')
        
        # Make sure the label column is clean
        main_df['label'] = main_df['label'].fillna('Benign')
        
        all_dataframes.append(main_df)
        print(f"  > Finished processing {scenario_name}. Storing in memory.")

    # === 4. Combine and Save ===
    if not all_dataframes:
        print("[Error] No data was processed. Exiting.")
        return
        
    print(f"\n--- Combining all {len(all_dataframes)} scenarios and saving... ---")
    
    try:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"  > Final DataFrame shape: {final_df.shape}")
        
        # --- FIX: Convert all object columns to string type BEFORE saving ---
        print("  > Converting all object columns to string type for safety...")
        for col in final_df.select_dtypes(include=['object']).columns:
            final_df[col] = final_df[col].astype(str)
        
        # Save to a single, efficient Parquet file
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        final_output_file = os.path.join(OUTPUT_PATH, 'merged_data.parquet')
        final_df.to_parquet(final_output_file, engine='fastparquet')
        
        end_time = time.time()
        print("\n--- Success! ---")
        print(f"Processing complete in {end_time - start_time:.2f} seconds.")
        print(f"Clean data saved to: {final_output_file}")
        print("\nFinal Label distribution:")
        print(final_df['label'].value_counts())
        
    except MemoryError:
        print("\n[CRITICAL ERROR] Failed at the final combination step due to memory.")
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred during final save: {e}")


if __name__ == "__main__":
    main()
