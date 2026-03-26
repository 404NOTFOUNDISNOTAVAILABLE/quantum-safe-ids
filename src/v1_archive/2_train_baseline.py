"""
Step 2: Train Baseline CNN Model [P5]
Trains on the preprocessed sampled_data.parquet
Model: 1D CNN (3 conv blocks + 2 dense layers)
Output: saves to models/baseline_cnn.h5
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ CONFIG ============
DATA_FILE = "data/sampled_data.parquet"
MODEL_PATH = "models/baseline_cnn.h5"
EPOCHS = 10
BATCH_SIZE = 32
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# ============ LOAD DATA ============

def load_and_preprocess_data():
    """Load sampled data and prepare for training"""
    logger.info(f"Loading {DATA_FILE}...")
    start = time.time()
    
    with tqdm(total=100, desc="Loading", unit="%", ncols=80) as pbar:
        df = pd.read_parquet(DATA_FILE)
        pbar.update(25)
        
        # Remove NaN labels
        df = df.dropna(subset=['label'])
        pbar.update(25)
        
        # Select numeric features (exclude label, ts, etc.)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in numeric_cols:
            numeric_cols.remove('label')
        if 'packet_count' in numeric_cols:
            numeric_cols.remove('packet_count')
        if 'flow_count' in numeric_cols:
            numeric_cols.remove('flow_count')
        
        # Drop rows with ANY NaN in feature columns
        df = df.dropna(subset=numeric_cols)
        pbar.update(25)
        
        # Fill any remaining NaN with 0
        df[numeric_cols] = df[numeric_cols].fillna(0)
        pbar.update(25)
    
    elapsed = time.time() - start
    logger.info(f"✓ Loaded {len(df):,} rows in {elapsed:.1f}s")
    logger.info(f"  Features: {len(numeric_cols)}")
    logger.info(f"  Label distribution: {(df['label']==1).sum()} attacks, {(df['label']==0).sum()} normal")
    
    X = df[numeric_cols].values
    y = df['label'].values
    
    return X, y, numeric_cols



# ============ PREPARE DATA ============

def prepare_train_test_data(X, y):
    """Split and standardize data"""
    logger.info("\nPreparing train/test split...")
    
    with tqdm(total=100, desc="Splitting", unit="%", ncols=80) as pbar:
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42, stratify=y
        )
        pbar.update(33)
        
        # Standardize
        logger.info("Standardizing features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        pbar.update(33)
        
        # Reshape for CNN: (samples, timesteps, features) → (samples, features, 1)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        pbar.update(34)
    
    logger.info(f"  Train shape: {X_train.shape}")
    logger.info(f"  Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


# ============ CREATE MODEL ============

def create_cnn_model(input_dim):
    """[P5] Section 5.2: CNN architecture for FL-IDS"""
    logger.info(f"\nCreating CNN model (input_dim={input_dim})...")
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        
        # Conv Block 1
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Conv Block 2
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Conv Block 3
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    logger.info(model.summary())
    return model


# ============ TRAIN MODEL ============

def train_model(model, X_train, y_train, X_test, y_test):
    """Train CNN model"""
    logger.info(f"\nTraining model for {EPOCHS} epochs...")
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SIZE,
        verbose=1
    )
    
    return history


# ============ EVALUATE ============

def evaluate_model(model, X_test, y_test):
    """Evaluate on test set"""
    logger.info("\nEvaluating on test set...")
    
    with tqdm(total=100, desc="Evaluating", unit="%", ncols=80) as pbar:
        loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
        pbar.update(100)
    
    logger.info(f"✓ Test Loss: {loss:.4f}")
    logger.info(f"  Test Accuracy: {accuracy:.4f}")
    logger.info(f"  Test AUC: {auc:.4f}")
    
    return loss, accuracy, auc


# ============ SAVE MODEL ============

def save_model(model):
    """Save trained model"""
    model_path = Path(MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving model to {MODEL_PATH}...")
    
    with tqdm(total=100, desc="Saving", unit="%", ncols=80) as pbar:
        model.save(MODEL_PATH)
        pbar.update(100)
    
    logger.info(f"✓ Model saved! Size: {model_path.stat().st_size / (1024**2):.2f} MB")


# ============ MAIN ============

def main():
    logger.info("="*80)
    logger.info("TRAIN BASELINE CNN MODEL [P5]")
    logger.info("="*80)
    
    start_time = time.time()
    
    try:
        # Step 1: Load data
        X, y, feature_names = load_and_preprocess_data()
        
        # Step 2: Prepare data
        X_train, X_test, y_train, y_test = prepare_train_test_data(X, y)
        
        # Step 3: Create model
        model = create_cnn_model(X_train.shape[1])
        
        # Step 4: Train
        history = train_model(model, X_train, y_train, X_test, y_test)
        
        # Step 5: Evaluate
        loss, accuracy, auc = evaluate_model(model, X_test, y_test)
        
        # Step 6: Save
        save_model(model)
        
        total_elapsed = time.time() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("✓✓✓ BASELINE MODEL TRAINING COMPLETE! ✓✓✓")
        logger.info("="*80)
        logger.info(f"Model: {MODEL_PATH}")
        logger.info(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        logger.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.2f}m)")
        logger.info("Ready for Federated Learning!")
        
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
