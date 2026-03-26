"""
Creates synthetic attack samples, balances dataset, retrains CNN, evaluates using F1/TPR/FPR
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIG ---
DATA_FILE = "data/sampled_data.parquet"
MODEL_PATH = "models/cnn_balanced_attacks.h5"

# --- LOAD DATA ---
df = pd.read_parquet(DATA_FILE)

# Use only normal for generating synthetic attacks
normal = df[df["label"] == 0]
num_attack = min(3000, len(normal))

logging.info(f"Original normal samples: {len(normal)}")

# Generate attacks by injecting anomalies
attacks = normal.sample(num_attack, random_state=42).copy()
for c in df.select_dtypes(include=[np.number]).columns:
    if c not in ["packet_count", "flow_count"]:
        attacks[c] = attacks[c] * np.random.uniform(3, 10)  # Over-inflate random numeric fields
attacks["label"] = 1 

df_balanced = pd.concat([normal.sample(num_attack, random_state=42), attacks]).sample(frac=1, random_state=42)
logging.info(f"Final dataset: {df_balanced.shape}")

# --- PREPARE FEATURES ---
features = [c for c in df_balanced.select_dtypes(include=[np.number]).columns if c != "label"]

# Drop rows with any NaNs in feature columns
df_balanced_clean = df_balanced.dropna(subset=features)

# Optional: fill any remaining NaN just in case
df_balanced_clean[features] = df_balanced_clean[features].fillna(0)

X = df_balanced_clean[features].values
y = df_balanced_clean["label"].values

# --- SPLIT AND SCALE ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --- BUILD MODEL ---
def create_cnn(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.Conv1D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

model = create_cnn(X_train.shape[1])

# --- TRAIN ---
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)

# --- EVAL ---
metrics = model.evaluate(X_test, y_test, verbose=1)
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

y_pred = (model.predict(X_test) > 0.5).astype(int).ravel()
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
auc = roc_auc_score(y_test, y_pred)

print(f"\n\n=== FINAL EVALUATION ===")
print(f"Test Accuracy: {metrics[1]:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall (TPR): {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"AUC: {auc:.3f}")

model.save(MODEL_PATH)
