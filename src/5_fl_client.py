import flwr as fl
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.model_selection import train_test_split
from pqc_utils import pqc_encrypt_weights, pqc_decrypt_weights

# --- CONFIG ---
DATA_FILE = "data/sampled_data.parquet"
NUMERIC_FEATURES = [
    'duration_min', 'duration_max', 'duration_mean',
    'orig_bytes_min', 'orig_bytes_max', 'orig_bytes_mean',
    'resp_bytes_min', 'resp_bytes_max', 'resp_bytes_mean',
    'missed_bytes_min', 'missed_bytes_max', 'missed_bytes_mean',
    'orig_pkts_min', 'orig_pkts_max', 'orig_pkts_mean',
    'orig_ip_bytes_min', 'orig_ip_bytes_max', 'orig_ip_bytes_mean',
    'resp_pkts_min', 'resp_pkts_max', 'resp_pkts_mean',
    'resp_ip_bytes_min', 'resp_ip_bytes_max', 'resp_ip_bytes_mean',
    'local_orig_min', 'local_orig_max', 'local_orig_mean',
    'local_resp_min', 'local_resp_max', 'local_resp_mean'
]
BATCH_SIZE = 32
EPOCHS = 1  # Each FL round = 1 local epoch

CLIENT_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NUM_CLIENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 2

def load_partition():
    df = pd.read_parquet(DATA_FILE)
    features = [c for c in NUMERIC_FEATURES if c in df.columns]
    df = df.dropna(subset=['label'] + features)
    df["cid_split"] = df.index % NUM_CLIENTS
    df_part = df[df["cid_split"] == (CLIENT_ID-1)]
    X = df_part[features].values
    y = df_part["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = scaler.transform(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))
    return X_train, y_train, X_test, y_test

def build_cnn(input_dim):
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

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)
        model_weights = self.model.get_weights()

        ciphertext, encrypted_weights = pqc_encrypt_weights(model_weights, None)
        print(f"Client {CLIENT_ID} simulated encryption of weights")

        return encrypted_weights, len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return float(loss), len(self.X_test), {"accuracy": float(acc)}

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_partition()
    model = build_cnn(X_train.shape[1])
    fl.client.start_numpy_client(server_address="localhost:8080", client=FLClient(model, X_train, y_train, X_test, y_test))
