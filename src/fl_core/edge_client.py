import flwr as fl
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import numpy as np
import time

from data.data_streamer import ParquetDataStreamer
from model.ids_cnn import IntrusionDetectionCNN
from model.mobilenet_ids import MobileNetV2IDS
from crypto.kyber_manager import KyberManager

# DP epsilon accountant — requires tensorflow-privacy==0.9.0
from dp_accounting.rdp import rdp_privacy_accountant
from dp_accounting import dp_event

class QuantumEdgeClient(fl.client.NumPyClient):
    def __init__(
        self,
        data_path: str,
        client_id: int,
        num_clients: int = 2,
        num_classes: int = 4,
        dp_enabled: bool = False,
        l2_norm_clip: float = 1.0,
        noise_multiplier: float = 1.1,
        num_microbatches: int = 1,
        dp_delta: float = 1e-5,
        model_arch: str = "cnn",
    ):
        """
        Quantum-safe FL edge client with optional DP-SGD.

        Args:
            data_path: Path to the Parquet file.
            client_id: Integer client identifier for data partitioning.
            num_clients: Total number of federated clients.
            dp_enabled: Enables DP-SGD training via DPKerasAdamOptimizer.
            l2_norm_clip: Gradient clipping bound for DP-SGD.
            noise_multiplier: Gaussian noise multiplier for DP-SGD.
            num_microbatches: Microbatch count for per-example gradient computation.
                              Must satisfy: batch_size % num_microbatches == 0.
            dp_delta: Target delta for (epsilon, delta)-DP guarantee. Default 1e-5.
        """
        self.client_id = client_id
        self.dp_enabled = dp_enabled
        self.noise_multiplier = noise_multiplier
        self.num_microbatches = num_microbatches
        self.dp_delta = dp_delta
        self.rounds_elapsed: int = 0  # Tracks cumulative rounds for epsilon accounting

        # 1. Initialize the Data Streamer — num_samples exposed for epsilon computation
        self.streamer = ParquetDataStreamer(
            file_path=data_path,
            batch_size=32,
            num_clients=num_clients,
            client_id=client_id
        )
        self.tf_dataset = self.streamer.get_tf_dataset()
        self.num_samples: int = self.streamer.num_samples
        self.batch_size: int = self.streamer.batch_size

        # 2. Initialize model with DP configuration
        input_shape = (len(self.streamer.features), 1)
        if model_arch == "mobilenet":
            model = MobileNetV2IDS(
                input_shape=input_shape,
                num_classes=num_classes,
                learning_rate=0.001,
                dp_enabled=dp_enabled,
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
            )
            print(f"[Client] Architecture: MobileNetV2-1D (alpha=0.75)")
        else:
            model = IntrusionDetectionCNN(
                input_shape=input_shape,
                num_classes=num_classes,
                learning_rate=0.001,
                dp_enabled=dp_enabled,
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
            )
            print(f"[Client] Architecture: 1D-CNN (baseline)")
        self.cnn = model

        # 3. Initialize PQC engine — use NIST final standard name
        self.crypto_engine = KyberManager(alg_name="ML-KEM-512")

        print(f"[Client {self.client_id}] Initialized. "
              f"Samples≈{self.num_samples}, DP={'ON' if dp_enabled else 'OFF'}")

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        """Returns current local model weights."""
        return self.cnn.get_weights()

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Core FL training round:
          1. Apply global weights
          2. Train locally with DP-SGD (if enabled)
          3. Encrypt trained weights with ML-KEM-512 + AES-GCM
          4. Compute and log epsilon for this round (if DP enabled)
          5. Return encrypted payload + IEEE telemetry metrics

        All timing uses time.perf_counter_ns() for sub-millisecond precision.
        """
        self.rounds_elapsed += 1
        local_epochs: int = int(config.get("local_epochs", 1))

        print(f"\n[Client {self.client_id}] Round {self.rounds_elapsed} — "
              f"applying global weights, starting local training...")

        # 1. Apply global weights
        self.cnn.set_weights(parameters)

        # 2. Local training — perf_counter_ns for IEEE metrics table
        train_start_ns: int = time.perf_counter_ns()
        history = self.cnn.train_on_stream(self.tf_dataset, epochs=local_epochs)
        train_elapsed_ms: float = (time.perf_counter_ns() - train_start_ns) / 1e6

        # 3. Extract trained weights (DP noise already applied during optimizer step)
        new_weights = self.cnn.get_weights()

        # 4. PQC Encryption
        server_pub_key: Optional[bytes] = config.get("server_public_key", None)

        encap_ms: float = 0.0
        sym_enc_ms: float = 0.0
        encrypted_parameters: List[np.ndarray]

        if server_pub_key:
            secure_payload = self.crypto_engine.encapsulate_and_encrypt(
                server_pub_key, new_weights
            )

            # Sub-timings exposed directly by KyberManager for IEEE metrics table
            encap_ms = secure_payload["encap_latency_ns"] / 1e6
            sym_enc_ms = secure_payload["sym_enc_latency_ns"] / 1e6

            # Zero-copy byte → ndarray conversion using np.frombuffer (no intermediate list)
            encrypted_parameters = [
                np.frombuffer(secure_payload["kyber_ciphertext"], dtype=np.uint8).copy(),
                np.frombuffer(secure_payload["aes_nonce"], dtype=np.uint8).copy(),
                np.frombuffer(secure_payload["encrypted_weights"], dtype=np.uint8).copy(),
            ]

            encrypted_payload_bytes: int = (
                len(secure_payload["kyber_ciphertext"])
                + len(secure_payload["aes_nonce"])
                + len(secure_payload["encrypted_weights"])
            )
            plaintext_payload_bytes: int = secure_payload["original_size_bytes"]

        else:
            print(f"[Client {self.client_id}] WARNING: No server public key. Sending plaintext.")
            encrypted_parameters = new_weights
            encrypted_payload_bytes = sum(w.nbytes for w in new_weights)
            plaintext_payload_bytes = encrypted_payload_bytes

        # 5. Epsilon accounting (per-round cumulative) via dp_accounting RDP
        epsilon_consumed: float = -1.0  # Sentinel: DP disabled
        if self.dp_enabled and self.num_samples > 0:
            try:
                cumulative_epochs: int = self.rounds_elapsed * local_epochs
                steps: int = cumulative_epochs * (self.num_samples // self.batch_size)
                sampling_probability: float = self.batch_size / self.num_samples

                accountant = rdp_privacy_accountant.RdpAccountant()
                accountant.compose(
                    dp_event.SelfComposedDpEvent(
                        dp_event.PoissonSampledDpEvent(
                            sampling_probability,
                            dp_event.GaussianDpEvent(self.noise_multiplier)
                        ),
                        count=steps
                    )
                )
                epsilon_consumed = float(accountant.get_epsilon(self.dp_delta))
                print(f"[Client {self.client_id}] DP Budget: ε={epsilon_consumed:.4f}, "
                      f"δ={self.dp_delta} (cumulative over {self.rounds_elapsed} rounds)")
            except Exception as e:
                print(f"[Client {self.client_id}] WARNING: Epsilon computation failed: {e}")

        # IEEE metrics — all times in milliseconds, sizes in bytes
        metrics: Dict = {
            "client_id": int(self.client_id),
            "train_time_ms": float(train_elapsed_ms),
            "pqc_total_overhead_ms": float(encap_ms + sym_enc_ms),
            "pqc_encap_ms": float(encap_ms),
            "sym_enc_ms": float(sym_enc_ms),
            "encrypted_payload_bytes": int(encrypted_payload_bytes),
            "plaintext_payload_bytes": int(plaintext_payload_bytes),
            "is_encrypted": bool(server_pub_key),
            "dp_enabled": bool(self.dp_enabled),
            "epsilon_consumed": float(epsilon_consumed),
            "dp_delta": float(self.dp_delta),
            "round": int(self.rounds_elapsed),
        }

        return encrypted_parameters, self.num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, fl.common.Scalar]
    ) -> Tuple[float, int, Dict]:
        """Evaluates the global model on the local data stream."""
        self.cnn.set_weights(parameters)
        result = self.cnn.evaluate_stream(self.tf_dataset)
        return result["loss"], self.num_samples, {
            "accuracy": result["accuracy"],
            "client_id": self.client_id,
        }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="6G Quantum-Safe Edge Client")
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--dp", action="store_true", help="Enable DP-SGD")
    parser.add_argument("--noise", type=float, default=1.1, help="DP noise multiplier")
    parser.add_argument("--clip", type=float, default=1.0, help="DP l2 norm clip")
    parser.add_argument("--rounds", type=int, default=20, help="Expected FL rounds (for DP budget display)")
    parser.add_argument("--dataset", type=str, default="toniot",
                        choices=["toniot", "ciciot2023"],
                        help="Dataset to use for training")
    parser.add_argument("--model", type=str, default="cnn",
                        choices=["cnn", "mobilenet"],
                        help="Model architecture: cnn (default) or mobilenet")
    parser.add_argument("--local-epochs", type=int, default=2,
                        help="Number of local training epochs per round (default: 2)")
    args = parser.parse_args()

    if args.dataset == "ciciot2023":
        data_path = "../data/ciciot2023_sampled.parquet"
        num_classes = 7
    else:
        data_path = "../data/sampled_data.parquet"
        num_classes = 4

    client = QuantumEdgeClient(
        data_path=data_path,
        client_id=args.id,
        num_clients=args.clients,
        num_classes=num_classes,
        dp_enabled=args.dp,
        noise_multiplier=args.noise,
        l2_norm_clip=args.clip,
        model_arch=args.model,
    )

    print(f"[Client {args.id}] Connecting to server at 127.0.0.1:8080...")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()