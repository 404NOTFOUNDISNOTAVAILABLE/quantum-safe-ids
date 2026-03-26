import csv
import os
import argparse
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from typing import List, Tuple, Dict, Optional, Union
import numpy as np

from crypto.kyber_manager import KyberManager

CSV_FIELDNAMES = [
    "condition", "run", "server_round", "client_id",
    "train_time_ms", "pqc_total_overhead_ms", "pqc_encap_ms", "sym_enc_ms",
    "encrypted_payload_bytes", "plaintext_payload_bytes",
    "is_encrypted", "dp_enabled", "epsilon_consumed",
    "global_loss", "global_accuracy",
    "eval_client_id", "eval_loss", "eval_accuracy",
]


class SecureQuantumStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, no_pqc: bool = False, csv_log: Optional[str] = None,
                 condition: str = "", run_id: int = 0, local_epochs: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_pqc = no_pqc
        self.csv_log = csv_log
        self.condition = condition
        self.run_id = run_id
        self.local_epochs = local_epochs

        if not no_pqc:
            self.crypto_engine = KyberManager(alg_name="ML-KEM-512")
            print("\n[Server] ML-KEM-512 engine ready. Keypairs generated per-round (PFS).")
        else:
            self.crypto_engine = None
            print("\n[Server] PQC disabled — plaintext baseline mode.")

        if csv_log:
            write_header = not os.path.exists(csv_log)
            with open(csv_log, "a", newline="") as f:
                if write_header:
                    csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writeheader()

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: fl.server.client_manager.ClientManager
                      ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Phase 1: Broadcast. Generates a fresh ephemeral keypair each round (PFS) unless no_pqc."""
        config_schedule = super().configure_fit(server_round, parameters, client_manager)

        if not self.no_pqc:
            round_public_key: bytes = self.crypto_engine.generate_keypair()
            print(f"[Server Round {server_round}] Fresh ML-KEM-512 keypair generated (PFS).")

        for client, fit_ins in config_schedule:
            if not self.no_pqc:
                fit_ins.config["server_public_key"] = round_public_key
            fit_ins.config["local_epochs"] = self.local_epochs

        return config_schedule

    def aggregate_fit(self, server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Phase 2: Secure Aggregation. Decrypts (if PQC), aggregates, and logs metrics."""
        if not results:
            return None, {}

        print(f"\n[Server Round {server_round}] Received payloads from {len(results)} clients.")
        decrypted_weights_with_sizes = []

        for client, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)

            if self.no_pqc:
                # Plaintext baseline — weights arrive directly
                decrypted_weights_with_sizes.append((ndarrays, fit_res.num_examples))

            elif len(ndarrays) == 3:
                # PQC mode — reconstruct and decrypt
                secure_payload = {
                    "kyber_ciphertext": ndarrays[0].astype(np.uint8).tobytes(),
                    "aes_nonce":        ndarrays[1].astype(np.uint8).tobytes(),
                    "encrypted_weights": ndarrays[2].astype(np.uint8).tobytes(),
                }
                print(f" -> Unlocking payload from Client {client.cid}...")
                decrypted_weights = self.crypto_engine.decapsulate_and_decrypt(secure_payload)
                decrypted_weights_with_sizes.append((decrypted_weights, fit_res.num_examples))

            else:
                print(f" -> WARNING: Unexpected payload from Client {client.cid}. Rejecting.")
                continue

            # Log per-client metrics to CSV
            if self.csv_log and fit_res.metrics:
                self._append_row(server_round, fit_res.metrics)

        print(f"[Server Round {server_round}] Aggregating weights...")
        aggregated_ndarrays = aggregate(decrypted_weights_with_sizes)
        return ndarrays_to_parameters(aggregated_ndarrays), {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Intercepts per-client evaluation results after each FL round.
        Computes FedAvg-weighted global loss and accuracy, then logs
        both aggregate and per-client metrics to the CSV.
        """
        if not results:
            return None, {}

        total_examples: int = sum(eval_res.num_examples for _, eval_res in results)
        weighted_loss: float = sum(
            eval_res.loss * eval_res.num_examples for _, eval_res in results
        ) / total_examples
        weighted_accuracy: float = sum(
            eval_res.metrics.get("accuracy", 0.0) * eval_res.num_examples
            for _, eval_res in results
        ) / total_examples

        print(f"[Server Round {server_round}] Global — "
              f"loss={weighted_loss:.4f}, accuracy={weighted_accuracy:.4f}")

        if self.csv_log:
            # Aggregate row
            self._append_row(server_round, {
                "client_id": "aggregate",
                "global_loss": weighted_loss,
                "global_accuracy": weighted_accuracy,
                "eval_client_id": "aggregate",
                "eval_loss": weighted_loss,
                "eval_accuracy": weighted_accuracy,
            })
            # Per-client rows
            for client, eval_res in results:
                self._append_row(server_round, {
                    "client_id": eval_res.metrics.get("client_id", client.cid),
                    "global_loss": weighted_loss,
                    "global_accuracy": weighted_accuracy,
                    "eval_client_id": client.cid,
                    "eval_loss": eval_res.loss,
                    "eval_accuracy": eval_res.metrics.get("accuracy", ""),
                })

        return weighted_loss, {"accuracy": weighted_accuracy}

    def _append_row(self, server_round: int, metrics: dict) -> None:
        row = {
            "condition": self.condition,
            "run": self.run_id,
            "server_round": server_round,
            **{k: metrics.get(k, "") for k in CSV_FIELDNAMES
               if k not in ("condition", "run", "server_round")},
        }
        with open(self.csv_log, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore").writerow(row)


class SecureQuantumFedProx(fl.server.strategy.FedProx):
    def __init__(self, *args, proximal_mu: float = 0.1, no_pqc: bool = False,
                 csv_log: Optional[str] = None, condition: str = "", run_id: int = 0,
                 local_epochs: int = 2, **kwargs):
        super().__init__(*args, proximal_mu=proximal_mu, **kwargs)
        self.no_pqc = no_pqc
        self.csv_log = csv_log
        self.condition = condition
        self.run_id = run_id
        self.local_epochs = local_epochs

        if not no_pqc:
            self.crypto_engine = KyberManager(alg_name="ML-KEM-512")
            print("\n[Server] ML-KEM-512 engine ready. Keypairs generated per-round (PFS).")
        else:
            self.crypto_engine = None
            print("\n[Server] PQC disabled — plaintext baseline mode.")

        if csv_log:
            write_header = not os.path.exists(csv_log)
            with open(csv_log, "a", newline="") as f:
                if write_header:
                    csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writeheader()

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: fl.server.client_manager.ClientManager
                      ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Phase 1: Broadcast. Generates a fresh ephemeral keypair each round (PFS) unless no_pqc."""
        config_schedule = super().configure_fit(server_round, parameters, client_manager)

        if not self.no_pqc:
            round_public_key: bytes = self.crypto_engine.generate_keypair()
            print(f"[Server Round {server_round}] Fresh ML-KEM-512 keypair generated (PFS).")

        for client, fit_ins in config_schedule:
            if not self.no_pqc:
                fit_ins.config["server_public_key"] = round_public_key
            fit_ins.config["local_epochs"] = self.local_epochs

        return config_schedule

    def aggregate_fit(self, server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Phase 2: Secure Aggregation. Decrypts (if PQC), aggregates, and logs metrics."""
        if not results:
            return None, {}

        print(f"\n[Server Round {server_round}] Received payloads from {len(results)} clients.")
        decrypted_weights_with_sizes = []

        for client, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)

            if self.no_pqc:
                # Plaintext baseline — weights arrive directly
                decrypted_weights_with_sizes.append((ndarrays, fit_res.num_examples))

            elif len(ndarrays) == 3:
                # PQC mode — reconstruct and decrypt
                secure_payload = {
                    "kyber_ciphertext": ndarrays[0].astype(np.uint8).tobytes(),
                    "aes_nonce":        ndarrays[1].astype(np.uint8).tobytes(),
                    "encrypted_weights": ndarrays[2].astype(np.uint8).tobytes(),
                }
                print(f" -> Unlocking payload from Client {client.cid}...")
                decrypted_weights = self.crypto_engine.decapsulate_and_decrypt(secure_payload)
                decrypted_weights_with_sizes.append((decrypted_weights, fit_res.num_examples))

            else:
                print(f" -> WARNING: Unexpected payload from Client {client.cid}. Rejecting.")
                continue

            # Log per-client metrics to CSV
            if self.csv_log and fit_res.metrics:
                self._append_row(server_round, fit_res.metrics)

        print(f"[Server Round {server_round}] Aggregating weights...")
        aggregated_ndarrays = aggregate(decrypted_weights_with_sizes)
        return ndarrays_to_parameters(aggregated_ndarrays), {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Intercepts per-client evaluation results after each FL round.
        Computes FedAvg-weighted global loss and accuracy, then logs
        both aggregate and per-client metrics to the CSV.
        """
        if not results:
            return None, {}

        total_examples: int = sum(eval_res.num_examples for _, eval_res in results)
        weighted_loss: float = sum(
            eval_res.loss * eval_res.num_examples for _, eval_res in results
        ) / total_examples
        weighted_accuracy: float = sum(
            eval_res.metrics.get("accuracy", 0.0) * eval_res.num_examples
            for _, eval_res in results
        ) / total_examples

        print(f"[Server Round {server_round}] Global — "
              f"loss={weighted_loss:.4f}, accuracy={weighted_accuracy:.4f}")

        if self.csv_log:
            # Aggregate row
            self._append_row(server_round, {
                "client_id": "aggregate",
                "global_loss": weighted_loss,
                "global_accuracy": weighted_accuracy,
                "eval_client_id": "aggregate",
                "eval_loss": weighted_loss,
                "eval_accuracy": weighted_accuracy,
            })
            # Per-client rows
            for client, eval_res in results:
                self._append_row(server_round, {
                    "client_id": eval_res.metrics.get("client_id", client.cid),
                    "global_loss": weighted_loss,
                    "global_accuracy": weighted_accuracy,
                    "eval_client_id": client.cid,
                    "eval_loss": eval_res.loss,
                    "eval_accuracy": eval_res.metrics.get("accuracy", ""),
                })

        return weighted_loss, {"accuracy": weighted_accuracy}

    def _append_row(self, server_round: int, metrics: dict) -> None:
        row = {
            "condition": self.condition,
            "run": self.run_id,
            "server_round": server_round,
            **{k: metrics.get(k, "") for k in CSV_FIELDNAMES
               if k not in ("condition", "run", "server_round")},
        }
        with open(self.csv_log, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore").writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum-Safe 6G FL Server")
    parser.add_argument("--rounds",    type=int,  default=20,    help="Number of FL rounds")
    parser.add_argument("--no-pqc",   action="store_true",       help="Disable PQC (plaintext baseline)")
    parser.add_argument("--csv-log",  type=str,  default=None,   help="Path to append metrics CSV")
    parser.add_argument("--condition", type=str, default="",     help="Benchmark condition label")
    parser.add_argument("--run",      type=int,  default=0,      help="Benchmark run index")
    parser.add_argument("--fedprox",  action="store_true",       help="Use FedProx aggregation instead of FedAvg")
    parser.add_argument("--proximal-mu", type=float, default=0.1, help="FedProx proximal term mu (default: 0.1)")
    parser.add_argument("--clients", type=int, default=2,         help="Number of FL clients (default: 2)")
    parser.add_argument("--local-epochs", type=int, default=2,   help="Local epochs per round (default: 2)")
    args = parser.parse_args()

    if args.fedprox:
        strategy = SecureQuantumFedProx(
            fraction_fit=1.0,
            min_fit_clients=args.clients,
            min_available_clients=args.clients,
            proximal_mu=args.proximal_mu,
            no_pqc=args.no_pqc,
            csv_log=args.csv_log,
            condition=args.condition,
            run_id=args.run,
            local_epochs=args.local_epochs,
        )
        print(f"[Server] Aggregation: FedProx (mu={args.proximal_mu})")
    else:
        strategy = SecureQuantumStrategy(
            fraction_fit=1.0,
            min_fit_clients=args.clients,
            min_available_clients=args.clients,
            no_pqc=args.no_pqc,
            csv_log=args.csv_log,
            condition=args.condition,
            run_id=args.run,
            local_epochs=args.local_epochs,
        )
        print("[Server] Aggregation: FedAvg")

    print(f"Starting FL Server — {args.rounds} rounds, "
          f"PQC={'OFF' if args.no_pqc else 'ON'}, "
          f"Aggregation={'FedProx' if args.fedprox else 'FedAvg'}, "
          f"Clients={args.clients}, port 8080...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
