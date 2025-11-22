import flwr as fl
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Server config
ROUNDS = 3
MIN_CLIENTS = 2

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )
