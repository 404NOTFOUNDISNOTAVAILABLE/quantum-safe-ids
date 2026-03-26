import flwr as fl

def main():
    fl.server.start_server(
        server_address="localhost:8080",
        # Remove config param or specify properly per API docs
    )

if __name__ == "__main__":
    main()
