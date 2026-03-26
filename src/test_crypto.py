from crypto.kyber_manager import KyberManager
import numpy as np

def run_quantum_test():
    print("--- Starting Hybrid Quantum KEM Test ---")
    
    # Simulate a dummy CNN weight layer (e.g., 500 floats)
    dummy_weights = [np.random.rand(500).astype(np.float32)]
    print(f"Original Weights (Sample): {dummy_weights[0][:3]}...\n")

    # 1. Server initialization
    print("[Server] Generating Kyber-512 Keypair...")
    server_crypto = KyberManager()
    server_pub_key = server_crypto.generate_keypair()

    # 2. Client initialization & Encryption
    print("[Client] Encapsulating Secret & Encrypting Weights...")
    client_crypto = KyberManager()
    secure_payload = client_crypto.encapsulate_and_encrypt(server_pub_key, dummy_weights)
    
    print(f"   -> Kyber Ciphertext Size: {len(secure_payload['kyber_ciphertext'])} bytes")
    print(f"   -> AES Encrypted Weights Size: {len(secure_payload['encrypted_weights'])} bytes\n")

    # 3. Server Decryption
    print("[Server] Decapsulating Secret & Decrypting Weights...")
    decrypted_weights = server_crypto.decapsulate_and_decrypt(secure_payload)
    print(f"Decrypted Weights (Sample): {decrypted_weights[0][:3]}...")

    # Verification
    if np.allclose(dummy_weights[0], decrypted_weights[0]):
        print("\nSUCCESS! Quantum-Safe Hybrid Encryption is fully operational.")
    else:
        print("\nFAILED: Decrypted weights do not match.")

    server_crypto.close()
    client_crypto.close()

if __name__ == "__main__":
    run_quantum_test()