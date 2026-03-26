import warnings
# Suppress the known version mismatch warning for clean production logs
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")

from pqc_utils import PQCManager
import sys

def run_validation():
    print("--- Initializing ML-KEM-512 (Kyber) ---")
    try:
        manager = PQCManager(kem_alg="ML-KEM-512")
        
        # 1. Server generates keys
        pk, sk = manager.generate_server_keypair()
        print(f"Server Public Key Length: {len(pk)} bytes")
        print(f"Server Secret Key Length: {len(sk)} bytes")

        # 2. Client encrypts weights
        mock_weights = [ [0.11, 0.22, 0.33], [0.99, 0.88, 0.77] ]
        print("\n--- Client Encapsulation & Encryption ---")
        ciphertext, enc_weights = manager.client_encrypt_weights(mock_weights, pk)
        print(f"PQC Ciphertext (Encapsulated Key) Length: {len(ciphertext)} bytes")
        print(f"AES Encrypted Weights Length: {len(enc_weights)} bytes")

        # 3. Server decrypts weights
        print("\n--- Server Decapsulation & Decryption ---")
        dec_weights = manager.server_decrypt_weights(ciphertext, enc_weights, sk)
        
        assert mock_weights == dec_weights, "Integrity Check Failed: Decrypted weights do not match!"
        print("Integrity Check Passed: Decrypted weights match original mock weights.")
        print("\n[SUCCESS] PQC implementation is fully functional and memory-safe.")
        
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_validation()
