import oqs
import pickle
import base64
import hashlib
from cryptography.fernet import Fernet
from typing import Tuple, Any

class PQCManager:
    """
    Handles Post-Quantum Cryptographic operations for Federated Learning.
    Utilizes ML-KEM-512 (Kyber-512) for secure key encapsulation and 
    AES-128 (via Fernet) for symmetric weight encryption.
    """
    def __init__(self, kem_alg: str = "ML-KEM-512"):
        self.kem_alg = kem_alg
        
    def generate_server_keypair(self) -> Tuple[bytes, bytes]:
        """ Generates the PQC keypair on the central server. """
        with oqs.KeyEncapsulation(self.kem_alg) as kem:
            public_key = kem.generate_keypair()
            secret_key = kem.export_secret_key()
            return public_key, secret_key

    def client_encrypt_weights(self, weights: Any, server_public_key: bytes) -> Tuple[bytes, bytes]:
        """
        Executed on Edge Devices:
        1. Encapsulates a shared secret using the server's public PQC key.
        2. Encrypts the model weights symmetrically using the shared secret.
        """
        with oqs.KeyEncapsulation(self.kem_alg) as kem:
            # 1. PQC Encapsulation
            ciphertext, shared_secret = kem.encap_secret(server_public_key)
            
            # 2. Derive a valid 32-byte URL-safe base64 key for Fernet (AES)
            derived_key = base64.urlsafe_b64encode(hashlib.sha256(shared_secret).digest())
            cipher_suite = Fernet(derived_key)
            
            # 3. Serialize and Encrypt the Weights
            serialized_weights = pickle.dumps(weights)
            encrypted_weights = cipher_suite.encrypt(serialized_weights)
            
            return ciphertext, encrypted_weights

    def server_decrypt_weights(self, ciphertext: bytes, encrypted_weights: bytes, server_secret_key: bytes) -> Any:
        """
        Executed on Central Server:
        1. Decapsulates the shared secret using the PQC secret key.
        2. Decrypts the model weights.
        """
        with oqs.KeyEncapsulation(self.kem_alg, server_secret_key) as kem:
            # 1. PQC Decapsulation
            shared_secret = kem.decap_secret(ciphertext)
            
            # 2. Derive the corresponding AES key
            derived_key = base64.urlsafe_b64encode(hashlib.sha256(shared_secret).digest())
            cipher_suite = Fernet(derived_key)
            
            # 3. Decrypt and Deserialize
            decrypted_bytes = cipher_suite.decrypt(encrypted_weights)
            weights = pickle.loads(decrypted_bytes)
            
            return weights
