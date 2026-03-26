import oqs
import os
import pickle
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class KyberManager:
    def __init__(self, alg_name: str = "ML-KEM-512"):
        """
        Initializes the Post-Quantum KEM (Key Encapsulation Mechanism).
        Defaulting to NIST-standardized Kyber-512.
        """
        self.alg_name = alg_name
        # The C-library wrapper object
        self.kem = oqs.KeyEncapsulation(alg_name) 
        self.public_key = None
        # Note: The private key is strictly kept in C-memory by liboqs for security.

    def generate_keypair(self) -> bytes:
        """
        Server-side: Generates a Post-Quantum Public/Private key pair.
        Returns the Public Key to be broadcasted to Edge Clients.
        """
        self.public_key = self.kem.generate_keypair()
        return self.public_key

    def encapsulate_and_encrypt(
        self,
        target_public_key: bytes,
        model_weights: list
    ) -> dict:
        """
        Client-side hybrid encryption:
          1. ML-KEM-512 encapsulation (key agreement) — timed separately
          2. AES-GCM symmetric encryption of serialized weights — timed separately
        Sub-timings are exposed in the returned dict for IEEE metrics table.

        Args:
            target_public_key: Server's ML-KEM-512 public key bytes.
            model_weights: List of NumPy weight arrays from the local model.

        Returns:
            Dict with keys: kyber_ciphertext, aes_nonce, encrypted_weights,
            original_size_bytes, encap_latency_ns, sym_enc_latency_ns.
        """
        import time

        # 1. ML-KEM-512 encapsulation
        encap_start_ns: int = time.perf_counter_ns()
        ciphertext_kyber, shared_secret = self.kem.encap_secret(target_public_key)
        encap_latency_ns: int = time.perf_counter_ns() - encap_start_ns

        # 2. Serialize weights
        weights_bytes: bytes = pickle.dumps(model_weights)

        # 3. AES-GCM symmetric encryption
        sym_start_ns: int = time.perf_counter_ns()
        aesgcm = AESGCM(shared_secret)
        nonce: bytes = os.urandom(12)
        encrypted_weights: bytes = aesgcm.encrypt(nonce, weights_bytes, None)
        sym_enc_latency_ns: int = time.perf_counter_ns() - sym_start_ns

        return {
            "kyber_ciphertext": ciphertext_kyber,
            "aes_nonce": nonce,
            "encrypted_weights": encrypted_weights,
            "original_size_bytes": len(weights_bytes),
            "encap_latency_ns": encap_latency_ns,
            "sym_enc_latency_ns": sym_enc_latency_ns,
        }

    def decapsulate_and_decrypt(self, secure_payload: dict) -> list:
        """
        Server-side: Uses its internal Private Key to unlock the Kyber secret, 
        then uses that secret to decrypt the AES weights.
        """
        # 1. Quantum Decapsulation: Unlock the shared secret
        shared_secret = self.kem.decap_secret(secure_payload["kyber_ciphertext"])
        
        # 2. Symmetric Decryption: Unlock the model weights
        aesgcm = AESGCM(shared_secret)
        decrypted_bytes = aesgcm.decrypt(
            secure_payload["aes_nonce"], 
            secure_payload["encrypted_weights"], 
            None
        )
        
        # 3. Deserialize back into Python lists/arrays
        model_weights = pickle.loads(decrypted_bytes)
        return model_weights

    def close(self):
        """Safely clears the cryptographic keys from the computer's memory."""
        self.kem.free()