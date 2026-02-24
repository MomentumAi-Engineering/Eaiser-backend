"""
TOTP Secret Encryption Utility
Encrypts/decrypts 2FA TOTP secrets before storing in MongoDB.
Uses Fernet symmetric encryption (from the cryptography library).
"""
import os
import logging
from cryptography.fernet import Fernet
import base64
import hashlib

logger = logging.getLogger(__name__)

# Derive a Fernet key from SECRET_KEY (must be 32 url-safe base64-encoded bytes)
_raw_key = os.getenv("SECRET_KEY", "fallback-secret-key-for-encryption")
_key_hash = hashlib.sha256(_raw_key.encode()).digest()
FERNET_KEY = base64.urlsafe_b64encode(_key_hash)

_fernet = Fernet(FERNET_KEY)


def encrypt_totp_secret(plaintext_secret: str) -> str:
    """Encrypt a TOTP secret before storing in database."""
    try:
        encrypted = _fernet.encrypt(plaintext_secret.encode("utf-8"))
        return encrypted.decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encrypt TOTP secret: {e}")
        raise


def decrypt_totp_secret(encrypted_secret: str) -> str:
    """Decrypt a TOTP secret retrieved from database."""
    try:
        decrypted = _fernet.decrypt(encrypted_secret.encode("utf-8"))
        return decrypted.decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to decrypt TOTP secret: {e}")
        raise


def is_encrypted(value: str) -> bool:
    """Check if a value looks like a Fernet-encrypted string."""
    try:
        # Fernet tokens start with 'gAAAAA'
        return value.startswith("gAAAAA") and len(value) > 50
    except Exception:
        return False
