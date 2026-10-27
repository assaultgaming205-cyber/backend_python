import os
import time
import json
import base64
import hashlib
import logging
from typing import Dict, Any, Tuple, Optional, Union

try:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad, unpad
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

logger = logging.getLogger("bankg_crypto")


def pad_key(merchant_key: str) -> bytes:
    """
    Pad the merchant key to 32 bytes (256-bit) as per Gateway v3.2 Section 1:
    'The merchant key is padded to 32 bytes (256-bit).'
    """
    if not merchant_key:
        raise ValueError("Merchant key cannot be empty for BankG encryption.")
    b = merchant_key.encode("utf-8")
    if len(b) > 32:
        return b[:32]
    return b.ljust(32, b"\0")


def encrypt_aes_cbc(data: bytes, merchant_key: str) -> str:
    """
    BankG Gateway v3.2 Page 2, Section 1:
    AES-256-CBC with PKCS7 padding and a 16-byte random IV.
    Wire format: Base64(IV) + ":" + Base64(Ciphertext)
    """
    if not HAS_CRYPTO:
        raise RuntimeError("pycryptodome is required for BankG AES-256-CBC encryption.")
    key = pad_key(merchant_key)
    iv = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(data, AES.block_size))
    iv_b64 = base64.b64encode(iv).decode("ascii")
    ct_b64 = base64.b64encode(ciphertext).decode("ascii")
    return f"{iv_b64}:{ct_b64}"


def decrypt_aes_cbc(wire_format: str, merchant_key: str) -> bytes:
    """
    Decrypts BankG AES-256-CBC payload formatted as Base64(IV) + ":" + Base64(Ciphertext).
    """
    if not HAS_CRYPTO:
        raise RuntimeError("pycryptodome is required for BankG AES-256-CBC decryption.")
    if ":" not in wire_format:
        raise ValueError("Invalid wire format: expected Base64(IV):Base64(Ciphertext)")
    iv_b64, ct_b64 = wire_format.split(":", 1)
    iv = base64.b64decode(iv_b64)
    ciphertext = base64.b64decode(ct_b64)
    key = pad_key(merchant_key)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ciphertext), AES.block_size)


def extract_request_body_request(body: Union[Dict[str, Any], str, bytes]) -> str:
    """
    BankG Gateway v3.2 Page 2, Section 2:
    'Compute SHA-256 over request.body.request'
    If body is a JSON dict with key 'request', extract body['request'].
    If body is a JSON string or bytes, extract 'request' if present, otherwise use string representation.
    """
    if isinstance(body, dict):
        req_val = body.get("request")
        if req_val is not None:
            return req_val if isinstance(req_val, str) else json.dumps(req_val)
        return json.dumps(body)
    elif isinstance(body, bytes):
        try:
            parsed = json.loads(body.decode("utf-8"))
            if isinstance(parsed, dict) and "request" in parsed:
                req_val = parsed["request"]
                return req_val if isinstance(req_val, str) else json.dumps(req_val)
        except Exception:
            pass
        return body.decode("utf-8", errors="replace")
    elif isinstance(body, str):
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and "request" in parsed:
                req_val = parsed["request"]
                return req_val if isinstance(req_val, str) else json.dumps(req_val)
        except Exception:
            pass
        return body
    return str(body)


def generate_bankg_xchecksum(
    body: Union[Dict[str, Any], str, bytes],
    merchant_key: str,
    expiry_ms: int = 300000,
) -> str:
    """
    BankG Gateway v3.2 Page 2, Section 2:
    'Every encrypted request requires an 'xchecksum' HTTP header. Compute SHA-256 over
    request.body.request, package into metadata {"checksum": hash, "timeStamp": Date.now() + 300000}
    (5-minute expiry window), and encrypt it using AES-256-CBC with the shared key.'
    """
    target_content = extract_request_body_request(body)
    sha256_hash = hashlib.sha256(target_content.encode("utf-8")).hexdigest()
    expiry_timestamp = int(time.time() * 1000) + expiry_ms
    meta_json = json.dumps({"checksum": sha256_hash, "timeStamp": expiry_timestamp})
    return encrypt_aes_cbc(meta_json.encode("utf-8"), merchant_key)


def verify_bankg_xchecksum(
    xchecksum_header: str,
    body: Union[Dict[str, Any], str, bytes],
    merchant_key: str,
) -> bool:
    """
    Verifies that an incoming xchecksum matches SHA-256(request.body.request) and has not expired.
    Rejects tampered or expired checksums as per BankG Gateway v3.2 Page 2.
    """
    try:
        decrypted_meta_bytes = decrypt_aes_cbc(xchecksum_header, merchant_key)
        meta = json.loads(decrypted_meta_bytes.decode("utf-8"))
        target_content = extract_request_body_request(body)
        expected_hash = hashlib.sha256(target_content.encode("utf-8")).hexdigest()

        if meta.get("checksum") != expected_hash:
            logger.warning("xchecksum hash mismatch against request.body.request")
            return False
        current_time_ms = int(time.time() * 1000)
        if meta.get("timeStamp", 0) < current_time_ms:
            logger.warning("xchecksum expired timestamp")
            return False
        return True
    except Exception as e:
        logger.error(f"Error verifying xchecksum: {e}")
        return False


def prepare_bankg_request(
    payload: Dict[str, Any],
    merchant_key: str,
    encrypt_body: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Prepares request body and headers strictly adhering to BankG Gateway v3.2 Page 2.
    If encrypt_body is True:
        body = {"request": encrypt_aes_cbc(json(payload))}
        xchecksum = generate_bankg_xchecksum(body["request"])
    If encrypt_body is False (cURL examples mode):
        body = payload
        xchecksum = generate_bankg_xchecksum(body)
    """
    if encrypt_body:
        payload_bytes = json.dumps(payload).encode("utf-8")
        encrypted_wire = encrypt_aes_cbc(payload_bytes, merchant_key)
        body = {"request": encrypted_wire}
        xchecksum = generate_bankg_xchecksum(body, merchant_key)
    else:
        body = payload
        xchecksum = generate_bankg_xchecksum(body, merchant_key)

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "xchecksum": xchecksum,
    }
    return body, headers
