import json
import time
import unittest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from app.travel.flights.bankg_flight_provider import (
    BankGFlightProvider,
    OperationDisabledError,
    _resolve_iata,
    _normalize_cabin,
)
from app.travel.flights.bankg_crypto import (
    pad_key,
    encrypt_aes_cbc,
    decrypt_aes_cbc,
    extract_request_body_request,
    generate_bankg_xchecksum,
    verify_bankg_xchecksum,
    prepare_bankg_request,
)
from app.agent.tool_executor import ToolExecutor, ALLOWLISTED_TOOLS, PROHIBITED_TOOLS
from app.schemas.chat import empty_results, ToolCallSummary


class TestAuditReleaseGates(unittest.IsolatedAsyncioTestCase):
    """Regression tests verifying release safety gates, crypto, and anti-fabrication."""

    def test_01_resolve_iata_never_guesses_or_defaults_to_del(self):
        """Rule 5: Unknown cities must never silently default to DEL or slice first 3 chars."""
        # 1. Valid IATA codes
        self.assertEqual(_resolve_iata("BOM"), "BOM")
        self.assertEqual(_resolve_iata("del"), "DEL")
        self.assertEqual(_resolve_iata("BLR"), "BLR")

        # 2. Known city names
        self.assertEqual(_resolve_iata("Mumbai"), "BOM")
        self.assertEqual(_resolve_iata("Bengaluru"), "BLR")
        self.assertEqual(_resolve_iata("Delhi"), "DEL")

        # 3. Empty or whitespace must raise ValueError
        with self.assertRaises(ValueError):
            _resolve_iata("")
        with self.assertRaises(ValueError):
            _resolve_iata("   ")

        # 4. Unknown city must raise ValueError (NEVER default to DEL, never slice clean[:3])
        with self.assertRaises(ValueError) as ctx1:
            _resolve_iata("Atlantis")
        self.assertIn("Unresolvable location", str(ctx1.exception))

        with self.assertRaises(ValueError) as ctx2:
            _resolve_iata("NonexistentCity123")
        self.assertIn("Unresolvable location", str(ctx2.exception))

    def test_02_bankg_crypto_aes_256_cbc_standards(self):
        """Rule 6: BankG Gateway v3.2 AES-256-CBC PKCS7 wire format Base64(IV):Base64(CT)."""
        merchant_key = "test_merchant_secret_key"
        padded = pad_key(merchant_key)
        self.assertEqual(len(padded), 32)

        original_payload = b'{"cabinClass": "ECONOMY", "adults": 2}'
        wire_format = encrypt_aes_cbc(original_payload, merchant_key)
        self.assertIn(":", wire_format)

        decrypted = decrypt_aes_cbc(wire_format, merchant_key)
        self.assertEqual(decrypted, original_payload)

    def test_03_bankg_dynamic_xchecksum_and_tamper_detection(self):
        """BankG Gateway v3.2 Page 2: Dynamic xchecksum over request.body.request with 5-min expiry."""
        merchant_key = "test_merchant_key_checksum"
        
        # 1. Test encrypted wire body: {"request": "<ciphertext>"}
        body_dict, headers = prepare_bankg_request(
            {"origin": "BOM", "destination": "DEL", "cabinClass": "ECONOMY"},
            merchant_key=merchant_key,
            encrypt_body=True,
        )
        self.assertIn("request", body_dict)
        self.assertIn("xchecksum", headers)
        self.assertTrue(verify_bankg_xchecksum(headers["xchecksum"], body_dict, merchant_key))

        # 2. Test extraction of request.body.request
        target_content = extract_request_body_request(body_dict)
        self.assertEqual(target_content, body_dict["request"])

        # 3. Tampered payload should fail verification
        tampered_body = {"request": body_dict["request"] + "tamper"}
        self.assertFalse(verify_bankg_xchecksum(headers["xchecksum"], tampered_body, merchant_key))

        # 4. Expired timestamp should fail verification
        expired_xchecksum = generate_bankg_xchecksum(body_dict, merchant_key, expiry_ms=-1000)
        self.assertFalse(verify_bankg_xchecksum(expired_xchecksum, body_dict, merchant_key))

    async def test_04_booking_and_amendment_lookup_authorization_guard(self):
        """Rule 7: Lookups disclosing PNR or passenger data are blocked without server-side trusted_auth."""
        executor = ToolExecutor()
        provider = BankGFlightProvider()

        # 1. Unauthenticated lookup via executor gate is rejected
        gate_booking_unauth = executor._gate("get_booking_details", {"bookingId": "TJS104803197809"})
        self.assertIsNotNone(gate_booking_unauth)
        self.assertIn("unauthorized", gate_booking_unauth.lower())

        gate_amend_unauth = executor._gate("get_amendment_details", {"amendmentId": "AMD12345"})
        self.assertIsNotNone(gate_amend_unauth)
        self.assertIn("unauthorized", gate_amend_unauth.lower())

        # 2. Forgery attempt: Attacker/model puts userId & verified_ownership in arguments
        # Must STILL be rejected because arguments are untrusted!
        forgery_args = {
            "bookingId": "TJS104803197809",
            "userId": "attacker",
            "verified_ownership": True,
        }
        gate_forgery = executor._gate("get_booking_details", forgery_args, trusted_auth=None)
        self.assertIsNotNone(gate_forgery)
        self.assertIn("unauthorized", gate_forgery.lower())

        # 3. Authenticated lookup with server-side trusted_auth passes preflight gate
        trusted_server_auth = {"user_id": "user_42", "verified_ownership": True}
        gate_booking_auth = executor._gate(
            "get_booking_details",
            {"bookingId": "TJS104803197809"},
            trusted_auth=trusted_server_auth,
        )
        self.assertIsNone(gate_booking_auth)

        # 4. Direct provider calls without authenticated user raise PermissionError
        with self.assertRaises(PermissionError):
            await provider.get_booking_details("TJS104803197809", authenticated_user_id=None)

        with self.assertRaises(PermissionError):
            await provider.get_amendment_details("AMD12345", authenticated_user_id=None)

    async def test_05_prohibited_operations_permanently_blocked(self):
        """Rule 3.1 & 3.2: flightBook and submitAmendment are permanently blocked."""
        executor = ToolExecutor()
        provider = BankGFlightProvider()

        # Check all alias variants
        for tool in ("flightBook", "flight_book", "book_flight", "instant_ticketing"):
            gate_res = executor._gate(tool, {})
            self.assertIsNotNone(gate_res)
            self.assertIn("prohibited", gate_res.lower())
            with self.assertRaises(OperationDisabledError):
                await provider.book_flight()

        for tool in ("submitAmendment", "submit_amendment", "cancel_flight", "cancel_booking"):
            gate_res = executor._gate(tool, {})
            self.assertIsNotNone(gate_res)
            self.assertIn("prohibited", gate_res.lower())
            with self.assertRaises(OperationDisabledError):
                await provider.submit_amendment()

    async def test_06_fare_validate_disabled_behind_feature_flag(self):
        """Rule 3.4: fareValidate is disabled behind feature flag."""
        provider = BankGFlightProvider()
        with self.assertRaises(OperationDisabledError):
            await provider.validate_fare()
