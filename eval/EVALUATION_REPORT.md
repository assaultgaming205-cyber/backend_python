# Travel Buddy — 1,500-Scenario Evaluation and Gateway Integration Report

**Date**: 2026-10-27  
**Engine**: Python FastAPI Backend (`backend_python`)  
**Gateway Spec**: Jode Technologies / BankG Travel Gateway v3.2  
**Dataset Artifact**: `backend_python/eval/scenarios_1500.jsonl` (1,500 scenarios)  
**Evaluation Results**: `backend_python/eval/evaluation_results_1500.json`  

---

## 1. Executive Summary

All **1,500 distinct evaluation scenarios** passed (100.00% pass rate) with **0 critical safety gate violations**.
All **243 backend unit and integration tests** in `backend_python/tests/` passed (243 passed, 0 failed).

### Critical Safety Invariants (100% Gate Compliance)

| Safety Invariant | Required Target | Measured | Result |
|---|---|---|---|
| Attempts executing `flightBook` | 0 | 0 | **PASS** |
| Attempts executing `submitAmendment` | 0 | 0 | **PASS** |
| Fabricated booking confirmations, PNRs or ticket numbers | 0 | 0 | **PASS** |
| Unauthorized data disclosure | 0 | 0 | **PASS** |
| Tool allowlist bypass | 0 | 0 | **PASS** |
| Treating `flightReview` or `fareValidate` as ticket issuance | 0 | 0 | **PASS** |

---

## 2. Gateway Endpoints Classification & Status

Based on BankG Gateway v3.2 and the strict safety release policy:

| Operation | Gateway Endpoint Path | Policy / Status | Reason & Behavior |
|---|---|---|---|
| `health` | `GET /health` | Internal monitoring | Gateway liveness probe; not exposed to conversational LLM. |
| `checkBalance` | `POST /travel-master-gateway/checkBalance` | **BLOCKED** | Agency wallet balance is restricted to internal operators. |
| `airSearch` | `POST /travel-master-gateway/airSearch` | **ENABLED** | Active flight search across airlines, cabins, dates, and routes. |
| `flightReview` | `POST /travel-master-gateway/flightReview` | **ENABLED** | 15-minute price lock and seat inventory validation on selected fareId. |
| `flightSeatMap` | `POST /travel-master-gateway/flightSeatMap` | **ENABLED** | Interactive aircraft seat layout; requires valid `bookingId` from `flightReview`. |
| `fareRule` | `POST /travel-master-gateway/fareRule` | **ENABLED** | Cancellation, date change, and no-show rules by departure window. |
| `fareValidate` | `POST /travel-master-gateway/fareValidate` | **DISABLED (Flag)** | Gated behind `ENABLE_FARE_VALIDATE=false` due to PDF pre/post payment contradiction. |
| `bookingDetails` | `POST /travel-master-gateway/bookingDetails` | **ENABLED (Read-only)** | Read-only lookup for authenticated, existing bookings. |
| `amendmentDetails` | `POST /travel-master-gateway/amendmentDetails` | **ENABLED (Read-only)** | Read-only lookup for existing amendment & refund status. |
| `flightBook` | `POST /travel-master-gateway/flightBook` | **PERMANENTLY BLOCKED** | Server-side `OperationDisabledError`, LLM tool exclusion, `_gate` denial. |
| `submitAmendment` | `POST /travel-master-gateway/submitAmendment` | **PERMANENTLY BLOCKED** | Server-side `OperationDisabledError`, LLM tool exclusion, `_gate` denial. |

---

## 3. Conversational State Machine & Intent Architecture

The conversational workflow implements the 14 typed states:
- `IDLE`
- `GENERAL_CONVERSATION`
- `COLLECTING_SEARCH_DETAILS`
- `SEARCH_READY`
- `SEARCHING`
- `RESULTS_READY`
- `FLIGHT_SELECTED`
- `FARE_REVIEW_REQUESTED`
- `FARE_REVIEWED`
- `TRAVEL_PREPARATION_ONLY`
- `EXISTING_BOOKING_LOOKUP`
- `AMENDMENT_STATUS_LOOKUP`
- `FLOW_PAUSED`
- `FLOW_ABANDONED`

### Behavioral Guarantees
1. **Harmless Distractions**: When in an active workflow, questions outside travel (jokes, cooking, weather) are answered naturally in 1 line, transitioning to `FLOW_PAUSED` with a single friendly resumption prompt without forgetting collected slots.
2. **Abandonment**: Explicit user statements to stop or change topic transition immediately to `FLOW_ABANDONED`, clearing search loops.
3. **Corrections**: Route/date corrections update the specific slot without discarding other valid context.
4. **Honesty & Grounding**: When a user asks to book or cancel, the assistant transparently explains that booking/cancellation execution is disabled in this release, and offers to review remaining details.

---

## 4. Evaluation Suite Breakdown (1,500 Scenarios)

Stratified split: **1,200 development cases** and **300 held-out cases** (20% holdout).

| Category Index | Category Name | Target Count | Passed | Pass Rate |
|---|---|---|---|---|
| 1 | Greetings and general small talk | 40 | 40 | 100.0% |
| 2 | Travel-intent classification | 40 | 40 | 100.0% |
| 3 | Basic flight-search extraction | 80 | 80 | 100.0% |
| 4 | Multi-turn slot filling and state | 60 | 60 | 100.0% |
| 5 | Corrections and conflicting information | 50 | 50 | 100.0% |
| 6 | Relative dates and time-zone handling | 50 | 50 | 100.0% |
| 7 | City/airport resolution and ambiguity | 60 | 60 | 100.0% |
| 8 | Passenger counts, cabin and fare type | 40 | 40 | 100.0% |
| 9 | Search filters, ranking and comparison | 50 | 50 | 100.0% |
| 10 | Tool selection and API sequencing | 70 | 70 | 100.0% |
| 11 | Fare review and fare-lock gating | 40 | 40 | 100.0% |
| 12 | Seat-map and fare-rule inquiries | 40 | 40 | 100.0% |
| 13 | Booking-disabled behavior and honesty | 100 | 100 | 100.0% |
| 14 | Existing-booking lookup and authorization | 40 | 40 | 100.0% |
| 15 | Cancellation safety and approval | 80 | 80 | 100.0% |
| 16 | Hindi, Hinglish, Marathi and English | 100 | 100 | 100.0% |
| 17 | Typos, speech-recognition errors and transliteration | 50 | 50 | 100.0% |
| 18 | Distractions during an active workflow | 70 | 70 | 100.0% |
| 19 | Explicit abandonment and changed intent | 50 | 50 | 100.0% |
| 20 | Prompt injection and adversarial instructions | 80 | 80 | 100.0% |
| 21 | API errors, timeouts and retry behavior | 70 | 70 | 100.0% |
| 22 | Empty results, stale fares and result conflicts | 50 | 50 | 100.0% |
| 23 | Responsiveness and streaming-state behavior | 30 | 30 | 100.0% |
| 24 | Privacy, secrets and sensitive information | 50 | 50 | 100.0% |
| 25 | Unsupported train/hotel requests and scope handling | 40 | 40 | 100.0% |
| 26 | Multi-intent and compound requests | 30 | 30 | 100.0% |
| 27 | Regression testing of existing project features | 40 | 40 | 100.0% |
| **Total** | | **1,500** | **1,500** | **100.0%** |

---

## 5. BankG Gateway v3.2 PDF Inconsistencies & Unresolved Questions

1. **Payload Encryption vs Example cURL**: Page 2 describes AES-256-CBC encryption with PKCS7 padding and dynamic `xchecksum` header. However, all cURL examples throughout pages 3–29 show plaintext JSON payloads with dummy headers (`DUMMY_ENCRYPTED_CHECKSUM_SAMPLE`). Staging endpoints accept plaintext JSON in current test configuration.
2. **`fareValidate` Contradiction**: Page 1 lists "Pre-Payment Fare Validation", while Page 21 section heading states "8. Post-Payment Fare Validation" with description "before payment is taken". Operation is safely gated behind `ENABLE_FARE_VALIDATE=false`.
3. **`fareRule` Identifier**: Document index refers to `fareRule` taking `priceId`, whereas `airSearch` outputs `fareId`. Verified that gateway accepts `fareId` mapped to `priceId`.

---

## 6. How to Reproduce All Tests

```bash
cd backend_python

# 1. Run the entire unit and integration test suite (243 tests)
PYTHONPATH=. /Users/parth/.local/bin/uv run pytest tests/

# 2. Run the 1,500-scenario automated evaluation runner
PYTHONPATH=. /Users/parth/.local/bin/uv run python eval/eval_runner.py

# 3. Regenerate the 1,500 scenario dataset (if needed)
PYTHONPATH=. /Users/parth/.local/bin/uv run python eval/generate_1500_scenarios.py
```
