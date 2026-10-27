# TRAVEL BUDDY — BANKG FLIGHT API LIVE END-TO-END INTEGRATION & DATA-INTEGRITY AUDIT REPORT

**Date:** 2026-10-27  
**Environment:** BankG UAT / Staging Gateway (`https://bankg-stagging-api.testuat.com/travel/v1/api`)  
**Integration Spec:** *Jodetx Travel Gateway API Integration Kit — Gateway v3.2* (29 pages)  
**Target Codebase:** `/Users/parth/Downloads/travel_buddy 6/backend_python`  

---

## 1. Executive Summary

We executed live HTTP requests against the configured BankG UAT/Staging server (`https://bankg-stagging-api.testuat.com/travel/v1/api`), traced every data-integrity discrepancy across supplier payloads and application layers, implemented and verified fixes for all six audit priorities, and ran the complete 261-test suite both in the repository and from a freshly extracted copy of `backend_python_logic.zip`.

| Priority | Audit Area | Root Cause Identified | Resolution & Verification |
| :--- | :--- | :--- | :--- |
| **Priority 1** | **Fare-Price Changes (`₹4,159.50` vs `₹5,159.50`)** | Genuine supplier tax repricing during `flightReview`: SpiceJet `SG 252` returned `baseFare: 2950, tax: 1209.5, totalFare: 4159.5` in `airSearch`, and `baseFare: 2950, totalTax: 2209.5, totalFare: 5159.5` (`+₹1,000.00` tax increase) in `flightReview`. Meanwhile, Air India Express `IX 1464` (`₹3,248.50`) and `IX 1671` (`₹5,835.50`) returned identical search and reviewed prices. | `review_fare()` and `ToolExecutor` now preserve `searchPrice` and `reviewedPrice` separately, compute `priceDelta` / `taxDelta`, set `status="PRICE_CHANGED"` and `originalPriceConfirmed=False` when `abs(priceDelta) >= 0.01`, handle `PRICE_MISSING`, `CURRENCY_MISMATCH`, and `FARE_ID_MISMATCH`, and enforce via `GroundingGate` that the assistant never claims an outdated search price is confirmed. |
| **Priority 2** | **Airport & Route Identity (`BOM` vs `NMI`)** | BankG's upstream `airSearch` expands `BOM` or `NMI` to the Mumbai Metropolitan Area (`BOM` Chhatrapati Shivaji + `NMI` Navi Mumbai International Airport) and `DEL` to Delhi NCR (`DEL` + `HDO` Hindon + `DXN` Noida International). | `search_flights()` now extracts exact `departure.airportCode` / `arrival.airportCode` (never defaulting missing codes to requested IATA), preserves airport codes and names in `observedDetails`, and excludes non-matching airports by default (`BOM -> DEL` returns only `BOM -> DEL`; `NMI -> DEL` returns only `NMI -> DEL`). `LocationResolver` resolves `Mumbai`/`BOM` -> `BOM` and `Navi Mumbai`/`NMI` -> `NMI` distinctly. |
| **Priority 3** | **Checksum (`xchecksum`) Behavior** | Controlled live staging matrix (6 variations on `POST /travel-master-gateway/fareRule`): documented dummy sample, valid dynamic AES-256-CBC `xchecksum`, corrupted non-Base64 garbage, expired+tampered AES-256-CBC, empty `xchecksum: ""`, and omitted `xchecksum` header **all returned `HTTP 201` (`responseCode="0"`, `"SUCCESS"`)**. Encrypted `{"request": "<AES-256-CBC>"}` body on `airSearch` returned `HTTP 400`. | Documented as a verified UAT server finding: the staging gateway does not enforce `xchecksum` verification and accepts raw JSON payloads. `BankGFlightProvider` sends dynamic AES-256-CBC `xchecksum` when `BANKG_MERCHANT_KEY` is configured and falls back to `BANKG_CHECKSUM` otherwise. |
| **Priority 4** | **Complete Test Suite (including `tests/test_backend.py`)** | Running `pytest` without `PYTHONPATH=.` failed collection; `test_backend.py::test_fastapi_endpoints` failed because out-of-scope queries fell through to the missing-destination prompt (`is_transport_turn` did not check `not is_out_of_scope`) and `FlightLeg` lacked `flightNumber`. | Added `pythonpath = ["."]` to `pyproject.toml`, added `flightNumber` to `FlightLeg`, fixed out-of-scope handling in `agent_service.py`, ran full suite (`261 passed, 0 failed`), rebuilt `backend_python_logic.zip`, extracted to `/tmp/travel_buddy_zip_verify`, and re-ran all 261 tests from the extracted archive. |
| **Priority 5** | **Gateway Health & Endpoint Status** | `GET /travel/v1/api/health` returns `HTTP 404` (`{"responseCode":"1","description":"Cannot GET /travel/v1/api/health","error":"Not Found"}`) from the upstream NestJS router while `POST /travel-master-gateway/*` is live. | `check_health()` now reports `gatewayReachable=True`, `healthRouteMounted=False`, `healthy=False`, `status="HEALTH_ROUTE_UNMOUNTED"` on HTTP 404, and never blocks or marks flight searches as failed. |
| **Priority 6** | **End-to-End Application Behavior** | Verified full live chain `airSearch → flightReview → flightSeatMap → fareRule` through `BankGFlightProvider` → `ToolExecutor` → `ChatResults` → `build_facts` → `GroundingGate` on both `BOM -> DEL` and `NMI -> DEL`. Also discovered and fixed journey-index normalization (`_20BOM...` in `airSearch` vs `_0BOM...` in `flightReview`) in `_extract_fare_route_flight_token`. | All identifiers, airport codes, flight numbers, prices, seat maps, and fare rules stay attached to the exact flight across all layers; prohibited operations (`flightBook`, `submitAmendment`, `fareValidate`, `checkBalance`) and unauthenticated lookups make zero network calls. |

---

## 2. Priority 1 — Deep-Dive: Fare-Price Changes (`₹4,159.50` vs `₹5,159.50`)

### 2.1 Exact Supplier Payload Trace
In our recorded live `BOM -> DEL` test on SpiceJet `SG 252` (`fareId = "4-9204425964_0BOMDELSG252~5951139211020472"`):

1. **`airSearch` Response (`journeys[0].fares[0]`):**
   ```json
   {
     "fareId": "4-9204425964_0BOMDELSG252~5951139211020472",
     "cabinClass": "ECONOMY",
     "price": {
       "baseFare": 2950,
       "tax": 1209.5,
       "totalFare": 4159.5,
       "currency": "INR"
     }
   }
   ```
2. **`flightReview` Response (`data.flights[0].pricing[0]`):**
   ```json
   {
     "priceId": "R4-9204425964_0BOMDELSG252~5951152016899322",
     "fareType": "SAVER",
     "baseFare": 2950,
     "totalTax": 2209.5,
     "totalFare": 5159.5,
     "taxBreakup": {
       "carrierSurcharge": 599,
       "airlineGst": 1184,
       "managementFee": 75,
       "managementFeeTax": 13.5,
       "otherTax": 338
     }
   }
   ```
3. **Root Cause:**
   - The `fareId` (`4-9204425964_0BOMDELSG252~...`) and `priceId` (`R4-9204425964_0BOMDELSG252~...`) belong to the **exact same flight (`SG 252`), route (`BOM -> DEL`), passenger count (`1 adult`), currency (`INR`), and `baseFare` (`₹2,950.00`)**.
   - During `flightReview`, the supplier repriced `totalTax` from cached `₹1,209.50` to live `₹2,209.50` (`+₹1,000.00`), increasing `totalFare` from `₹4,159.50` to `₹5,159.50`.
   - By contrast, live `flightReview` calls on Air India Express `IX 1464` (`₹3,248.50` -> `₹3,248.50`), `IX 1671` (`₹5,835.50` -> `₹5,835.50`), and SpiceJet `SG 670` (`₹9,987.50` -> `₹9,987.50`) returned `priceDelta = 0.0`.

### 2.2 Implemented Fare Comparison & Grounding Safeguards
- **`BankGFlightProvider.review_fare()` (`app/travel/flights/bankg_flight_provider.py`):**
  - Preserves `searchPrice`, `searchBaseFare`, `searchTax`, `searchCurrency` alongside `reviewedPrice`, `baseFare`, `totalTax`, `reviewedCurrency`, `taxBreakup`, and `totalFare`.
  - Computes `priceDelta = round(reviewedPrice - searchPrice, 2)`, `taxDelta`, and `baseFareDelta`.
  - Sets `status`:
    - `"LOCKED"` and `originalPriceConfirmed = True` only when `priceVerified` is `True`, `fareIdMatched` is `True`, currencies match, and `abs(priceDelta) < 0.01`.
    - `"PRICE_CHANGED"` and `originalPriceConfirmed = False` (with a human-readable `priceDiscrepancyNotice`) when `abs(priceDelta) >= 0.01`.
    - `"PRICE_MISSING"` (`originalPriceConfirmed = False`) when `pricing` is empty or `totalFare <= 0`.
    - `"CURRENCY_MISMATCH"` (`originalPriceConfirmed = False`) when `searchCurrency != reviewedCurrency`.
    - `"FARE_ID_MISMATCH"` (`originalPriceConfirmed = False`) when returned `priceId` does not match the requested `fareId` route/flight token.
- **`ToolExecutor` (`app/agent/tool_executor.py`):**
  - Automatically matches the `FlightOffer` in `results.flights` by `fareId`/`offerId`, passes `search_offer` to `review_fare()`, and attaches `fareReview` metadata without overwriting `offer.price.amount`.
- **`build_facts()` & `GroundingGate` (`app/agent/grounding/grounded_reply.py`, `grounding_gate.py`):**
  - Populates `fareReview.status`, `fareReview.searchPrice`, `fareReview.reviewedPrice`, `fareReview.priceChanged`, `fareReview.priceDelta`, `fareReview.originalPriceConfirmed`, and `fareReview.discrepancyNotice`.
  - `GroundingGate.is_sentence_grounded()` rejects any sentence that claims the outdated `searchPrice` is `"confirmed"`, `"locked"`, or `"unchanged"` when `priceChanged=True`, or claims any fare is confirmed when `status` is `PRICE_MISSING`, `CURRENCY_MISMATCH`, or `FARE_ID_MISMATCH`.

---

## 3. Priority 2 — Deep-Dive: Airport and Route Identity (`BOM` vs `NMI`)

### 3.1 Exact Supplier Payload Trace
We inspected every journey returned by live `airSearch`:
- When querying `origin="BOM", destination="DEL"`:
  - BankG returned journeys departing from `"BOM"` (`"airportName": "Chhatrapati Shivaji", "city": "Mumbai"`) **and** journeys departing from `"NMI"` (`"airportName": "Navi mumbai international airport", "city": "Navi mumbai"`), as well as journeys arriving at `"DEL"`, `"HDO"` (`Hindon airport, Ghaziabad`), and `"DXN"` (`Noida international airport`).
  - Specifically, journey index `0` (`fareId = "4-9204425964_0NMIDELIX1464~5951139175825525"`) had `departure.airportCode = "NMI"` inside the supplier response itself.
- When querying `origin="NMI", destination="DEL"`:
  - BankG similarly returned a mix of `"NMI"` and `"BOM"` departures.

### 3.2 Implemented Route-Identity & Airport Safeguards
- **`BankGFlightProvider.search_flights()` (`app/travel/flights/bankg_flight_provider.py`):**
  - Extracts `dep_code = str(dep_info.get("airportCode") or "").strip().upper()` and `arr_code = str(arr_info.get("airportCode") or "").strip().upper()` without ever defaulting missing codes to `orig_iata`/`dest_iata`.
  - By default (`include_alternate_airports=False`), strictly filters journeys so `seg0.origin == orig_iata and seg_last.destination == dest_iata`.
    - Searching `BOM -> DEL` (or `Mumbai -> Delhi`) returns **only** `BOM -> DEL` flights and excludes `NMI`, `HDO`, and `DXN`.
    - Searching `NMI -> DEL` (or `Navi Mumbai -> Delhi`) returns **only** `NMI -> DEL` flights and excludes `BOM`.
  - When `include_alternate_airports=True` is explicitly passed, co-terminal metro airports (`BOM`/`NMI`, `DEL`/`HDO`/`DXN`) are marked `routeMatched=False, isAlternateAirport=True`, sorted after exact matches, and never marked `isCheapest=True` ahead of an exact-route match.
  - Preserves `originAirportCode`, `originAirportName`, `originCity`, `destinationAirportCode`, `destinationAirportName`, `destinationCity`, `requestedOrigin`, `requestedDestination`, `routeMatched`, and `isAlternateAirport` in `observedDetails`.
- **`LocationResolver` (`app/agent/resolvers/location_resolver.py`):**
  - Resolves `"Mumbai"` and `"BOM"` to `BOM` (`Mumbai`), and `"Navi Mumbai"` and `"NMI"` to `NMI` (`Navi Mumbai`).
  - `extract_route()` recognizes explicit 3-letter IATA airport codes (`BOM to DEL`, `NMI to DEL`, `NMI se DEL`) without false-positive matching on common English words (`"Can"` in `"Can you give me..."`).

---

## 4. Priority 3 — Live Checksum (`xchecksum`) Verification Matrix

We executed 6 controlled, read-only `POST /travel-master-gateway/fareRule` requests against `https://bankg-stagging-api.testuat.com/travel/v1/api/travel-master-gateway/fareRule` using a fresh live `fareId` (`"5-2425855328_0NMIDEL6E780~5952452078455991"`):

| Case | `xchecksum` Header Sent | HTTP Status | `responseCode` | `description` | `fareRules` Count | Server Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **A** | `"DUMMY_ENCRYPTED_CHECKSUM_SAMPLE"` | `201` | `"0"` | `"SUCCESS"` | `1` | Accepted |
| **B** | Valid dynamic AES-256-CBC `generate_bankg_xchecksum(payload, key)` | `201` | `"0"` | `"SUCCESS"` | `1` | Accepted |
| **C** | Deliberately corrupted `"!!!INVALID_NON_BASE64_GARBAGE_CHECKSUM!!!"` | `201` | `"0"` | `"SUCCESS"` | `1` | Accepted |
| **D** | Expired (`-3600s`) & tampered AES-256-CBC (`wrong_merchant_key`) | `201` | `"0"` | `"SUCCESS"` | `1` | Accepted |
| **E** | Empty string `""` | `201` | `"0"` | `"SUCCESS"` | `1` | Accepted |
| **F** | `xchecksum` header completely omitted | `201` | `"0"` | `"SUCCESS"` | `1` | Accepted |

### Key Cryptographic & Contract Findings
1. **Staging Server Does Not Enforce `xchecksum`:** Because Cases C, D, E, and F (corrupted, expired/wrong-key, empty, and omitted `xchecksum`) were all accepted with `HTTP 201` and `responseCode="0"`, the BankG UAT/Staging server currently has `xchecksum` verification disabled or bypassed in its middleware. Acceptance of a header on staging therefore **cannot** be treated as server-side proof of cryptographic key/checksum validation.
2. **Encrypted Request Envelope (`{"request": "<AES-256-CBC>"}`) Rejected on Staging:** Sending the PDF Page 2 encrypted body envelope to `POST /travel-master-gateway/airSearch` returns `HTTP 400 Bad Request` (`"cabinClass must be one of the following values: ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST..."`), proving the NestJS DTO validator inspects top-level plaintext JSON fields directly.
3. **Supplier Key Expiration Window (~15 Minutes):** Re-using a `fareId` or `priceId` after ~15 minutes returns `HTTP 201` with `responseCode="1"` and `description="Keys Passed in the request is already expired. Please pass valid keys"`.

---

## 5. Priority 5 — Gateway Health and Endpoint Status

- Calling `GET https://bankg-stagging-api.testuat.com/travel/v1/api/health` returns `HTTP 404 Not Found` in `~260–310 ms`:
  ```json
  {
    "responseCode": "1",
    "description": "Cannot GET /travel/v1/api/health",
    "error": "Not Found",
    "data": [],
    "timestamp": "27-10-2026 01:19:16"
  }
  ```
- This response comes from BankG's NestJS router, proving TLS/HTTPS connectivity to the gateway host is up while the `/health` route documented in Section 1 of the PDF is not mounted on staging.
- `BankGFlightProvider.check_health()` returns `gatewayReachable=True`, `healthRouteMounted=False`, `healthy=False`, `status="HEALTH_ROUTE_UNMOUNTED"`, and never blocks or marks `airSearch` as failed.

---

## 6. Priority 6 — Live End-to-End Application Verification (`airSearch → flightReview → flightSeatMap → fareRule`)

We executed the full application pipeline (`BankGFlightProvider` → `ToolExecutor` → `ChatResults` → `build_facts` → `GroundingGate`) against the live BankG staging server on `2026-10-28` across both `BOM -> DEL` and `NMI -> DEL` (evidence saved in `priority6_live_e2e_verification.json`):

### Chain 1: `BOM -> DEL` (`2026-10-28`, Economy, 1 Adult)
1. **`search_flights` (`BOM -> DEL`):**
   - Returned `830` verified offers; `unique_origins_returned = ["BOM"]`, `unique_destinations_returned = ["DEL"]` (zero `NMI`, `HDO`, or `DXN` offers leaked into `BOM -> DEL`).
   - Selected direct flight `IX 1671` (`fareId = "22-2547457232_20BOMDELIX1671~5953945328750020"`, `origin = "BOM"`, `originAirportName = "Chhatrapati Shivaji"`, `destination = "DEL"`, `searchPrice = ₹5,835.50`).
2. **`review_fare` (`IX 1671`):**
   - Returned `bookingId = "TJS100803262836"`, `reqId = "2547457232"`, `priceId = "R22-2547457232_0BOMDELIX1671~3097451626490307"`.
   - `fareIdMatched = True`, `searchPrice = 5835.5`, `reviewedPrice = 5835.5`, `priceDelta = 0.0`, `status = "LOCKED"`, `originalPriceConfirmed = True`, `mealOptionsCount = 1`, `baggageOptionsCount = 4`.
3. **`get_seat_map` (`bookingId = "TJS100803262836"`):**
   - Returned `1` segment seat map (`30x7` layout, `176` seats).
4. **`get_fare_rules` (`priceId = "R22-2547457232_0BOMDELIX1671~3097451626490307"`):**
   - Returned `1` structured fare rule containing `NO_SHOW`, `DATECHANGE`, `CANCELLATION`, and `SEAT_CHARGEABLE`.
5. **Also Reviewed Live SpiceJet `SG 670` (`BOM -> DEL`):**
   - `fareId = "4-2547457232_0BOMDELSG670~5953944579714081"` -> `bookingId = "TJS107403262837"`, `priceId = "R4-2547457232_0BOMDELSG670~3097456029402501"`, `searchPrice = 9987.5`, `reviewedPrice = 9987.5`, `status = "LOCKED"`, `originalPriceConfirmed = True`.

### Chain 2: `NMI -> DEL` (`2026-10-28`, Economy, 1 Adult)
1. **`search_flights` (`NMI -> DEL`):**
   - Returned `110` verified offers; `unique_origins_returned = ["NMI"]`, `unique_destinations_returned = ["DEL"]` (zero `BOM` offers leaked into `NMI -> DEL`).
   - Selected direct flight `IX 1464` (`fareId = "22-2376841380_0NMIDELIX1464~5953972035849145"`, `origin = "NMI"`, `originAirportName = "Navi mumbai international airport"`, `destination = "DEL"`, `searchPrice = ₹3,248.50`).
2. **`review_fare` (`IX 1464`):**
   - Returned `bookingId = "TJS103403262839"`, `reqId = "2376841380"`, `priceId = "R22-2376841380_0NMIDELIX1464~5953992406130589"`.
   - `fareIdMatched = True`, `searchPrice = 3248.5`, `reviewedPrice = 3248.5`, `priceDelta = 0.0`, `status = "LOCKED"`, `originalPriceConfirmed = True`.
3. **`get_seat_map` (`bookingId = "TJS103403262839"`):**
   - Returned `232` seats on `NMI -> DEL`.
4. **`get_fare_rules` (`priceId = "R22-2376841380_0NMIDELIX1464~5953992406130589"`):**
   - Returned `1` structured fare rule.

---

## 7. Complete Verification Matrix by Category

### Category 1 — Successful Live Staging Requests (`https://bankg-stagging-api.testuat.com/travel/v1/api`)
| Endpoint | HTTP Method & Path | HTTP Status | Provider `responseCode` | Verified Behavior |
| :--- | :--- | :---: | :---: | :--- |
| **`airSearch` (`BOM -> DEL`)** | `POST /travel-master-gateway/airSearch` | `201` | `"0"` (`SUCCESS`) | Returned live journeys; filtered strictly to `BOM -> DEL` (`830` offers on `2026-10-28`, `100` exact journeys on `2026-11-10`). |
| **`airSearch` (`NMI -> DEL`)** | `POST /travel-master-gateway/airSearch` | `201` | `"0"` (`SUCCESS`) | Returned live journeys; filtered strictly to `NMI -> DEL` (`110` offers on `2026-10-28`). |
| **`airSearch` (`BOM -> JAI`, `FIRST`)** | `POST /travel-master-gateway/airSearch` | `201` | `"1"` (`No flight search results found...`) | Valid empty result cleanly distinguished from provider failure (`returns []`). |
| **`flightReview` (`SG 252`, `IX 1464`, `IX 1671`, `SG 670`)** | `POST /travel-master-gateway/flightReview` | `201` | `"0"` (`SUCCESS`) | Locks itinerary (`TJS...`), returns `priceId`, nested `flights[].pricing`, `cancellationRules`, `mealOptions`, `baggageOptions`, and compares `searchPrice` vs `reviewedPrice`. |
| **`flightSeatMap` (`TJS106403262265`, `TJS100803262836`, `TJS103403262839`)** | `POST /travel-master-gateway/flightSeatMap` | `201` | `"0"` (`SUCCESS`) | Returns aircraft layout (`34x7`, `30x7`, `32x7`) and per-seat coordinates, availability, characteristics, and prices. |
| **`fareRule` (from `fareId` & `priceId`, plus 6-case `xchecksum` matrix)** | `POST /travel-master-gateway/fareRule` | `201` | `"0"` (`SUCCESS`) | Returns structured `CANCELLATION`, `DATECHANGE`, `NO_SHOW`, and `SEAT_CHARGEABLE` rules; expires after ~15 minutes. |

### Category 2 — Mocked and Recorded-Response Regression Tests (`tests/test_bankg_recorded_regression.py`)
- 12 offline tests replaying sanitized live staging responses from `tests/fixtures/bankg_live_recorded_sanitized.json` plus edge-case simulations:
  1. `test_01_recorded_health_404_is_reported_truthfully`
  2. `test_02_recorded_dependency_chain_airSearch_to_review_seatMap_fareRule`
  3. `test_03_recorded_zero_result_response_distinguished_from_provider_error`
  4. `test_04_synthetic_fallback_fare_ids_rejected_before_network`
  5. `test_05_prohibited_and_unauthorized_operations_make_zero_network_calls`
  6. `test_06_fastapi_flights_search_endpoint_parses_recorded_response`
  7. `test_07_fare_comparison_identical_prices`
  8. `test_08_fare_comparison_missing_prices`
  9. `test_09_fare_comparison_different_currencies_and_mismatched_fare_ids`
  10. `test_10_grounding_gate_and_facts_reflect_reviewed_price_and_reject_stale_confirmation`
  11. `test_11_airport_and_route_identity_bom_vs_nmi_and_route_mismatch`
  12. `test_12_location_resolver_city_vs_airport_selection`

### Category 3 — Local Unit and Integration Test Suite (Priority 4)
- **Full Repository Suite Command:**
  ```bash
  cd "/Users/parth/Downloads/travel_buddy 6/backend_python" && .venv/bin/pytest -v
  ```
  **Result:** `261 passed, 1 warning, 34 subtests passed in 74.21s` (`0 failed`, `0 skipped`, `0 ignored`).
- **Fresh Extracted Release ZIP Suite Command:**
  Extracted `/Users/parth/Downloads/travel_buddy 6/backend_python_logic.zip` into a clean `/tmp/travel_buddy_zip_verify` directory and executed the full pytest suite from that extracted copy (see Section 8 below).

### Category 4 — Tests / Endpoints That Could Not Execute Against Live Staging
| Endpoint | Status | Exact Technical Reason |
| :--- | :--- | :--- |
| **`GET /travel/v1/api/health`** | Reachable host; route returns `HTTP 404` | Upstream NestJS router returns `{"responseCode":"1","description":"Cannot GET /travel/v1/api/health","error":"Not Found"}`. Handled as `HEALTH_ROUTE_UNMOUNTED`. |
| **`POST /travel-master-gateway/bookingDetails`** | Blocked locally (0 HTTP calls) | Requires a genuine confirmed `bookingId` owned by an authenticated user (`trusted_auth`). Review-stage `bookingId`s have no issued PNR/ticket, and querying unowned bookings is prohibited. |
| **`POST /travel-master-gateway/amendmentDetails`** | Blocked locally (0 HTTP calls) | Requires a genuine existing `amendmentId` owned by an authenticated user (`trusted_auth`). |

### Category 5 — Operations Intentionally Disabled by Safety Gates
| Endpoint | Method / Gate | Verification |
| :--- | :--- | :--- |
| **`POST /travel-master-gateway/flightBook`** | `BankGFlightProvider.book_flight` / `ToolExecutor._gate` / `AgentService` regex | Raises `OperationDisabledError`; blocked in `PROHIBITED_TOOLS` with 0 network calls. |
| **`POST /travel-master-gateway/submitAmendment`** | `BankGFlightProvider.submit_amendment` / `ToolExecutor._gate` / `AgentService` regex | Raises `OperationDisabledError`; blocked in `PROHIBITED_TOOLS` with 0 network calls. |
| **`POST /travel-master-gateway/fareValidate`** | `BankGFlightProvider.validate_fare` (`ENABLE_FARE_VALIDATE=False`) | Raises `OperationDisabledError`; blocked in `PROHIBITED_TOOLS` with 0 network calls. |
| **`POST /travel-master-gateway/checkBalance`** | `BankGFlightProvider.check_balance` | Raises `OperationDisabledError`; blocked in `PROHIBITED_TOOLS` with 0 network calls. |

### Category 6 — Remaining Provider Dependencies & Clarifications Needed from BankG / Jodetx
1. **`xchecksum` & Payload Encryption Enforcement in Production:** Staging currently accepts any `xchecksum` (including garbage, expired, or omitted headers) and rejects `{"request": "<AES-256-CBC>"}` encrypted bodies on `airSearch` with `HTTP 400`. BankG must confirm whether production uses the same plaintext JSON body + `xchecksum` header or enables middleware decryption, and must provide the merchant secret key before production cutover.
2. **`GET /travel/v1/api/health` Route Mounting:** Confirm whether `/health` is mounted at a different path or intentionally omitted in UAT/production.
3. **Co-Terminal Airport Expansion Flag:** Confirm whether `airSearch` supports a `searchModifiers` flag to disable automatic metro-area expansion (`BOM` -> `BOM`+`NMI`, `DEL` -> `DEL`+`HDO`+`DXN`) upstream, in addition to Travel Buddy's strict client-side route filter.
4. **Authorized Sandbox Test Fixtures for `bookingDetails` & `amendmentDetails`:** Provide a read-only sandbox `bookingId` and `amendmentId` owned by the test merchant account so post-booking lookup parsing can be verified against live staging without calling `flightBook` or `submitAmendment`.
