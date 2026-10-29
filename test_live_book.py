"""
End-to-End Test Script: airSearch -> flightReview -> flightBook -> bookingDetails
Against BankG Staging Gateway: https://bankg-stagging-api.testuat.com/travel/v1/api/travel-master/
"""

import sys
import json
import httpx

BASE_URL = "https://bankg-stagging-api.testuat.com/travel/v1/api/travel-master"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "xchecksum": "DUMMY_ENCRYPTED_CHECKSUM_SAMPLE",
}


def run_booking_flow(
    origin: str = "BOM",
    destination: str = "DEL",
    date: str = "2026-11-20",
    first_name: str = "Parth",
    last_name: str = "Thakur",
    title: str = "Mr",
    email: str = "pt541551@gmail.com",
    contact: str = "8652080057",
):
    print("=" * 70)
    print("STEP 1: Air Search (BOM -> DEL)")
    print("=" * 70)
    search_payload = {
        "cabinClass": "ECONOMY",
        "adults": 1,
        "segments": [{"origin": origin, "destination": destination, "date": date}],
        "searchModifiers": {
            "isDirectFlight": True,
            "isConnectingFlight": False,
            "pft": "REGULAR",
        },
    }
    with httpx.Client(timeout=30.0) as client:
        r_search = client.post(f"{BASE_URL}/airSearch", json=search_payload, headers=HEADERS)
        if r_search.status_code != 200 or r_search.json().get("responseCode") != "0":
            print("AirSearch failed:", r_search.text)
            return

        search_data = r_search.json().get("data", {})
        journeys = search_data.get("journeys", [])
        if not journeys:
            print("No journeys returned")
            return

        selected_fare = journeys[0]["fares"][0]
        fare_id = selected_fare["fareId"]
        flight_info = journeys[0]["flights"][0]
        airline = flight_info.get("airline", {}).get("name")
        flight_num = flight_info.get("flightNumber")
        print(f"Selected Flight: {airline} {flight_num}")
        print(f"Selected fareId: {fare_id}")
        print(f"Search Price: INR {selected_fare.get('price', {}).get('totalFare')}")

        print("\n" + "=" * 70)
        print("STEP 2: Flight Review (Fare Lock)")
        print("=" * 70)
        review_payload = {"fareId": [fare_id]}
        r_rev = client.post(f"{BASE_URL}/flightReview", json=review_payload, headers=HEADERS)
        rev_json = r_rev.json()
        if r_rev.status_code != 200 or rev_json.get("responseCode") != "0":
            print("FlightReview failed:", r_rev.text)
            return

        rev_data = rev_json.get("data", {})
        booking_id = rev_data.get("bookingId")
        pricing_info = rev_data.get("flights", [{}])[0].get("pricing", [{}])[0]
        total_fare = pricing_info.get("totalFare")
        print(f"Generated Live BookingId: {booking_id}")
        print(f"Reviewed Total Fare: INR {total_fare}")

        print("\n" + "=" * 70)
        print("STEP 3: Flight Book (Instant Ticketing)")
        print("=" * 70)
        book_payload = {
            "bookingId": booking_id,
            "paymentInfos": [{"amount": total_fare}],
            "travellerInfo": [
                {
                    "ti": title,
                    "fN": first_name,
                    "lN": last_name,
                    "pt": "ADULT",
                    "dob": "1998-05-15",
                    "pNat": "IN",
                    "pNum": "Z8792011",
                    "eD": "2030-01-01",
                    "di": "DOC123456",
                }
            ],
            "deliveryInfo": {
                "emails": [email],
                "contacts": [contact],
            },
        }

        r_book = client.post(f"{BASE_URL}/flightBook", json=book_payload, headers=HEADERS)
        book_json = r_book.json()
        print("FlightBook HTTP Status:", r_book.status_code)
        print("FlightBook Response:", json.dumps(book_json, indent=2))

        if book_json.get("responseCode") == "0":
            print("\n" + "=" * 70)
            print("STEP 4: Retrieve Booking Details")
            print("=" * 70)
            r_details = client.post(
                f"{BASE_URL}/bookingDetails",
                json={"bookingId": booking_id},
                headers=HEADERS,
            )
            print("BookingDetails HTTP Status:", r_details.status_code)
            print("BookingDetails Response:", json.dumps(r_details.json(), indent=2))


if __name__ == "__main__":
    run_booking_flow()

