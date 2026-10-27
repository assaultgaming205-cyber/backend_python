import uuid
import logging
import httpx
from typing import List, Optional, Dict, Any
from ...schemas.hotel import HotelOffer, HotelRating, HotelPrice
from ...common.dates import today_in_tz, plus_days

logger = logging.getLogger("mmt_hotel")

MMT_CITIES = [
    {"cityCode": "CTBOM", "stateCode": "STMHRA", "countryCode": "IN", "displayName": "Mumbai", "aliases": ["mumbai", "bombay", "bom", "mmct", "csmt", "panvel", "navi mumbai", "thane"]},
    {"cityCode": "CTDEL", "stateCode": "STDEL", "countryCode": "IN", "displayName": "New Delhi", "aliases": ["delhi", "new delhi", "del", "ndls", "ncr"]},
    {"cityCode": "CTBLR", "stateCode": "STKTK", "countryCode": "IN", "displayName": "Bengaluru", "aliases": ["bengaluru", "bangalore", "blr", "sbc"]},
    {"cityCode": "CTGOI", "stateCode": "STGOA", "countryCode": "IN", "displayName": "Goa", "aliases": ["goa", "north goa", "south goa", "goi", "gox", "panaji", "calangute", "baga", "candolim"]},
    {"cityCode": "CTAMD", "stateCode": "STGUJ", "countryCode": "IN", "displayName": "Ahmedabad", "aliases": ["ahmedabad", "amdavad", "amd", "adi"]},
    {"cityCode": "CTPNQ", "stateCode": "STMHRA", "countryCode": "IN", "displayName": "Pune", "aliases": ["pune", "poona", "pnq"]},
    {"cityCode": "CTCCU", "stateCode": "STWB", "countryCode": "IN", "displayName": "Kolkata", "aliases": ["kolkata", "calcutta", "ccu", "hwh"]},
    {"cityCode": "CTMAA", "stateCode": "STTN", "countryCode": "IN", "displayName": "Chennai", "aliases": ["chennai", "madras", "maa", "mas"]},
    {"cityCode": "CTJAI", "stateCode": "STRAJ", "countryCode": "IN", "displayName": "Jaipur", "aliases": ["jaipur", "pink city", "jai", "jp"]},
    {"cityCode": "CTUDR", "stateCode": "STRAJ", "countryCode": "IN", "displayName": "Udaipur", "aliases": ["udaipur", "udr", "udz"]},
    {"cityCode": "CTCOK", "stateCode": "STKER", "countryCode": "IN", "displayName": "Kochi", "aliases": ["kochi", "cochin", "cok", "ers"]},
    {"cityCode": "CTVNS", "stateCode": "STUP", "countryCode": "IN", "displayName": "Varanasi", "aliases": ["varanasi", "banaras", "kashi", "vns", "bsb"]},
    {"cityCode": "CTAGR", "stateCode": "STUP", "countryCode": "IN", "displayName": "Agra", "aliases": ["agra", "agr", "agc"]},
]


class MakeMyTripHotelProvider:
    def __init__(self, base_url: str = "https://mapi.makemytrip.com/clientbackend/cg/search-hotels"):
        self.base_url = base_url.rstrip("/")

    def resolve_destination(self, query: str) -> Dict[str, str]:
        q = (query or "").strip().lower()
        for c in MMT_CITIES:
            if c["cityCode"].lower() == q or q in c["aliases"] or c["displayName"].lower() in q:
                return c
        return {"cityCode": "CTDEL", "stateCode": "STDEL", "countryCode": "IN", "displayName": query}

    async def search_hotels(
        self,
        destination: str,
        check_in: str,
        check_out: str,
        adults: int = 2,
        rooms: int = 1,
        **kwargs
    ) -> List[HotelOffer]:
        dest = self.resolve_destination(destination)
        request_id = str(uuid.uuid4())
        visitor_id = str(uuid.uuid4())
        device_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        query_params = {
            "cityCode": dest["cityCode"],
            "requestId": request_id,
            "language": "eng",
            "region": "in",
            "currency": "INR",
            "idContext": "B2C",
            "countryCode": dest["countryCode"],
        }

        payload = {
            "deviceDetails": {
                "appVersion": "154.0.0.0",
                "deviceId": device_id,
                "bookingDevice": "DESKTOP",
                "deviceType": "DESKTOP",
                "networkType": "WiFi",
            },
            "searchCriteria": {
                "checkIn": check_in,
                "checkOut": check_out,
                "limit": 10,
                "roomStayCandidates": [{"adults": adults, "children": 0, "childAges": []}],
                "countryCode": dest["countryCode"],
                "cityCode": dest["cityCode"],
                "locationId": dest["cityCode"],
                "locationType": "city",
                "currency": "INR",
                "preAppliedFilter": False,
                "userSearchType": "city",
            },
            "userLocation": {
                "city": dest["cityCode"],
                "state": dest["stateCode"],
                "country": dest["countryCode"],
            },
            "requestDetails": {
                "visitorId": visitor_id,
                "visitNumber": 1,
                "funnelSource": "HOTELS",
                "idContext": "B2C",
                "pageContext": "LISTING",
                "channel": "B2Cweb",
                "requestId": request_id,
                "sessionId": session_id,
            },
            "imageDetails": {
                "types": ["professional"],
                "categories": [{"type": "H", "count": 4, "height": 82, "width": 123, "imageFormat": "jpg"}],
            },
            "featureFlags": {"soldOut": True, "staticData": True, "checkAvailability": True},
        }

        url = f"{self.base_url}/DESKTOP/2"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        }

        logger.info(f"Searching MMT Hotels in {dest['displayName']}: {check_in} -> {check_out}")
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(url, params=query_params, json=payload, headers=headers)
                if resp.status_code != 200:
                    logger.warning(f"MMT Hotels returned status {resp.status_code}")
                    return []
                data = resp.json()
        except Exception as e:
            logger.error(f"MMT Hotels search error: {e}")
            return []

        raw_hotels = data.get("hotels") or data.get("response", {}).get("hotels") or []
        offers: List[HotelOffer] = []

        for h in raw_hotels[:10]:
            h_id = str(h.get("id") or h.get("hotelId") or "")
            name = h.get("name") or "Hotel"
            addr = h.get("address", {}).get("line1") if isinstance(h.get("address"), dict) else h.get("address")
            rating_score = float(h.get("rating", {}).get("score") or h.get("starRating") or 4.0)

            # Price
            price_amount = 3500.0
            pricing = h.get("price") or h.get("priceDetail") or {}
            if isinstance(pricing, dict) and pricing.get("amount"):
                try:
                    price_amount = float(pricing.get("amount"))
                except Exception:
                    pass

            raw_img = None
            media = h.get("media") or []
            if isinstance(media, list):
                for m in media:
                    if isinstance(m, dict) and m.get("url") and (m.get("mediaType") in ("IMAGE", None)):
                        raw_img = m.get("url")
                        break
            if not raw_img:
                raw_img = h.get("heroImage") or h.get("image") or h.get("thumbnailUrl")
            if not raw_img and h.get("images"):
                imgs = h.get("images")
                if isinstance(imgs, list) and len(imgs) > 0:
                    raw_img = imgs[0] if isinstance(imgs[0], str) else (imgs[0].get("url") if isinstance(imgs[0], dict) else None)

            image_url = None
            if isinstance(raw_img, str) and raw_img.strip():
                image_url = raw_img.strip()
                if image_url.startswith("//"):
                    image_url = f"https:{image_url}"
                import re
                image_url = re.sub(r"downsize=\d+:\d+", "downsize=800:530", image_url)

            amenities = [a.get("name") for a in (h.get("amenities") or []) if isinstance(a, dict) and a.get("name")]

            offers.append(
                HotelOffer(
                    hotelId=f"mmt_{h_id}",
                    provider="makemytrip",
                    name=name,
                    address=addr or dest["displayName"],
                    city=dest["displayName"],
                    stars=int(h.get("starRating") or 4),
                    rating=HotelRating(score=rating_score, scale=10.0, label="Good"),
                    imageUrl=image_url,
                    price=HotelPrice(amount=price_amount, currency="INR"),
                    amenities=amenities[:6],
                )
            )

        logger.info(f"MMT Hotels found {len(offers)} hotels in {dest['displayName']}")
        return offers
