import logging
import httpx
from typing import List, Optional, Dict, Any
from ...schemas.train import TrainOption, Station, TrainClassFare, RAIL_CLASS_DISPLAY_NAMES
from ...common.dates import today_in_tz

logger = logging.getLogger("mmt_rail")


class MakeMyTripRailProvider:
    def __init__(self, base_url: str = "https://railways.makemytrip.com", timeout_sec: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    async def search_trains(self, origin: str, destination: str, date: str) -> List[TrainOption]:
        orig_code = origin.strip().upper()
        dest_code = destination.strip().upper()
        doj = (date or today_in_tz()).replace("-", "")[:8]

        url = f"{self.base_url}/api/tbsWithAvailabilityAndRecommendation/{orig_code}/{dest_code}/{doj}?supportLadiesQuota=true"
        headers = {
            "Accept": "*/*",
            "User-Agent": "curl/8.7.1"
        }

        logger.info(f"Querying MMT Rail: {orig_code}->{dest_code} on {doj}")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code != 200:
                    logger.warning(f"MMT Rail returned status {resp.status_code}")
                    return []
                data = resp.json()
        except Exception as e:
            logger.error(f"MMT Rail query error: {e}")
            return []

        raw_trains = data.get("trainBtwnStnsList") or []
        origin_stn = Station(code=data.get("frmStationCode", orig_code), name=data.get("frmStationName", orig_code))
        dest_stn = Station(code=data.get("toStationCode", dest_code), name=data.get("toStationName", dest_code))

        trains: List[TrainOption] = []
        for t in raw_trains:
            if not t.get("trainNumber") or not t.get("trainName"):
                continue

            run_days = []
            if t.get("runningMon") == "Y": run_days.append("Mon")
            if t.get("runningTue") == "Y": run_days.append("Tue")
            if t.get("runningWed") == "Y": run_days.append("Wed")
            if t.get("runningThu") == "Y": run_days.append("Thu")
            if t.get("runningFri") == "Y": run_days.append("Fri")
            if t.get("runningSat") == "Y": run_days.append("Sat")
            if t.get("runningSun") == "Y": run_days.append("Sun")

            class_fares: List[TrainClassFare] = []
            classes = list(t.get("avlClasses") or [])

            for av in t.get("tbsAvailability") or []:
                cc = av.get("classType") or av.get("className")
                if not cc:
                    continue
                if cc not in classes:
                    classes.append(cc)
                fare = av.get("totalFare")
                if fare and isinstance(fare, (int, float)) and fare > 0:
                    status = av.get("prettyPrintingAvailablityStatus") or av.get("availablityStatus")
                    class_fares.append(
                        TrainClassFare(
                            classCode=cc,
                            className=RAIL_CLASS_DISPLAY_NAMES.get(cc, cc),
                            fare=float(fare),
                            status=status,
                            currency="INR",
                        )
                    )

            trains.append(
                TrainOption(
                    provider="makemytrip",
                    trainNumber=str(t.get("trainNumber")),
                    trainName=t.get("trainName"),
                    origin=Station(code=t.get("frmStnCode") or origin_stn.code, name=t.get("frmStnName") or origin_stn.name),
                    destination=Station(code=t.get("toStnCode") or dest_stn.code, name=t.get("toStnName") or dest_stn.name),
                    departureTime=t.get("departureTime"),
                    arrivalTime=t.get("arrivalTime"),
                    durationMinutes=t.get("duration"),
                    distanceKm=t.get("distance"),
                    runDays=run_days or None,
                    classes=classes or None,
                    classFares=class_fares or None,
                    departureDate=date,
                    journeyKind="primary",
                )
            )

        logger.info(f"MMT Rail found {len(trains)} trains for {orig_code}->{dest_code}")
        return trains
