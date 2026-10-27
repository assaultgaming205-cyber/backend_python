import asyncio
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import quote
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

logger = logging.getLogger("agoda_session")


class AgodaSessionService:
    def __init__(self):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._lock = asyncio.Lock()

    async def _init_browser(self):
        if self._browser and self._context:
            return
        async with self._lock:
            if self._browser and self._context:
                return
            logger.info("Launching Playwright Chromium for Agoda sessions...")
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu',
                ]
            )
            self._context = await self._browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
                viewport={'width': 1280, 'height': 800},
                locale='en-GB',
                timezone_id='Asia/Kolkata',
            )
            logger.info("Agoda browser session initialized.")

    async def close(self):
        try:
            if self._context:
                await self._context.close()
                self._context = None
            if self._browser:
                await self._browser.close()
                self._browser = None
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
            logger.info("Agoda browser session closed.")
        except Exception as e:
            logger.warning(f"Error closing Agoda browser: {e}")

    async def search_flights(self, origin: str, destination: str, departure_date: str, adults: int = 1, cabin_class: str = "economy") -> List[Dict[str, Any]]:
        for attempt in range(1, 3):
            try:
                await self._init_browser()
                page = await self._context.new_page()
                flight_items: List[Dict[str, Any]] = []

                async def handle_response(res):
                    nonlocal flight_items
                    if "/api/flights-bff/search/v1/flights" in res.url:
                        try:
                            data = await res.json()
                            items = data.get("data", {}).get("response", {}).get("content", {}).get("items", [])
                            if isinstance(items, list) and items:
                                flight_items = items
                                logger.debug(f"Intercepted {len(items)} Agoda flights from BFF API")
                        except Exception:
                            pass

                page.on("response", handle_response)

                cabin_type = (
                    "First"
                    if cabin_class == "first"
                    else "Business"
                    if cabin_class == "business"
                    else "PremiumEconomy"
                    if cabin_class == "premium_economy"
                    else "Economy"
                )
                search_url = (
                    f"https://www.agoda.com/en-gb/flights/results?"
                    f"departureFrom={quote(origin)}&departureFromType=1&"
                    f"arrivalTo={quote(destination)}&arrivalToType=0&"
                    f"departDate={quote(departure_date)}&"
                    f"adults={adults}&cabinType={cabin_type}&sort=8"
                )

                logger.info(f"Navigating Agoda: {origin} -> {destination} on {departure_date}")
                try:
                    await page.goto(search_url, wait_until="domcontentloaded", timeout=25000)
                except Exception as e:
                    logger.warning(f"Navigation timeout or error on page.goto: {e}")

                max_wait = 12.0
                step = 0.4
                elapsed = 0.0
                while elapsed < max_wait:
                    if flight_items:
                        break
                    await asyncio.sleep(step)
                    elapsed += step

                await page.close()
                logger.info(f"Agoda search for {origin}->{destination}: found {len(flight_items)} offers")
                return flight_items
            except Exception as e:
                logger.warning(f"Agoda search attempt {attempt} failed: {e}")
                await self.close()
                if attempt == 2:
                    return []
                await asyncio.sleep(1)
        return []
