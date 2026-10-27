import re
from typing import Optional, List, Tuple
from ..resolvers.location_resolver import LocationResolver

NON_TRAVEL_KEYWORDS = [
    r"\b(iphone|ipad|macbook|apple\s*watch|airpods|imac|applecare|ios)\b",
    r"\b(android|samsung|galaxy|oneplus|xiaomi|redmi|realme|vivo|oppo|pixel|motorola|moto|iqoo|poco)\b",
    r"\b(laptop|desktop|computer|smartwatch|earbuds|headphones|television|camera|dslr)\b",
    r"\b(processor|snapdragon|bionic|chipset|ram|rom|megapixels?|charger|display|amoled|oled|curved\s*screen)\b",
    r"\b(\d+\s*gb|\d+\s*tb|256gb|512gb|128gb|64gb|battery\s*life|refresh\s*rate)\b",
    r"\b(warranty|guarantee|specifications|specs|unboxing)\b",
    r"\b(flipkart|flipkat|amazon|myntra|meesho|ajio|nykaa|blinkit|zepto|instamart|swiggy|zomato)\b",
    r"\b(bbd|big\s*billion\s*days?|great\s*indian\s*festival|buy\s*back\s*discount)\b",
    r"\b(discount\s*coupon|coupon\s*code|promo\s*code|voucher|cashback|sale\s*price|offers?\s*on)\b",
    r"\b(t-?shirt|shoes?|clothes?|jeans|fashion|dress|saree|makeup|cosmetics|grocery|groceries)\b",
    r"\b(buy\s+(?:a\s+)?(?:phone|mobile|laptop|tv|watch|car|bike|shirt|shoe))\b",
    r"\b(python|javascript|typescript|java|c\+\+|golang|rust|php|html|css|sql|react|angular|vue|node\.?js)\b",
    r"\b(write\s+(?:a\s+)?(?:code|script|program|function|class)|debug|compiler|syntax\s*error|algorithm)\b",
    r"\b(git\s+repo|github|docker|kubernetes|database\s+query)\b",
    r"\b(write\s+(?:a\s+)?(?:essay|poem|poetry|story|joke|article|paragraph))\b",
    r"\b(chutkula|kahani|shayari|kavita|story\s*telling)\b",
    r"\b(solve\s+(?:this\s+)?(?:math|equation|problem|integral|derivative)|homework|physics|chemistry|biology)\b",
    r"\b(who\s+is\s+(?!the\s+(?:station\s+master|pilot|driver|conductor))[\w\s]{2,30})\b",
    r"\b(narendra\s*modi|donald\s*trump|bjp|congress|election|vote|politics|minister|president)\b",
    r"\b(stock\s*market|share\s*market|nifty|sensex|share\s*price|crypto|cryptocurrency|bitcoin|btc|ethereum|eth|trading)\b",
    r"\b(cricket\s*score|live\s*score|ipl\s*score|world\s*cup|football\s*score|messi|ronaldo)\b",
    r"\b(recipe|cook(?:ing)?|how\s+to\s+make|banane\s+ki\s+vidhi|ingredients)\b",
    r"\b(medicine|doctor|symptoms?|disease|illness|bimar|tablet|painkiller|headache|fever)\b",
]

TRAVEL_KEYWORDS = [
    r"\b(weather|season|local outing|day out|places? to visit)\b|मौसम",
    r"\b(flight|flights|plane|planes|airplane|airplanes|airline|airlines|fly|flying|airfare|airport|airports)\b",
    r"\b(indigo|air\s*india|spicejet|vistara|akasa|airasia|boeing|airbus|terminal|layover|non-?stop)\b",
    r"\b(cabin|economy|premium\s*economy|business\s*class|baggage|luggage)\b",
    r"फ्लाइट|विमान|हवाई\s*जहाज|एयरलाइन|एयरपोर्ट",
    r"\b(train|trains|rail|railway|railways|irctc|station|railgadi|gadi|gaadi|coach|berth)\b",
    r"\b(rajdhani|shatabdi|vande\s*bharat|duronto|garib\s*rath|tejas|intercity|superfast|express)\b",
    r"\b(sleeper|3ac|3a|third\s*ac|2ac|2a|second\s*ac|1ac|1a|first\s*ac|chair\s*car|tatkal|pnr|waiting\s*list|rac)\b",
    r"ट्रेन|रेल|रेलवे|आईआरसीटीसी|स्टेशन|सवारी\s*गाड़ी|राजधानी|शताब्दी|वंदे\s*भारत",
    r"\b(hotel|hotels|resort|resorts|stay|stays|room|rooms|hostel|homestay|villa|check-?in|check-?out)\b",
    r"होटल|रिसॉर्ट|कमरा|ठहरना",
    r"\b(travel|travelling|traveling|trip|tour|tourism|vacation|holiday|holidays|journey|safar|yatra)\b",
    r"\b(ghumne|ghoomne|ghuma|ghumo|ghumna|chalein|chalo|visit|sightseeing|itinerary)\b",
    r"\b(ticket|tickets|booking|book|reserve|reservation|fare|fares|kiraya|ticket\s*price)\b",
    r"\b(tum\s+suggest\s+karo|tu\s+suggest\s+kar|aap\s+suggest\s+karo|you\s+suggest|suggest\s+karo)\b",
    r"\b(kab\s+jana\s+chahiye|kab\s+jau|kab\s+jaye|sasta\s+kab|cheapest\s+date|flexible\s+date)\b",
    r"यात्रा|सफर|टिकट|बुकिंग|किराया|घूमने",
    r"\b(?:from\s+[\w\s]+?\s+to\s+[\w\s]+)\b",
    r"\b(?:[\w\s]+?\s+se\s+[\w\s]+?\s+(?:jana|jaana|chale|flight|train|safar))\b",
]

GREETINGS_AND_BOT_INTRO = [
    r"\b(?:hi|hii|hello|hey|heyy|namaste|namaskar|kem\s*cho|vanakkam|sat\s*sri\s*akaal|pranam|adaab|salaam|jode|ram\s*ram|radhe\s*radhe|good\s*(?:morning|afternoon|evening))\b",
    r"\b(?:meera|kabir|companion|travel\s*buddy|buddy)\b",
    r"\b(?:kaise\s+ho|kya\s+haal\s+hai|how\s+are\s+you|what'?s\s+up)\b",
    r"\b(?:who\s+are\s+you|what\s+can\s+you\s+do|tum\s+kaun\s+ho|aap\s+kaun\s+hain|help\s+me|madad\s+karo)\b",
]

ACTIVE_CONVERSATION_ANSWERS = [
    r"^(?:ha|haan|yes|yeah|yup|ok|okay|sure|theek\s+hai|nahi|no|nope|cancel|reset)[.!?]*$",
    r"\b(?:today|tomorrow|kal|parso|aaj|next\s+week|this\s+weekend)\b",
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)\b",
    r"\b(?:morning|subah|evening|shaam|night|raat|cheapest|sasta|fastest|direct)\b",
]


class ScopeCheckResult:
    def __init__(self, is_out_of_scope: bool = False, is_mixed: bool = False, refusal_message: Optional[str] = None, reason: Optional[str] = None):
        self.isOutOfScope = is_out_of_scope
        self.isMixed = is_mixed
        self.refusalMessage = refusal_message
        self.reason = reason


class TravelDomainGuard:
    @staticmethod
    def evaluate(
        text: str,
        user_language: str = "hinglish",
        character: str = "meera",
        has_active_session_with_location: bool = False
    ) -> ScopeCheckResult:
        raw = text.strip()
        if not raw:
            return ScopeCheckResult()

        non_travel_matches = []
        for pattern in NON_TRAVEL_KEYWORDS:
            m = re.search(pattern, raw, re.IGNORECASE)
            if m:
                non_travel_matches.append(m.group(0))

        has_non_travel = len(non_travel_matches) > 0
        has_travel = any(re.search(p, raw, re.IGNORECASE) for p in TRAVEL_KEYWORDS)
        is_greeting = any(re.search(p, raw, re.IGNORECASE) for p in GREETINGS_AND_BOT_INTRO)
        is_active_answer = has_active_session_with_location and any(re.search(p, raw, re.IGNORECASE) for p in ACTIVE_CONVERSATION_ANSWERS)

        orig, dest, reg = LocationResolver.extract_route(raw)
        has_city_or_route = bool(orig or dest or reg or LocationResolver.resolve_city(raw))

        is_travel_related = has_travel or is_greeting or is_active_answer or has_city_or_route

        def get_refusal(user_lang: str, char: str) -> str:
            is_male = (char == "kabir")
            verb = "sakta" if is_male else "sakti"
            verb_hi = "सकता" if is_male else "सकती"
            if user_lang == "english":
                return "I am a travel agent companion, so I can only help with travel and bookings. You can ask me anything about flights, trains, hotels, or trip planning!"
            elif user_lang == "hindi":
                return f"मैं एक ट्रैवल एजेंट / बडी हूँ, मैं यह सब नहीं बता {verb_hi}। आप मुझसे यात्रा, फ्लाइट, ट्रेन या होटल बुकिंग के बारे में कुछ भी पूछ सकते हैं!"
            else:
                return f"Main ek travel agent / travel buddy hoon, main ye sab nahi bata {verb}। Aap travel ke bare me kuch bhi pooch sakte hain jaise flights, trains ya hotel bookings!"

        # CASE A: Explicit non-travel inquiry
        if has_non_travel and not has_travel and not has_city_or_route:
            return ScopeCheckResult(
                is_out_of_scope=True,
                is_mixed=False,
                reason=f"Detected non-travel inquiry: {', '.join(non_travel_matches)}",
                refusal_message=get_refusal(user_language, character),
            )

        # CASE B: Mixed non-travel + travel inquiry
        if has_non_travel and (has_travel or has_city_or_route):
            return ScopeCheckResult(
                is_out_of_scope=False,
                is_mixed=True,
                reason=f"Mixed request with non-travel ({', '.join(non_travel_matches)}) and travel content",
            )

        # CASE C: Pure general prompt with NO travel intent, greetings, city, or active session
        if not is_travel_related and not has_active_session_with_location:
            return ScopeCheckResult(
                is_out_of_scope=True,
                is_mixed=False,
                reason="Input does not contain travel, flight, train, or hotel queries.",
                refusal_message=get_refusal(user_language, character),
            )

        return ScopeCheckResult(is_out_of_scope=False, is_mixed=False)
