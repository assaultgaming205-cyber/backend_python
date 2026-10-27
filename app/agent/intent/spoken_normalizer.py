import re
from typing import Dict, List
from ...common.dates import today_in_tz

DEVANAGARI_DIGITS = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

INDIC_NUMBERS: Dict[str, str] = {
    'पहला': '1st', 'दूसरा': '2nd', 'तीसरा': '3rd', 'चौथा': '4th',
    'एक': '1', 'दो': '2', 'तीन': '3', 'चार': '4', 'पांच': '5', 'पाँच': '5',
    'छह': '6', 'सात': '7', 'आठ': '8', 'नौ': '9', 'दस': '10',
    'ग्यारह': '11', 'बारह': '12', 'तेरह': '13', 'चौदह': '14', 'पंद्रह': '15',
    'सोलह': '16', 'सत्रह': '17', 'अठारह': '18', 'उन्नीस': '19', 'बीस': '20',
    'इक्कीस': '21', 'बाईस': '22', 'तेईस': '23', 'चौबीस': '24', 'पच्चीस': '25',
    'छब्बीस': '26', 'सत्ताईस': '27', 'अट्ठाईस': '28', 'उनतीस': '29', 'तीस': '30',
    'इकतीस': '31'
}

ENGLISH_ORDINALS: Dict[str, str] = {
    'twenty first': '21st', 'twenty-first': '21st',
    'twenty second': '22nd', 'twenty-second': '22nd',
    'twenty third': '23rd', 'twenty-third': '23rd',
    'twenty fourth': '24th', 'twenty-fourth': '24th',
    'twenty fifth': '25th', 'twenty-fifth': '25th',
    'twenty sixth': '26th', 'twenty-sixth': '26th',
    'twenty seventh': '27th', 'twenty-seventh': '27th',
    'twenty eighth': '28th', 'twenty-eighth': '28th',
    'twenty ninth': '29th', 'twenty-ninth': '29th',
    'thirty first': '31st', 'thirty-first': '31st',
    'thirtieth': '30th',
    'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th', 'fifth': '5th',
    'sixth': '6th', 'seventh': '7th', 'eighth': '8th', 'ninth': '9th', 'tenth': '10th',
    'eleventh': '11th', 'twelfth': '12th', 'thirteenth': '13th', 'fourteenth': '14th',
    'fifteenth': '15th', 'sixteenth': '16th', 'seventeenth': '17th', 'eighteenth': '18th',
    'nineteenth': '19th', 'twentieth': '20th',
}

ENGLISH_CARDINALS: Dict[str, str] = {
    'twenty one': '21', 'twenty-one': '21',
    'twenty two': '22', 'twenty-two': '22',
    'twenty three': '23', 'twenty-three': '23',
    'twenty four': '24', 'twenty-four': '24',
    'twenty five': '25', 'twenty-five': '25',
    'twenty six': '26', 'twenty-six': '26',
    'twenty seven': '27', 'twenty-seven': '27',
    'twenty eight': '28', 'twenty-eight': '28',
    'twenty nine': '29', 'twenty-nine': '29',
    'thirty one': '31', 'thirty-one': '31',
    'thirty': '30', 'twenty': '20',
    'nineteen': '19', 'eighteen': '18', 'seventeen': '17', 'sixteen': '16',
    'fifteen': '15', 'fourteen': '14', 'thirteen': '13', 'twelve': '12',
    'eleven': '11', 'ten': '10', 'nine': '9', 'eight': '8', 'seven': '7',
    'six': '6', 'five': '5', 'four': '4', 'three': '3', 'two': '2', 'one': '1',
}

HINGLISH_NUMBERS: Dict[str, str] = {
    'sattaees': '27', 'sataees': '27', 'sattais': '27', 'satais': '27',
    'atthaees': '28', 'athaees': '28', 'atthais': '28', 'athais': '28',
    'chhabees': '26', 'chhabis': '26',
    'pachees': '25', 'pachis': '25',
    'unantis': '29', 'unattis': '29', 'unatis': '29', 'tees': '30',
    'iktees': '31', 'ikatis': '31', 'ikattis': '31',
    'ikkees': '21', 'ikkis': '21',
    'baees': '22', 'bais': '22', 'teees': '23', 'teis': '23',
    'chaubees': '24', 'chaubis': '24', 'bees': '20', 'unnees': '19',
    'unnis': '19', 'athaarah': '18', 'atharah': '18', 'satrah': '17',
    'solah': '16', 'pandrah': '15', 'chaudah': '14', 'terah': '13',
    'baarah': '12', 'barah': '12', 'gyaarah': '11', 'gyarah': '11',
    'das': '10', 'nau': '9', 'aath': '8', 'saat': '7', 'chhe': '6',
    'che': '6', 'paanch': '5', 'panch': '5', 'chaar': '4', 'char': '4',
    'teen': '3', 'do': '2', 'ek': '1'
}

MONTHS_DEVA: Dict[str, str] = {
    'जनवरी': 'January', 'फरवरी': 'February', 'मार्च': 'March', 'अप्रैल': 'April',
    'मई': 'May', 'जून': 'June', 'जुलाई': 'July', 'अगस्त': 'August',
    'सितंबर': 'September', 'अक्टूबर': 'October', 'नवंबर': 'November', 'दिसंबर': 'December'
}

WEEKDAYS = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']
MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


def normalize_spoken_travel_text(input_text: str, time_zone: str = "Asia/Kolkata") -> str:
    if not input_text or not isinstance(input_text, str):
        return input_text or ""

    text = input_text
    for dev, ascii_digit in DEVANAGARI_DIGITS.items():
        text = text.replace(dev, ascii_digit)

    # Marathi travel suffixes & relative date words
    text = re.sub(r"मुंबईहून", "मुंबई se", text)
    text = re.sub(r"पुण्याहून", "पुणे se", text)
    text = re.sub(r"दिल्लीहून", "दिल्ली se", text)
    text = re.sub(r"नागपूरहून", "नागपूर se", text)
    text = re.sub(r"गोव्याहून", "गोवा se", text)
    text = re.sub(r"बंगळुरूहून|बेंगलुरुहून", "बेंगलुरु se", text)
    text = re.sub(r"([^\s,]+?)(?:हून|वरून)(?=\s|$)", r"\1 se", text)
    text = re.sub(r"दिल्लीला", "दिल्ली ko", text)
    text = re.sub(r"मुंबईला", "मुंबई ko", text)
    text = re.sub(r"पुण्याला", "पुणे ko", text)
    text = re.sub(r"गोव्याला", "गोवा ko", text)
    text = re.sub(r"बंगळुरूला|बेंगलुरुला|बंगळुरू", "बेंगलुरु ko", text)
    text = re.sub(r"चेन्नईला", "चेन्नई ko", text)
    text = re.sub(r"कोलकात्याला", "कोलकाता ko", text)
    text = re.sub(r"हैदराबादला", "हैदराबाद ko", text)
    text = re.sub(r"जयपूरला|जयपुरला", "जयपुर ko", text)
    text = re.sub(r"नागपूरला", "नागपूर ko", text)
    text = re.sub(r"(?<![\w])उद्या(?![\w])", "tomorrow", text)
    text = re.sub(r"(?<![\w])परवा(?![\w])", "parson", text)
    text = re.sub(r"\budya\b", "tomorrow", text, flags=re.IGNORECASE)
    text = re.sub(r"\bparva\b", "parson", text, flags=re.IGNORECASE)

    # Shorthand "frm BOM 2 BLR" or "BOM 2 DEL" before cardinal number replacement
    text = re.sub(r"\b(from|frm)\s+([A-Za-z]{3,15})\s+2\s+([A-Za-z]{3,15})\b", r"from \2 to \3", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([A-Za-z]{3})\s+2\s+([A-Za-z]{3})\b", r"\1 to \2", text, flags=re.IGNORECASE)

    # Common typos for travel terms, weekdays, and cities
    text = re.sub(r"\b(?:fligt|flght|fllight|flite|fligts|flghts|flyt)\b", "flight", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfrm\b", "from", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:nxt|nx)\b", "next", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:fryday|frday|fridai|fridy)\b", "friday", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:mnday|mondai|mndy)\b", "monday", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:tuesdai|tusday|tuesdy)\b", "tuesday", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:wednesdai|wensday|wednsday)\b", "wednesday", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:thursdai|thrsday|thurday)\b", "thursday", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:saturdai|satrday|satday)\b", "saturday", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:sundai|snday)\b", "sunday", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:tommorow|tomorow|tmrw|tmr|2mrw|tomm|tomoro)\b", "tomorrow", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:bengluru|banglore|bengalore|bangaluru|bnglr)\b", "bengaluru", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:mumbay|bambai|bomaby|mumabi)\b", "mumbai", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:dehli|delih|dili)\b", "delhi", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:kolkatta|calcuta|kolkta)\b", "kolkata", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:chenai|channai)\b", "chennai", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:hydrabad|hyderbad|hydrbad)\b", "hyderabad", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:ahmadabad|ahemdabad|ahmedbad)\b", "ahmedabad", text, flags=re.IGNORECASE)

    for word, num in INDIC_NUMBERS.items():
        pattern = rf"(?<![\w]){re.escape(word)}(?![\w])"
        text = re.sub(pattern, num, text, flags=re.IGNORECASE)

    for word, num in ENGLISH_ORDINALS.items():
        pattern = rf"(?<![\w]){re.escape(word)}(?![\w])"
        text = re.sub(pattern, num, text, flags=re.IGNORECASE)

    for word, num in ENGLISH_CARDINALS.items():
        pattern = rf"(?<![\w]){re.escape(word)}(?![\w])"
        text = re.sub(pattern, num, text, flags=re.IGNORECASE)

    # Spoken years: "do hajar sattaees" -> "2027", "do hajar chhabis" -> "2026"
    text = re.sub(r"\b(?:do|2)\s*(?:hajar|hazar)\s*(?:sattaees|sataees|sattais|satais|27)\b", "2027", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:do|2)\s*(?:hajar|hazar)\s*(?:chhabees|chhabis|26)\b", "2026", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:do|2)\s*(?:hajar|hazar)\s*(?:pachees|pachis|25)\b", "2025", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:do|2)\s*(?:hajar|hazar)\s*(?:atthaees|athaees|atthais|athais|28)\b", "2028", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:do|2)\s*(?:hajar|hazar)\b", "2000", text, flags=re.IGNORECASE)

    # Spoken hotel keywords typos: "ekin", "chekin", "chek in" -> "check in"
    text = re.sub(r"\b(?:e-?kin|chekin|chek\s*in)\b", "check in", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:chekout|chek\s*out)\b", "check out", text, flags=re.IGNORECASE)

    for word, num in HINGLISH_NUMBERS.items():
        pattern = rf"(?<![\w]){re.escape(word)}(?![\w])"
        text = re.sub(pattern, num, text, flags=re.IGNORECASE)

    for deva_month, eng_month in MONTHS_DEVA.items():
        pattern = rf"(?<![\w]){re.escape(deva_month)}(?![\w])"
        text = re.sub(pattern, eng_month, text, flags=re.IGNORECASE)

    # STT mangling of "next one week" / "economy class" / "saare din"
    text = re.sub(r"\b(?:nekst|nekt|neks)\b", "next", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:veek|wik|vik)\b", "week", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnext\s+(?:van|one|1|ek)\s+week\b", "next week", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsare\s+din\b", "saare din", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:ikonomik|ikonomic|economik|economic|ikonaamik)(?:\s*class)?\b", "economy", text, flags=re.IGNORECASE)

    # Flight cabin class speech/STT typos & Hindi/Hinglish normalization
    text = re.sub(
        r"(?:^|[\s\W])(?:"
        r"busineded|busined|businesd|busnid|busineds|bizness|"
        r"buisness|bussiness|busines|bussnes|bisnes|bijnes|bijnis|biytnes|biyetnes|"
        r"बायटनेस|बिजनेस|बिज़नेस|बिज़नेस"
        r")(?:\s*class)?(?:$|[\s\W])",
        " business class ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:^|[\s\W])(?:"
        r"pharst\s*class|farst\s*class|pharst|farst|फ़र्स्ट\s*क्लास|फर्स्ट\s*क्लास|फ़र्स्ट|फर्स्ट"
        r")(?:$|[\s\W])",
        " first class ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:^|[\s\W])(?:"
        r"ikonaami|ikonomi|इकोनॉमी\s*क्लास|इकॉनमी\s*क्लास|इकोनॉमी|इकॉनमी|इकॉनामी"
        r")(?:$|[\s\W])",
        " economy ",
        text,
        flags=re.IGNORECASE,
    )

    # 3 AC / 3 एसी -> 3AC
    text = re.sub(r"\b([123])\s*(?:AC|A\.C\.|ए\s?सी)\b", r"\1AC", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text
