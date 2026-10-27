import re
from typing import Optional


def classify_turn(text: str, previous_intent: Optional[str] = None, has_explicit_route: bool = False) -> str:
    t = text.lower()
    if re.search(r"\b(cancel|stop|chodo|rehne do|start over|new trip|reset)\b|रद्द|नया ट्रिप", t):
        return "cancel_reset"
    if re.search(r"fare.*kitna|is train ka fare", t):
        return "side_question"
    if re.search(r"\b(book|booking|checkout|visited|visit history)\b|pehle nahi gaya|बुकिंग", t):
        return "side_question"
    if re.search(r"museum.*open|ticket.*(?:kitna|price)|opening hours|\bwalk(?:ing)?\b|kitni der", t):
        return "discover_nearby_places"
    if re.search(r"\b(airport pickup|shuttle|cab included|breakfast|baggage|cancellation|refund)\b|पिकअप", t):
        return "side_question"
    if re.search(r"\b(nearby|paas kya|dekh sakte|sightseeing)\b|आस.?पास", t):
        return "discover_nearby_places"
    if re.search(r"\b(tum suggest karo|tu suggest kar|aap suggest karo|tum batao|tu bata|aap batao|suggest karo|you suggest|recommend|kab jana chahiye|kab jau|kab jaye|when should i go|when to travel|cheapest.*kab|sasta.*kab|cheapest date|sasti date|cheapest batao|lowest fare.*when|date fixed nahi|tarikh fix nahi|koi bhi din|any day|anytime|flexible|next.*(?:week|7\s*days?|hafte|saptaah)|agle.*(?:hafte|7\s*din)|kabhi\s*bhi|kabhee\s*bhi)\b", t, re.IGNORECASE):
        return "flexible_date_inquiry"
    if (re.search(r"\b(?:train|rail|railway).*(?:ya|or|versus|vs|better|se|phir|sasta|sasti|cheapest|cheap|me se|mein se).*(?:flight|flights|plane|planes|air|airways)\b", t, re.IGNORECASE) or
        re.search(r"\b(?:flight|flights|plane|planes|air|airways).*(?:ya|or|versus|vs|better|se|phir|sasta|sasti|cheapest|cheap|me se|mein se).*(?:train|rail|railway)\b", t, re.IGNORECASE) or
        re.search(r"\b(?:train.*(?:better|ya flight|ya plane|ya phir plane|ya phir flight)|flight.*(?:better|ya train|ya phir train)|plane.*(?:better|ya train|ya phir train)|compare|vs|kisse jau|kis se jau|dono compare|dono mein se|dono me se)\b", t, re.IGNORECASE)):
        return "compare_modes"
    if re.search(r"\b(business.*(?:expensive|costly|mehng)|expensive.*business)\b", t):
        return "side_question"
    if has_explicit_route:
        return "search_trip"
    if re.search(r"\b(bore|bored|boring|kya haal|kya chal raha|time pass)\b", t):
        return "small_talk"
    if re.search(r"\b(suggest|inspiration|surprise|getaway|hidden gem|kahi.*(?:ghum|ghoom)|kahan jau|kuch.*jagah)\b|घूमने|सुझाव", t):
        return "destination_inspiration"
    if re.search(r"\b(hotel|hotels|flight|flights|train|trains)\b|फ्लाइट|ट्रेन|होटल", t):
        return "search_trip"
    if re.search(r"\b(hills?|pahad|mountains?|beach|beaches|peaceful|adventure|heritage)\b|पहाड़|समुद्र", t):
        return "destination_inspiration"
    if previous_intent in ["destination_inspiration", "small_talk"]:
        return "destination_inspiration"
    return "answer_question"
