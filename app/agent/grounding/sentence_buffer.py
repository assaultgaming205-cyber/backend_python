import re
from typing import List

# Titles that precede capitalized names without ending a sentence
TITLES = {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st"}

# General abbreviations that only avoid sentence boundaries when followed by lowercase or numbers
ABBREVIATIONS = {
    "approx", "vs", "eg", "e.g", "ie", "i.e", "no", "min", "hrs", "hr",
    "p.m", "a.m", "pm", "am", "rs", "inr", "co", "corp", "inc", "ltd", "dept", "est"
}


class SentenceBuffer:
    """Buffers streaming LLM token fragments into complete conversational sentences.

    Features:
    - Strips JSON wrappers (e.g. {"reply": "..."} or ```json blocks) when present.
    - Prevents accidental splitting on:
      * Times (e.g. 7:25, 14:30)
      * Decimals (e.g. 3.5, 4.8)
      * Currency with dots (e.g. Rs. 4500, ₹4,500.00)
      * Standard abbreviations (e.g. approx., vs., e.g., i.e., min., hrs.)
      * Ellipsis (...)
    - Buffers incomplete sentences until completion delimiters (. ! ? । ॥ or \n\n) arrive.
    - Flushes any legitimate remaining final sentence upon finalize().
    - Preserves Unicode, Hinglish, emojis, and mathematical symbols.
    """

    def __init__(self):
        self.buffer: str = ""
        self.is_prefix_resolved: bool = False
        self.is_json: bool = False

    def _strip_envelope_prefix(self, text: str) -> tuple[str, bool]:
        """Detects and strips opening JSON envelope if present."""
        m = re.match(
            r"^\s*(?:```(?:json)?\s*)?\{\s*\"(?:reply|message|content|text|answer)\"\s*:\s*\"",
            text,
            re.IGNORECASE,
        )
        if m:
            return text[m.end():], True
        return text, False

    def feed(self, chunk: str) -> List[str]:
        """Consumes a streaming chunk and returns any completed sentences."""
        if not chunk:
            return []

        if not self.is_prefix_resolved:
            self.buffer += chunk
            # Check if this buffer might still be an incomplete JSON opening bracket
            stripped, is_json = self._strip_envelope_prefix(self.buffer)
            if is_json:
                self.buffer = stripped
                self.is_json = True
                self.is_prefix_resolved = True
            elif len(self.buffer) < 40 and "{" in self.buffer and not any(c in self.buffer for c in ".!?\n।"):
                # Wait for more tokens to decide whether it's JSON
                return []
            else:
                self.buffer = stripped
                self.is_json = is_json
                self.is_prefix_resolved = True
        else:
            self.buffer += chunk

        sentences: List[str] = []
        while True:
            boundary_idx = self._find_sentence_boundary(self.buffer)
            if boundary_idx == -1:
                break
            raw_sentence = self.buffer[:boundary_idx + 1].strip()
            self.buffer = self.buffer[boundary_idx + 1:].lstrip()
            clean_s = self._clean_sentence(raw_sentence)
            if clean_s:
                sentences.append(clean_s)

        return sentences

    def finalize(self) -> List[str]:
        """Flushes any remaining text in the buffer as the final sentence."""
        sentences: List[str] = []
        rem = self.buffer.strip()
        self.buffer = ""
        if rem:
            clean_s = self._clean_sentence(rem, is_final=True)
            if clean_s:
                sentences.append(clean_s)
        return sentences

    def _clean_sentence(self, text: str, is_final: bool = False) -> str:
        s = text
        if self.is_json:
            s = s.replace(r"\"", "\"").replace(r"\n", "\n").replace(r"\/", "/")
            if is_final:
                s = re.sub(r"\"\s*\}?\s*```?\s*$", "", s).strip()
        # Clean any accidental JSON tags or fences
        s = re.sub(r"^\s*\{[\s\S]*?\"(?:reply|message)\"\s*:\s*\"", "", s)
        if is_final:
            s = re.sub(r"\"\s*\}\s*$", "", s)
        # Normalize internal spacing
        s = re.sub(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]", " ", s)
        return s.strip()

    def _find_sentence_boundary(self, text: str) -> int:
        for idx in range(len(text)):
            char = text[idx]
            if char in ".!?।॥\n":
                if char == "\n":
                    # Double newline indicates a paragraph/sentence boundary
                    if idx + 1 < len(text) and text[idx + 1] == "\n":
                        return idx
                    continue

                if char == ".":
                    # Check ellipsis (...)
                    if (idx > 0 and text[idx - 1] == ".") or (idx + 1 < len(text) and text[idx + 1] == "."):
                        continue
                    # Check decimals (e.g. 3.5, 4,500.00)
                    if idx > 0 and idx + 1 < len(text) and text[idx - 1].isdigit() and text[idx + 1].isdigit():
                        continue
                    # Check abbreviations
                    prev_word_match = re.search(r"([a-zA-Z.]+)\Z", text[:idx])
                    if prev_word_match:
                        prev_word = prev_word_match.group(1).lower().rstrip(".")
                        if prev_word in TITLES:
                            continue
                        if prev_word in ABBREVIATIONS:
                            rem_after = text[idx + 1:].lstrip()
                            if not (rem_after and rem_after[0].isupper()):
                                continue

                # Ensure there is a following space, closing quote, bracket, or delimiter
                if idx + 1 == len(text):
                    # Boundary at end of current buffer: wait for next chunk unless finalize()
                    return -1
                if text[idx + 1].isspace() or text[idx + 1] in "\"\x27)]}":
                    return idx
        return -1
