import asyncio
import re
import json
import logging
import httpx
from typing import List, Dict, Any, Optional, AsyncIterator
from pydantic import BaseModel
from ..config import settings

logger = logging.getLogger("ai_service")


def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>[\s\S]*?(?:<\/think>|$)", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"^[\s\S]*?<\/think>", "", cleaned, flags=re.IGNORECASE)
    m = re.search(r"<response>([\s\S]*?)<\/response>", cleaned, flags=re.IGNORECASE)
    if m:
        cleaned = m.group(1)
    # Normalize punctuation: replace Devanagari danda with period
    cleaned = cleaned.replace("।", ".").replace("॥", ".")
    # Clean any accidental robotic quotes around 'sabki batao'
    cleaned = re.sub(r"[\'\"]\s*sab(?:ki|akee|ka)?\s*batao\s*[\'\"]", "saari classes", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bya\s+[\'\"]?sab(?:ki|akee|ka)?\s*batao[\'\"]?\??", "ya saari classes?", cleaned, flags=re.IGNORECASE)
    # Transliterate common Devanagari script slips (e.g. from 30B model) into Roman Hinglish
    _dev_map = {
        "सुबह": "subah", "शाम": "shaam", "रात": "raat", "दोपहर": "dopahar",
        "बजे": "baje", "रुपये": "₹", "रूपये": "₹", "हवाई": "hawai", "अड्डा": "adda",
        "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
        "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
    }
    for dev, rom in _dev_map.items():
        cleaned = cleaned.replace(dev, rom)
    return cleaned.strip()


OPEN_TAG_RE = re.compile(r"^<\s*(think|response)(?=[>\s/])(?:[^>]*)?>", re.IGNORECASE)
CLOSE_TAG_RE = re.compile(r"^<\s*/\s*(think|response)\s*>", re.IGNORECASE)
CLOSE_THINK_RE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)

CLOSE_THINK_PREFIX_RE = re.compile(
    r"^<\s*(?:/\s*(?:t(?:h(?:i(?:n(?:k\s*)?)?)?)?)?)?$",
    re.IGNORECASE,
)

OPEN_OR_CLOSE_PREFIX_RE = re.compile(
    r"^(?:"
    r"<\s*$"
    r"|<\s*/\s*$"
    r"|<\s*/\s*t(?:h(?:i(?:n(?:k\s*)?)?)?)?$"
    r"|<\s*/\s*r(?:e(?:s(?:p(?:o(?:n(?:s(?:e\s*)?)?)?)?)?)?)?$"
    r"|<\s*t(?:h(?:i(?:n(?:k)?)?)?)?$"
    r"|<\s*r(?:e(?:s(?:p(?:o(?:n(?:s(?:e)?)?)?)?)?)?)?$"
    r"|<\s*(?:think|response)(?:$|[\s/][^>]*$)"
    r")$",
    re.IGNORECASE,
)

MAX_TAG_BUFFER = 256


class StreamingThinkingFilter:
    """Incrementally filters <think>...</think> and </?response> blocks across streaming token chunks.

    Supports case-insensitive markup, whitespace, and attribute variations (e.g. <think attr="value">, <think >)
    while preserving ordinary mathematical text (e.g. <3, < 5) and Unicode without leaking internal reasoning.
    Buffering is strictly bounded to MAX_TAG_BUFFER characters to prevent unbounded memory growth.
    """

    def __init__(self):
        self.in_thinking = False
        self.buffer = ""

    def feed(self, chunk: str) -> str:
        text = self.buffer + chunk
        self.buffer = ""
        out = []

        while text:
            if self.in_thinking:
                m = CLOSE_THINK_RE.search(text)
                if m:
                    self.in_thinking = False
                    text = text[m.end():]
                else:
                    pos = text.rfind("<")
                    if pos != -1:
                        tail = text[pos:]
                        if len(tail) <= 30 and CLOSE_THINK_PREFIX_RE.match(tail):
                            self.buffer = tail
                    text = ""
            else:
                idx = text.find("<")
                if idx == -1:
                    out.append(text)
                    text = ""
                else:
                    if idx > 0:
                        out.append(text[:idx])
                        text = text[idx:]

                    m_open = OPEN_TAG_RE.match(text)
                    if m_open:
                        tag_name = m_open.group(1).lower()
                        if tag_name == "think":
                            self.in_thinking = True
                        text = text[m_open.end():]
                        continue

                    m_close = CLOSE_TAG_RE.match(text)
                    if m_close:
                        text = text[m_close.end():]
                        continue

                    if len(text) <= MAX_TAG_BUFFER and OPEN_OR_CLOSE_PREFIX_RE.match(text):
                        self.buffer = text
                        text = ""
                    else:
                        out.append(text[0])
                        text = text[1:]

        return "".join(out)

    def finalize(self) -> str:
        if self.in_thinking:
            self.buffer = ""
            self.in_thinking = False
            return ""
        rem = self.buffer
        self.buffer = ""
        return rem


class AiMessage(BaseModel):
    role: str
    content: str
    selected_offer_id: Optional[str] = None
    clarification_answer: Optional[Dict[str, Any]] = None


class AiResponsePayload(BaseModel):
    class Message(BaseModel):
        role: str = "assistant"
        content: str

    message: Message
    content: str
    raw: str
    model: str


class AiService:
    def __init__(self):
        self.api_key = settings.NVIDIA_API_KEY
        self.base_url = settings.NVIDIA_BASE_URL.rstrip("/")
        self.model = settings.NVIDIA_MODEL
        self.model_name = self.model
        self.timeout_sec = settings.NVIDIA_TIMEOUT_MS / 1000.0

    async def health_check(self) -> bool:
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def chat(
        self,
        system_prompt: str,
        messages: Optional[List[AiMessage]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.65,
        max_tokens: int = 1200,
        response_format: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AiResponsePayload:
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is not configured in environment.")

        target_model = model or self.model

        formatted_messages = [{"role": "system", "content": system_prompt}]
        if messages:
            for m in messages:
                formatted_messages.append({"role": m.role, "content": m.content})
        elif history:
            for m in history:
                formatted_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        if response_format:
            instruction = f"\nReturn ONLY valid JSON matching this schema: {json.dumps(response_format)}. Return raw JSON only, without markdown fences."
            formatted_messages[0]["content"] += instruction

        payload = {
            "model": target_model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": min(max_tokens, 600),
        }
        if "nemotron" in target_model:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if response_format:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "travel_response", "strict": True, "schema": response_format},
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        url = f"{self.base_url}/chat/completions"
        logger.info(f"Calling NVIDIA NIM ({target_model})...")

        data = None
        for attempt in range(3):
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                break
            logger.error(f"NVIDIA API error: status {resp.status_code}, text: {resp.text}")
            if resp.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"NVIDIA chat completions failed with status {resp.status_code}")
            await asyncio.sleep(1.5 * (attempt + 1))

        choices = data.get("choices") or []
        content = ""
        if choices:
            content = choices[0].get("message", {}).get("content", "")

        cleaned_content = strip_thinking(content)
        return AiResponsePayload(
            message=AiResponsePayload.Message(role="assistant", content=cleaned_content),
            content=cleaned_content,
            raw=content,
            model=target_model,
        )

    async def chat_stream(
        self,
        system_prompt: str,
        messages: Optional[List[AiMessage]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.65,
        max_tokens: int = 1200,
        response_format: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is not configured in environment.")

        target_model = model or self.model

        formatted_messages = [{"role": "system", "content": system_prompt}]
        if messages:
            for m in messages:
                formatted_messages.append({"role": m.role, "content": m.content})
        elif history:
            for m in history:
                formatted_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        if response_format:
            instruction = f"\nReturn ONLY valid JSON matching this schema: {json.dumps(response_format)}. Return raw JSON only, without markdown fences."
            formatted_messages[0]["content"] += instruction

        payload = {
            "model": target_model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": min(max_tokens, 600),
            "stream": True,
        }
        if "nemotron" in target_model:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if response_format:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "travel_response", "strict": True, "schema": response_format},
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        url = f"{self.base_url}/chat/completions"
        logger.info(f"Initiating NVIDIA NIM stream ({target_model})...")

        filter_ = StreamingThinkingFilter()

        client = httpx.AsyncClient(timeout=self.timeout_sec)
        try:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    logger.error(
                        f"NVIDIA streaming error: status {response.status_code}, text: {err_body.decode('utf-8', errors='replace')}"
                    )
                    raise RuntimeError(f"NVIDIA chat completions stream failed with status {response.status_code}")

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue

                    raw_data = line[5:].strip()
                    if raw_data == "[DONE]":
                        break

                    try:
                        event_data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Malformed SSE JSON from provider: {raw_data[:80]}")
                        continue

                    choices = event_data.get("choices") or []
                    if not choices:
                        continue

                    delta = choices[0].get("delta") or {}
                    content_fragment = delta.get("content")
                    if not content_fragment:
                        continue

                    filtered = filter_.feed(content_fragment)
                    if filtered:
                        yield filtered

            final_chunk = filter_.finalize()
            if final_chunk:
                yield final_chunk

        finally:
            await client.aclose()
