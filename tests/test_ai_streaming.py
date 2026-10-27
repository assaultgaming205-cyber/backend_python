import asyncio
import json
import unittest
from typing import AsyncIterator, List
import httpx

from app.ai.ai_service import AiService, AiMessage, StreamingThinkingFilter


class MockStreamTransport(httpx.AsyncBaseTransport):
    """Deterministic mock transport that returns pre-configured byte chunks or raises errors."""

    def __init__(self, status_code: int = 200, chunks: List[bytes] = None, exc: Exception = None):
        self.status_code = status_code
        self.chunks = chunks or []
        self.exc = exc
        self.closed = False
        self.request_captured = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.request_captured = request
        if self.exc:
            raise self.exc

        async def byte_stream() -> AsyncIterator[bytes]:
            for chunk in self.chunks:
                yield chunk

        headers = {"content-type": "text/event-stream" if self.status_code == 200 else "application/json"}
        return httpx.Response(
            status_code=self.status_code,
            headers=headers,
            content=byte_stream(),
            request=request,
        )

    async def aclose(self):
        self.closed = True


class TestAiStreaming(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = AiService()
        self.service.api_key = "test_key_nvidiasecret"
        self.service.base_url = "https://mock.nvidia.com/v1"
        self.service.model = "nvidia/nemotron-3.5-lightning-30b-a3b"

    def _patch_client(self, transport: MockStreamTransport):
        """Patches httpx.AsyncClient instantiation in AiService.chat_stream."""
        orig_async_client = httpx.AsyncClient

        def mock_client_factory(*args, **kwargs):
            kwargs["transport"] = transport
            return orig_async_client(*args, **kwargs)

        return mock_client_factory

    # 1. Multiple deltas: fragments arrive separately and are yielded in order
    async def test_multiple_deltas_in_order(self):
        sse_data = (
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Delhi \"}}]}\n\n"
            b"data: {\"choices\":[{\"delta\":{\"content\":\"se \"}}]}\n\n"
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Goa flight\"}}]}\n\n"
            b"data: [DONE]\n\n"
        )
        transport = MockStreamTransport(chunks=[sse_data])
        client_factory = self._patch_client(transport)

        orig_client = httpx.AsyncClient
        httpx.AsyncClient = client_factory
        try:
            tokens = []
            async for chunk in self.service.chat_stream(system_prompt="system", messages=[AiMessage(role="user", content="hi")]):
                tokens.append(chunk)
            self.assertEqual(tokens, ["Delhi ", "se ", "Goa flight"])
        finally:
            httpx.AsyncClient = orig_client

    # 2. SSE framing: comments, blank lines, CRLF, and event data split across chunks
    async def test_sse_framing_and_chunk_splitting(self):
        # Line split across two network chunks
        chunk1 = b": keep-alive comment\r\n\r\n" b"data: {\"choices\":[{\"delta\":{\"content\":\"Sabse "
        chunk2 = b"sasti \"}}]}\r\n\r\n" b"\r\n" b"data: [DONE]\r\n\r\n"
        transport = MockStreamTransport(chunks=[chunk1, chunk2])

        orig_client = httpx.AsyncClient
        httpx.AsyncClient = self._patch_client(transport)
        try:
            tokens = []
            async for chunk in self.service.chat_stream(system_prompt="system", messages=[AiMessage(role="user", content="hi")]):
                tokens.append(chunk)
            self.assertEqual(tokens, ["Sabse sasti "])
        finally:
            httpx.AsyncClient = orig_client

    # 3. Terminal event: [DONE] stops iteration cleanly
    async def test_terminal_done_stops_iteration(self):
        sse_data = (
            b"data: {\"choices\":[{\"delta\":{\"content\":\"End here.\"}}]}\n\n"
            b"data: [DONE]\n\n"
            b"data: {\"choices\":[{\"delta\":{\"content\":\"NEVER YIELD THIS\"}}]}\n\n"
        )
        transport = MockStreamTransport(chunks=[sse_data])

        orig_client = httpx.AsyncClient
        httpx.AsyncClient = self._patch_client(transport)
        try:
            tokens = []
            async for chunk in self.service.chat_stream(system_prompt="system", messages=[AiMessage(role="user", content="hi")]):
                tokens.append(chunk)
            self.assertEqual(tokens, ["End here."])
        finally:
            httpx.AsyncClient = orig_client

    # 4. Empty events: role-only, empty-content, and usage-only events are safely ignored
    async def test_empty_and_metadata_events_ignored(self):
        sse_data = (
            b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n"
            b"data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n\n"
            b"data: {\"usage\":{\"total_tokens\":42}}\n\n"
            b"data: {\"choices\":[]}\n\n"
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Real text\"}}]}\n\n"
            b"data: [DONE]\n\n"
        )
        transport = MockStreamTransport(chunks=[sse_data])

        orig_client = httpx.AsyncClient
        httpx.AsyncClient = self._patch_client(transport)
        try:
            tokens = []
            async for chunk in self.service.chat_stream(system_prompt="system", messages=[AiMessage(role="user", content="hi")]):
                tokens.append(chunk)
            self.assertEqual(tokens, ["Real text"])
        finally:
            httpx.AsyncClient = orig_client

    # 5. Unicode: English, Romanized Hinglish, currency, and emojis preserved correctly
    async def test_unicode_and_hinglish_preservation(self):
        sse_data = (
            "data: {\"choices\":[{\"delta\":{\"content\":\"Haan yaar! Flight ₹4,500 ki hai ✈️\"}}]}\n\n"
            "data: [DONE]\n\n"
        ).encode("utf-8")
        transport = MockStreamTransport(chunks=[sse_data])

        orig_client = httpx.AsyncClient
        httpx.AsyncClient = self._patch_client(transport)
        try:
            tokens = []
            async for chunk in self.service.chat_stream(system_prompt="system", messages=[AiMessage(role="user", content="hi")]):
                tokens.append(chunk)
            self.assertEqual(tokens, ["Haan yaar! Flight ₹4,500 ki hai ✈️"])
        finally:
            httpx.AsyncClient = orig_client

    # 6. Thinking-section filtering: reasoning inside <think>...</think> is completely suppressed,
    # including when tags span across multiple provider chunks
    async def test_thinking_section_filtered_across_chunk_boundaries(self):
        chunk1 = b"data: {\"choices\":[{\"delta\":{\"content\":\"<th\"}}]}\n\n"
        chunk2 = b"data: {\"choices\":[{\"delta\":{\"content\":\"ink>Checking hidden reasoning</th\"}}]}\n\n"
        chunk3 = b"data: {\"choices\":[{\"delta\":{\"content\":\"ink>Bhai mast options hain!\"}}]}\n\n"
        chunk4 = b"data: [DONE]\n\n"
        transport = MockStreamTransport(chunks=[chunk1, chunk2, chunk3, chunk4])

        orig_client = httpx.AsyncClient
        httpx.AsyncClient = self._patch_client(transport)
        try:
            tokens = []
            async for chunk in self.service.chat_stream(system_prompt="system", messages=[AiMessage(role="user", content="hi")]):
                tokens.append(chunk)
            self.assertEqual("".join(tokens), "Bhai mast options hain!")
        finally:
            httpx.AsyncClient = orig_client

    # 7. Malformed provider data: malformed line is safely skipped, stream continues deterministically
    async def test_malformed_sse_data_handled_gracefully(self):
        sse_data = (
            b"data: {\"choices\":[{\"delta\":{\"content\":\"First valid\"}}]}\n\n"
            b"data: {broken json not closed\n\n"
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Second valid\"}}]}\n\n"
            b"data: [DONE]\n\n"
        )
        transport = MockStreamTransport(chunks=[sse_data])

        orig_client = httpx.AsyncClient
        httpx.AsyncClient = self._patch_client(transport)
        try:
            tokens = []
            async for chunk in self.service.chat_stream(system_prompt="system", messages=[AiMessage(role="user", content="hi")]):
                tokens.append(chunk)
            self.assertEqual(tokens, ["First valid", "Second valid"])
        finally:
            httpx.AsyncClient = orig_client

    # 8. HTTP error: non-200 provider response raises a documented RuntimeError
    async def test_http_error_raises_runtime_error(self):
        transport = MockStreamTransport(status_code=500, chunks=[b"{\"error\": \"Internal server error\"}"])

        orig_client = httpx.AsyncClient
        httpx.AsyncClient = self._patch_client(transport)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                async for _ in self.service.chat_stream(system_prompt="system", messages=[AiMessage(role="user", content="hi")]):
                    pass
            self.assertIn("NVIDIA chat completions stream failed with status 500", str(ctx.exception))
        finally:
            httpx.AsyncClient = orig_client

    # 9. Timeout and connection failure: stream propagates timeout exception clearly
    async def test_timeout_propagates_exception(self):
        transport = MockStreamTransport(exc=httpx.ReadTimeout("Read timed out"))

        orig_client = httpx.AsyncClient
        httpx.AsyncClient = self._patch_client(transport)
        try:
            with self.assertRaises(httpx.ReadTimeout):
                async for _ in self.service.chat_stream(system_prompt="system", messages=[AiMessage(role="user", content="hi")]):
                    pass
        finally:
            httpx.AsyncClient = orig_client

    # 10. Cancellation: closing generator releases resources without dangling tasks
    async def test_cancellation_closes_stream(self):
        chunk1 = b"data: {\"choices\":[{\"delta\":{\"content\":\"Chunk1\"}}]}\n\n"
        chunk2 = b"data: {\"choices\":[{\"delta\":{\"content\":\"Chunk2\"}}]}\n\n"
        transport = MockStreamTransport(chunks=[chunk1, chunk2])

        orig_client = httpx.AsyncClient
        httpx.AsyncClient = self._patch_client(transport)
        try:
            gen = self.service.chat_stream(system_prompt="system", messages=[AiMessage(role="user", content="hi")])
            first = await gen.__anext__()
            self.assertEqual(first, "Chunk1")
            # Close generator early (client cancelled)
            await gen.aclose()
        finally:
            httpx.AsyncClient = orig_client

    # 11. Non-streaming regression: existing chat() remains non-streaming and retains contract
    async def test_non_streaming_chat_regression(self):
        mock_body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "{\"reply\": \"Delhi se Mumbai ki flights mil gayi.\"}"
                    }
                }
            ]
        }
        mock_bytes = json.dumps(mock_body).encode("utf-8")

        async def handler(request: httpx.Request) -> httpx.Response:
            # Verify stream is NOT true in non-streaming request
            body = json.loads(request.read())
            self.assertNotIn("stream", body)
            return httpx.Response(200, content=mock_bytes, headers={"content-type": "application/json"})

        transport = httpx.MockTransport(handler)
        orig_client = httpx.AsyncClient

        def mock_factory(*args, **kwargs):
            kwargs["transport"] = transport
            return orig_client(*args, **kwargs)

        httpx.AsyncClient = mock_factory
        try:
            res = await self.service.chat(
                system_prompt="system prompt",
                messages=[AiMessage(role="user", content="flights")],
            )
            self.assertEqual(res.message.role, "assistant")
            self.assertIn("flights mil gayi", res.content)
            self.assertEqual(res.model, self.service.model)
        finally:
            httpx.AsyncClient = orig_client

    # 12. No request leakage: verifies authorization header and payload parameters without leaking
    async def test_no_request_credential_leakage(self):
        captured_request = None

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_request
            captured_request = request
            sse = b"data: {\"choices\":[{\"delta\":{\"content\":\"safe\"}}]}\n\ndata: [DONE]\n\n"
            return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

        transport = httpx.MockTransport(handler)
        orig_client = httpx.AsyncClient

        def mock_factory(*args, **kwargs):
            kwargs["transport"] = transport
            return orig_client(*args, **kwargs)

        httpx.AsyncClient = mock_factory
        try:
            tokens = []
            async for token in self.service.chat_stream(system_prompt="secret_sys", messages=[AiMessage(role="user", content="hello")]):
                tokens.append(token)

            self.assertEqual(tokens, ["safe"])
            # Tokens yielded must not contain API key
            self.assertNotIn(self.service.api_key, tokens[0])
            # Verify request headers and payload
            self.assertIsNotNone(captured_request)
            self.assertEqual(captured_request.headers["authorization"], f"Bearer {self.service.api_key}")
            body = json.loads(captured_request.read())
            self.assertTrue(body.get("stream"))
            self.assertEqual(body.get("model"), self.service.model)
        finally:
            httpx.AsyncClient = orig_client

    async def _collect_stream_text(self, fragments: List[str]) -> str:
        """Helper to simulate SSE stream yielding fragments and collect exact concatenated visible text."""
        chunks = []
        for frag in fragments:
            sse_line = f'data: {json.dumps({"choices": [{"delta": {"content": frag}}]})}\n\n'.encode("utf-8")
            chunks.append(sse_line)
        chunks.append(b"data: [DONE]\n\n")

        transport = MockStreamTransport(chunks=chunks)
        orig_client = httpx.AsyncClient
        httpx.AsyncClient = self._patch_client(transport)
        try:
            tokens = []
            async for chunk in self.service.chat_stream(system_prompt="sys", messages=[AiMessage(role="user", content="u")]):
                tokens.append(chunk)
            return "".join(tokens)
        finally:
            httpx.AsyncClient = orig_client

    # 13. <response> opening tag split across chunks
    async def test_response_opening_tag_split_across_chunks(self):
        result = await self._collect_stream_text(["<res", "ponse>Delhi flights"])
        self.assertEqual(result, "Delhi flights")

    # 14. </response> closing tag split across chunks
    async def test_response_closing_tag_split_across_chunks(self):
        result = await self._collect_stream_text(["Delhi flights</resp", "onse>"])
        self.assertEqual(result, "Delhi flights")

    # 15. Mixed-case markup split across chunks
    async def test_mixed_case_markup_split_across_chunks(self):
        result_resp = await self._collect_stream_text(["<Res", "Ponse>Goa</resP", "onse>"])
        self.assertEqual(result_resp, "Goa")

        result_think = await self._collect_stream_text([
            "<TH", "INK>internal reasoning hidden</TH", "INK>Done"
        ])
        self.assertEqual(result_think, "Done")

    # 16. An incomplete <think> section at end of stream (fail closed, suppress internal section)
    async def test_incomplete_think_section_at_end_of_stream(self):
        result_pure = await self._collect_stream_text(["<think>unfinished thoughts without closing"])
        self.assertEqual(result_pure, "")

        result_prefix_think = await self._collect_stream_text([
            "Confirmed: ", "<th", "ink>unfinished reasoning"
        ])
        self.assertEqual(result_prefix_think, "Confirmed: ")

    # 17. An incomplete potential markup prefix at end of stream (flush safely)
    async def test_incomplete_potential_markup_prefix_at_end_of_stream(self):
        result_less = await self._collect_stream_text(["Price is <"])
        self.assertEqual(result_less, "Price is <")

        result_th = await self._collect_stream_text(["Duration <", "th"])
        self.assertEqual(result_th, "Duration <th")

        result_res = await self._collect_stream_text(["Options </", "res"])
        self.assertEqual(result_res, "Options </res")

    # 18. Ordinary text containing <3, mathematical symbols, and Unicode
    async def test_ordinary_text_containing_less_than_math_and_unicode(self):
        # Full fragment with <3, math < and >, and Unicode hindi/currency/emojis
        full_text = "I <3 travel: fare is ₹4,500 < ₹6,000 & 5 < 10 for मुंबई ✈️"
        result_full = await self._collect_stream_text([full_text])
        self.assertEqual(result_full, full_text)

        # Split across chunks precisely at '<' and '3'
        result_split = await self._collect_stream_text(["less than <", "3 flights available"])
        self.assertEqual(result_split, "less than <3 flights available")

    # 19. Multiple internal thinking sections in one response
    async def test_multiple_internal_thinking_sections(self):
        result = await self._collect_stream_text([
            "Step 1 ",
            "<think>thought 1</think>",
            " Step 2 ",
            "<think>thought 2</think>",
            " Step 3",
        ])
        self.assertEqual(result, "Step 1  Step 2  Step 3")

    # 20. Normal content immediately before and after internal sections
    async def test_normal_content_before_and_after_internal_sections(self):
        result = await self._collect_stream_text([
            "Before thoughts ",
            "<think>internal step by step reasoning</think>",
            " After thoughts",
        ])
        self.assertEqual(result, "Before thoughts  After thoughts")

    # 21. Thinking tags with whitespace and attributes
    async def test_thinking_and_response_tags_with_whitespace_and_attributes(self):
        # Attribute variations in <think>
        res1 = await self._collect_stream_text([
            '<think attr="value" score=0.9>hidden reasoning</think>Visible content'
        ])
        self.assertEqual(res1, "Visible content")

        # Whitespace variations in <think > and </think >
        res2 = await self._collect_stream_text([
            "<think >hidden reasoning</think >Visible content"
        ])
        self.assertEqual(res2, "Visible content")

        # Whitespace and attributes in <response>
        res3 = await self._collect_stream_text([
            '<response id="resp_1" status="ok">Clean answer</response >'
        ])
        self.assertEqual(res3, "Clean answer")

    # 22. Delimiters with attributes/whitespace split across arbitrary chunks
    async def test_attribute_tags_split_across_arbitrary_chunks(self):
        # Opening <think attr="..."> split across chunks
        res1 = await self._collect_stream_text([
            '<think attr="', 'val">internal thoughts</think>visible'
        ])
        self.assertEqual(res1, "visible")

        # Opening <think > split at space
        res2 = await self._collect_stream_text([
            '<think ', '>internal thoughts</think>visible'
        ])
        self.assertEqual(res2, "visible")

        # Closing </think > split at space
        res3 = await self._collect_stream_text([
            '<think>internal thoughts</think ', '>visible'
        ])
        self.assertEqual(res3, "visible")

        # Opening <response id="..."> split across chunks
        res4 = await self._collect_stream_text([
            '<response ', 'id="1">Hello world</response>'
        ])
        self.assertEqual(res4, "Hello world")

    # 23. Unterminated thinking tag with attributes at end of stream (fail closed)
    async def test_unterminated_attribute_think_fails_closed(self):
        res1 = await self._collect_stream_text([
            '<think model="nemotron">Unfinished internal reasoning'
        ])
        self.assertEqual(res1, "")

        res2 = await self._collect_stream_text([
            'Safe prefix: ', '<think attr="val">Unfinished internal reasoning'
        ])
        self.assertEqual(res2, "Safe prefix: ")

    # 24. Literal text resembling markup is preserved
    async def test_literal_text_resembling_markup_preserved(self):
        res1 = await self._collect_stream_text([
            "We are <thinking about options> for the user"
        ])
        self.assertEqual(res1, "We are <thinking about options> for the user")

        res2 = await self._collect_stream_text([
            "Select <destination> for flight < 5000"
        ])
        self.assertEqual(res2, "Select <destination> for flight < 5000")


class TestStreamingThinkingFilterDirect(unittest.TestCase):
    """Direct unit tests for the deterministic stateful StreamingThinkingFilter parser."""

    def _filter_run(self, chunks: List[str]) -> str:
        f = StreamingThinkingFilter()
        out = [f.feed(c) for c in chunks]
        fin = f.finalize()
        if fin:
            out.append(fin)
        return "".join(out)

    def test_filter_exact_outputs(self):
        cases = [
            (["<response>Hello</response>"], "Hello"),
            (["<Response>", "World", "</Response>"], "World"),
            (["<res", "ponse>Delhi", "</resp", "onse>"], "Delhi"),
            (["<th", "ink>thought</th", "ink>Visible"], "Visible"),
            (["<THINK>thought</THINK>Visible"], "Visible"),
            (["<think>unfinished"], ""),
            (["Normal text <", "3 flights"], "Normal text <3 flights"),
            (["Normal text <3 flights"], "Normal text <3 flights"),
            (["Normal text <"], "Normal text <"),
            (["Normal text <th"], "Normal text <th"),
            (["Normal text <train"], "Normal text <train"),
            (["Start <think>t1</think> Mid <think>t2", "</think> End"], "Start  Mid  End"),
            (["Haan ₹4500 ✈️"], "Haan ₹4500 ✈️"),
            (["x < y and y > z"], "x < y and y > z"),
            (["<response><think>reasoning</think>Answer</response>"], "Answer"),
            (["a << b"], "a << b"),
            (["a <", "< b"], "a << b"),
            # Whitespace and attribute test cases
            (["<think attr=\"value\">thought</think>visible"], "visible"),
            (["<think >thought</think >visible"], "visible"),
            (["<think\n>thought</think\n>visible"], "visible"),
            (["<think attr=\"", "val\">thought</think>visible"], "visible"),
            (["<think ", "attr=\"val\">thought</think>visible"], "visible"),
            (["<think ", ">thought</think>visible"], "visible"),
            (["<response ", "id=\"1\">hello</response>"], "hello"),
            (["<think>thought</think ", ">visible"], "visible"),
            (["<think>thought</", "think >visible"], "visible"),
            (["<THINK attr=\"X\">thought</THINK>visible"], "visible"),
            (["<Response id=\"1\">hello</Response>"], "hello"),
            (["<Res", "ponse id=\"1\">hello</Response>"], "hello"),
            (["<think attr=\"val\">unfinished"], ""),
            (["<th", "ink attr=\"val\">unfinished"], ""),
            (["Before <think attr=\"val\">unfinished"], "Before "),
            (["We are <thinking about travel>"], "We are <thinking about travel>"),
            (["Incomplete: fare <think attr=\""], "Incomplete: fare <think attr=\""),
        ]
        for chunks, expected in cases:
            with self.subTest(chunks=chunks):
                self.assertEqual(self._filter_run(chunks), expected)


if __name__ == "__main__":
    unittest.main()
