import json
import os
import unittest
from unittest.mock import patch

from httpx import (ASGITransport, AsyncClient, MockTransport, ReadTimeout,
                   Request, Response)

import app.main as main_module
from app.services.cache_service import (InMemoryCommentsAnalysisCache,
                                        InMemoryTTLCache)
from app.services.comments_service import (CommentSample,
                                           CommentsDisabledError,
                                           CommentsError,
                                           CommentsNotConfiguredError,
                                           NoCommentsError, PublicComment,
                                           fetch_comment_sample,
                                           prepare_comments_for_analysis)
from app.services.transcript_service import TranscriptError


def comment_item(comment_id: str, text: str | None = None) -> dict:
    return {
        "id": f"thread-{comment_id}",
        "snippet": {
            "topLevelComment": {
                "id": comment_id,
                "snippet": {
                    "textDisplay": text or f"Testo {comment_id}",
                    "likeCount": 3,
                    "publishedAt": "2026-07-10T10:00:00Z",
                    "authorDisplayName": "Non deve essere usato",
                    "authorChannelId": {"value": "private-author-id"},
                },
            }
        },
    }


def parse_ndjson(body: str) -> list[dict]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


class CommentsServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_balances_paginates_and_deduplicates_300_comments(self):
        requests: list[Request] = []

        def handler(request: Request) -> Response:
            requests.append(request)
            order = request.url.params["order"]
            page = request.url.params.get("pageToken")
            if order == "relevance" and not page:
                payload = {
                    "items": [comment_item(f"r{i}") for i in range(100)],
                    "nextPageToken": "relevance-2",
                }
            elif order == "relevance":
                payload = {"items": [comment_item(f"r{i}") for i in range(100, 200)]}
            elif not page:
                payload = {
                    "items": [comment_item(f"r{i}") for i in range(50)]
                    + [comment_item(f"t{i}") for i in range(50)],
                    "nextPageToken": "time-2",
                }
            else:
                payload = {"items": [comment_item(f"t{i}") for i in range(50, 150)]}
            return Response(200, json=payload)

        async with AsyncClient(transport=MockTransport(handler)) as client:
            sample = await fetch_comment_sample(
                "dQw4w9WgXcQ", api_key="test-key", client=client
            )

        self.assertEqual(sample.found_count, 300)
        self.assertEqual(sample.relevant_count, 150)
        self.assertEqual(sample.recent_count, 150)
        self.assertEqual(len({item.comment_id for item in sample.comments}), 300)
        self.assertEqual(len(requests), 4)
        self.assertTrue(all(request.url.params["textFormat"] == "plainText" for request in requests))
        self.assertFalse(hasattr(sample.comments[0], "author"))

    async def test_requires_server_side_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(CommentsNotConfiguredError):
                await fetch_comment_sample("dQw4w9WgXcQ")

    async def test_maps_comments_disabled_error(self):
        def handler(_request: Request) -> Response:
            return Response(
                403,
                json={"error": {"errors": [{"reason": "commentsDisabled"}]}},
            )

        async with AsyncClient(transport=MockTransport(handler)) as client:
            with self.assertRaises(CommentsDisabledError):
                await fetch_comment_sample(
                    "dQw4w9WgXcQ", api_key="test-key", client=client
                )

    async def test_rejects_empty_public_sample(self):
        async with AsyncClient(
            transport=MockTransport(lambda _request: Response(200, json={"items": []}))
        ) as client:
            with self.assertRaises(NoCommentsError):
                await fetch_comment_sample(
                    "dQw4w9WgXcQ", api_key="test-key", client=client
                )

    async def test_maps_youtube_timeout(self):
        def handler(request: Request) -> Response:
            raise ReadTimeout("timeout", request=request)

        async with AsyncClient(transport=MockTransport(handler)) as client:
            with self.assertRaises(CommentsError) as raised:
                await fetch_comment_sample(
                    "dQw4w9WgXcQ", api_key="test-key", client=client
                )
        self.assertEqual(raised.exception.code, "youtube_timeout")

    def test_prepares_balanced_bounded_anonymous_input(self):
        sample = CommentSample(
            comments=[
                PublicComment("secret-r", "ABCDEFGHIJ", 8, "2026-07-10", "relevance"),
                PublicComment("secret-t", "1234567890", 2, "2026-07-09", "time"),
            ],
            relevant_count=1,
            recent_count=1,
        )
        text, analyzed_count = prepare_comments_for_analysis(
            sample, per_comment_limit=5, total_limit=200
        )

        self.assertEqual(analyzed_count, 2)
        self.assertIn("ABCDE", text)
        self.assertIn("12345", text)
        self.assertNotIn("secret-r", text)
        self.assertNotIn("secret-t", text)
        self.assertLessEqual(len(text), 200)


class CommentsEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_comments_cache = main_module.comments_analysis_cache
        self.original_summary_cache = main_module.summary_cache
        main_module.comments_analysis_cache = InMemoryCommentsAnalysisCache()
        main_module.summary_cache = InMemoryTTLCache()
        self.client = AsyncClient(
            transport=ASGITransport(app=main_module.app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        main_module.comments_analysis_cache = self.original_comments_cache
        main_module.summary_cache = self.original_summary_cache

    async def test_streams_analysis_then_serves_it_from_cache(self):
        sample = CommentSample(
            comments=[
                PublicComment("r1", "Molto utile", 4, "2026-07-10", "relevance"),
                PublicComment("t1", "Ho una domanda", 0, "2026-07-10", "time"),
            ],
            relevant_count=1,
            recent_count=1,
        )

        async def fake_stream(*_args, **_kwargs):
            yield "## In breve\n"
            yield "Commenti generalmente positivi."

        with (
            patch.object(main_module, "fetch_comment_sample", return_value=sample) as fetch_mock,
            patch.object(main_module, "stream_analyze_comments", new=fake_stream),
        ):
            first = await self.client.post(
                "/api/comments/analyze/stream", json={"video_id": "dQw4w9WgXcQ"}
            )
            second = await self.client.post(
                "/api/comments/analyze/stream", json={"video_id": "dQw4w9WgXcQ"}
            )

        first_events = parse_ndjson(first.text)
        second_events = parse_ndjson(second.text)
        self.assertEqual([item["type"] for item in first_events], ["start", "meta", "chunk", "chunk", "done"])
        self.assertFalse(first_events[-1]["meta"]["cached"])
        self.assertTrue(second_events[-1]["meta"]["cached"])
        self.assertEqual(fetch_mock.call_count, 1)

    async def test_summary_stream_returns_partial_result_without_transcript(self):
        with patch.object(
            main_module,
            "fetch_transcript",
            side_effect=TranscriptError("Trascrizione non disponibile."),
        ):
            response = await self.client.post(
                "/api/summarize/stream",
                json={"url": "https://youtu.be/dQw4w9WgXcQ", "mode": "veloce"},
            )

        events = parse_ndjson(response.text)
        error = events[-1]
        self.assertEqual(error["type"], "error")
        self.assertTrue(error["partial"])
        self.assertEqual(error["video_id"], "dQw4w9WgXcQ")

    async def test_unexpected_transcript_failure_still_returns_partial_result(self):
        with patch.object(
            main_module,
            "fetch_transcript",
            side_effect=RuntimeError("errore provider transcript"),
        ):
            response = await self.client.post(
                "/api/summarize/stream",
                json={"url": "https://youtu.be/dQw4w9WgXcQ", "mode": "veloce"},
            )

        error = parse_ndjson(response.text)[-1]
        self.assertEqual(error["type"], "error")
        self.assertTrue(error["partial"])
        self.assertEqual(error["video_id"], "dQw4w9WgXcQ")

    async def test_server_rendered_partial_result_keeps_comments_tab(self):
        with patch.object(
            main_module,
            "summarize_video",
            side_effect=TranscriptError("Trascrizione non disponibile."),
        ):
            response = await self.client.post(
                "/summarize",
                data={"url": "https://youtu.be/dQw4w9WgXcQ", "mode": "veloce"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Commenti", response.text)
        self.assertIn('data-video-id="dQw4w9WgXcQ"', response.text)
        rendered_markup = response.text.split("<script>", 1)[0]
        self.assertNotIn('class="chat-section"', rendered_markup)


if __name__ == "__main__":
    unittest.main()
