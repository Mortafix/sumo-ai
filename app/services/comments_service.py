from __future__ import annotations

from dataclasses import dataclass
from os import getenv
from typing import Literal

from dotenv import load_dotenv
from httpx import AsyncClient, HTTPError, TimeoutException

from app.services.transcript_service import extract_video_id

load_dotenv()

YOUTUBE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
COMMENTS_PER_GROUP = 150
COMMENT_TEXT_MAX_CHARS = 1_000
COMMENTS_INPUT_MAX_CHARS = 50_000


class CommentsError(Exception):
    def __init__(self, message: str, *, code: str, status: int = 503) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class CommentsNotConfiguredError(CommentsError):
    def __init__(self) -> None:
        super().__init__(
            "Analisi commenti non configurata: imposta YOUTUBE_API_KEY.",
            code="not_configured",
            status=503,
        )


class CommentsDisabledError(CommentsError):
    def __init__(self) -> None:
        super().__init__(
            "I commenti sono disabilitati per questo video.",
            code="comments_disabled",
            status=409,
        )


class NoCommentsError(CommentsError):
    def __init__(self) -> None:
        super().__init__(
            "Non ci sono commenti pubblici da analizzare per questo video.",
            code="no_comments",
            status=404,
        )


class VideoNotFoundError(CommentsError):
    def __init__(self) -> None:
        super().__init__(
            "Video non disponibile o non accessibile tramite YouTube.",
            code="video_not_found",
            status=404,
        )


@dataclass(frozen=True)
class PublicComment:
    comment_id: str
    text: str
    like_count: int
    published_at: str
    source: Literal["relevance", "time"]


@dataclass(frozen=True)
class CommentSample:
    comments: list[PublicComment]
    relevant_count: int
    recent_count: int

    @property
    def found_count(self) -> int:
        return len(self.comments)


def _youtube_error_reason(payload: dict) -> str:
    error = payload.get("error")
    if not isinstance(error, dict):
        return ""
    for item in error.get("errors") or []:
        if isinstance(item, dict) and isinstance(item.get("reason"), str):
            return item["reason"]
    return ""


def _raise_youtube_error(payload: dict) -> None:
    reason = _youtube_error_reason(payload)
    if reason == "commentsDisabled":
        raise CommentsDisabledError()
    if reason == "videoNotFound":
        raise VideoNotFoundError()
    if reason in {"quotaExceeded", "dailyLimitExceeded"}:
        raise CommentsError(
            "Quota YouTube esaurita. Riprova più tardi.",
            code="youtube_quota_exceeded",
            status=503,
        )
    if reason in {"keyInvalid", "accessNotConfigured", "ipRefererBlocked"}:
        raise CommentsError(
            "Chiave YouTube non valida o YouTube Data API non abilitata.",
            code="youtube_configuration_error",
            status=503,
        )
    raise CommentsError(
        "Impossibile recuperare i commenti da YouTube.",
        code="youtube_api_error",
        status=503,
    )


def _parse_comments(
    payload: dict,
    *,
    source: Literal["relevance", "time"],
) -> list[PublicComment]:
    parsed: list[PublicComment] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        thread_snippet = item.get("snippet") or {}
        top_level = thread_snippet.get("topLevelComment") or {}
        snippet = top_level.get("snippet") or {}
        comment_id = str(top_level.get("id") or item.get("id") or "").strip()
        text = str(snippet.get("textDisplay") or "").strip()
        if not comment_id or not text:
            continue
        try:
            like_count = max(int(snippet.get("likeCount") or 0), 0)
        except (TypeError, ValueError):
            like_count = 0
        parsed.append(
            PublicComment(
                comment_id=comment_id,
                text=text,
                like_count=like_count,
                published_at=str(snippet.get("publishedAt") or "").strip(),
                source=source,
            )
        )
    return parsed


async def _fetch_group(
    client: AsyncClient,
    *,
    video_id: str,
    api_key: str,
    order: Literal["relevance", "time"],
    target_count: int,
    excluded_ids: set[str],
    max_pages: int,
) -> list[PublicComment]:
    comments: list[PublicComment] = []
    seen_ids = set(excluded_ids)
    page_token: str | None = None

    for _ in range(max_pages):
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "order": order,
            "textFormat": "plainText",
            "key": api_key,
            "fields": (
                "items(id,snippet(topLevelComment(id,snippet("
                "textDisplay,likeCount,publishedAt)))),nextPageToken"
            ),
        }
        if page_token:
            params["pageToken"] = page_token

        response = await client.get(YOUTUBE_COMMENTS_URL, params=params)
        try:
            payload = response.json()
        except ValueError as exc:
            raise CommentsError(
                "YouTube ha restituito una risposta non valida.",
                code="youtube_api_error",
                status=503,
            ) from exc
        if not isinstance(payload, dict):
            raise CommentsError(
                "YouTube ha restituito una risposta non valida.",
                code="youtube_api_error",
                status=503,
            )
        if response.is_error:
            _raise_youtube_error(payload)

        for comment in _parse_comments(payload, source=order):
            if comment.comment_id in seen_ids:
                continue
            seen_ids.add(comment.comment_id)
            comments.append(comment)
            if len(comments) >= target_count:
                return comments

        next_page_token = payload.get("nextPageToken")
        if not isinstance(next_page_token, str) or not next_page_token:
            break
        page_token = next_page_token

    return comments


async def fetch_comment_sample(
    video_id: str,
    *,
    api_key: str | None = None,
    client: AsyncClient | None = None,
) -> CommentSample:
    normalized_video_id = extract_video_id(video_id)
    resolved_api_key = (
        api_key if api_key is not None else (getenv("YOUTUBE_API_KEY") or "")
    ).strip()
    if not resolved_api_key:
        raise CommentsNotConfiguredError()

    owns_client = client is None
    active_client = client or AsyncClient(timeout=30.0)
    try:
        relevant = await _fetch_group(
            active_client,
            video_id=normalized_video_id,
            api_key=resolved_api_key,
            order="relevance",
            target_count=COMMENTS_PER_GROUP,
            excluded_ids=set(),
            max_pages=2,
        )
        recent = await _fetch_group(
            active_client,
            video_id=normalized_video_id,
            api_key=resolved_api_key,
            order="time",
            target_count=COMMENTS_PER_GROUP,
            excluded_ids={comment.comment_id for comment in relevant},
            max_pages=4,
        )
    except CommentsError:
        raise
    except TimeoutException as exc:
        raise CommentsError(
            "YouTube non ha risposto in tempo. Riprova.",
            code="youtube_timeout",
            status=503,
        ) from exc
    except HTTPError as exc:
        raise CommentsError(
            "Impossibile raggiungere YouTube. Riprova.",
            code="youtube_api_error",
            status=503,
        ) from exc
    finally:
        if owns_client:
            await active_client.aclose()

    comments = [*relevant, *recent]
    if not comments:
        raise NoCommentsError()
    return CommentSample(
        comments=comments,
        relevant_count=len(relevant),
        recent_count=len(recent),
    )


def prepare_comments_for_analysis(
    sample: CommentSample,
    *,
    per_comment_limit: int = COMMENT_TEXT_MAX_CHARS,
    total_limit: int = COMMENTS_INPUT_MAX_CHARS,
) -> tuple[str, int]:
    relevant = [comment for comment in sample.comments if comment.source == "relevance"]
    recent = [comment for comment in sample.comments if comment.source == "time"]
    balanced: list[PublicComment] = []
    max_group_length = max(len(relevant), len(recent), 0)
    for index in range(max_group_length):
        if index < len(relevant):
            balanced.append(relevant[index])
        if index < len(recent):
            balanced.append(recent[index])

    lines: list[str] = []
    used_chars = 0
    for comment in balanced:
        text = " ".join(comment.text.split()).strip()[:per_comment_limit]
        if not text:
            continue
        source_label = "rilevante" if comment.source == "relevance" else "recente"
        date_label = comment.published_at[:10] or "data sconosciuta"
        line = (
            f"[{source_label}; mi piace: {comment.like_count}; data: {date_label}] "
            f"{text}"
        )
        extra_chars = len(line) + (1 if lines else 0)
        if used_chars + extra_chars > total_limit:
            continue
        lines.append(line)
        used_chars += extra_chars

    if not lines:
        raise NoCommentsError()
    return "\n".join(lines), len(lines)
