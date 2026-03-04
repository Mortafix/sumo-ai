import html
import json
import logging
import re
from os import getenv
from time import perf_counter
from typing import Literal

from app.services.cache_service import (InMemoryChatSessionStore,
                                        InMemoryTTLCache, make_key)
from app.services.metrics_service import InMemoryMetrics
from app.services.summarizer_service import (SummarizerError,
                                             answer_about_transcript,
                                             stream_answer_about_transcript,
                                             stream_summarize_text,
                                             summarize_text)
from app.services.transcript_service import (InvalidYouTubeUrlError,
                                             TranscriptError, extract_video_id,
                                             fetch_transcript)
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

SummaryMode = Literal["one_line", "veloce", "dettagliato"]
MetaPage = Literal["index", "stats"]
SUMMARY_CACHE_TTL_SECONDS = 3600
CHAT_MAX_USER_MESSAGES = 3
CHAT_SESSION_TTL_SECONDS = SUMMARY_CACHE_TTL_SECONDS
SITE_URL = (getenv("SITE_URL") or "").rstrip("/")
SITE_NAME = "Sumo AI"
SITE_LABEL = "sumo"
DEFAULT_DESCRIPTION = "Genera il riassunto di un video YouTube"
STATS_DESCRIPTION = "Statistiche interne e metriche operative di Sumo AI"
THEME_COLOR = "#4f46e5"
COLOR_SCHEME = "dark"

app = FastAPI(title="Sumo AI")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
summary_cache = InMemoryTTLCache()
chat_sessions = InMemoryChatSessionStore()
metrics_service = InMemoryMetrics()
logger = logging.getLogger(__name__)


class SummarizeApiRequest(BaseModel):
    url: str = Field(..., min_length=1)
    mode: SummaryMode = "veloce"


class SummarizeMeta(BaseModel):
    video_id: str
    language: str
    mode: SummaryMode
    cached: bool
    processing_ms: float


class SummarizeApiResponse(BaseModel):
    summary: str
    meta: SummarizeMeta


class ChatStreamApiRequest(BaseModel):
    chat_id: str = Field(..., min_length=1)
    chat_token: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=600)


class MetricsBucketResponse(BaseModel):
    requests_total: int
    success_total: int
    failure_total: int
    cache_hits_total: int
    cache_misses_total: int
    cache_hit_rate: float
    error_rate: float
    avg_processing_ms: float


class MetricsPerModeResponse(BaseModel):
    one_line: MetricsBucketResponse
    veloce: MetricsBucketResponse
    dettagliato: MetricsBucketResponse


class MetricsResponse(BaseModel):
    since_start: MetricsBucketResponse
    per_mode: MetricsPerModeResponse


def ndjson_line(payload: dict) -> str:
    return f"{json.dumps(payload, ensure_ascii=False)}\n"


def normalize_mode(mode: str) -> SummaryMode:
    candidate = (mode or "").strip().lower()
    if candidate in {"one_line", "one-line", "oneline"}:
        return "one_line"
    if candidate in {"dettagliato", "detailed", "long"}:
        return "dettagliato"
    if candidate in {"veloce", "fast", "short"}:
        return "veloce"
    return "veloce"


def render_summary_html(summary: str) -> str:
    def inline_format(value: str) -> str:
        escaped = html.escape(value)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(
            r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", escaped
        )
        escaped = re.sub(r"(?<!_)_(?!\s)(.+?)(?<!\s)_(?!_)", r"<em>\1</em>", escaped)
        return escaped

    lines = (summary or "").splitlines()
    chunks: list[str] = []
    list_items: list[str] = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            chunks.append(
                "<ul>" + "".join(f"<li>{item}</li>" for item in list_items) + "</ul>"
            )
            list_items = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_list()
            continue

        if line.startswith(("- ", "* ")):
            list_items.append(inline_format(line[2:].strip()))
            continue

        flush_list()
        if line.startswith("### "):
            chunks.append(f"<h3>{inline_format(line[4:].strip())}</h3>")
        elif line.startswith("## "):
            chunks.append(f"<h2>{inline_format(line[3:].strip())}</h2>")
        elif line.startswith("# "):
            chunks.append(f"<h1>{inline_format(line[2:].strip())}</h1>")
        else:
            chunks.append(f"<p>{inline_format(line)}</p>")

    flush_list()
    return "".join(chunks) if chunks else "<p></p>"


def absolute_site_url(path: str) -> str:
    normalized_path = path if path.startswith("/") else f"/{path}"
    base_url = SITE_URL or "http://127.0.0.1:8000"
    return f"{base_url}{normalized_path}"


def build_meta(page: MetaPage, path: str) -> dict[str, str]:
    if page == "stats":
        title = f"Statistiche - {SITE_NAME}"
        description = STATS_DESCRIPTION
        robots = "noindex,nofollow,noarchive"
    else:
        title = SITE_NAME
        description = DEFAULT_DESCRIPTION
        robots = "index,follow,max-image-preview:large"

    canonical_url = absolute_site_url(path)
    social_image_url = absolute_site_url("/static/meta/social-card.png")
    image_alt = f"{SITE_NAME} preview"

    return {
        "title": title,
        "description": description,
        "canonical_url": canonical_url,
        "robots": robots,
        "og_type": "website",
        "og_site_name": SITE_NAME,
        "og_locale": "it_IT",
        "og_title": title,
        "og_description": description,
        "og_url": canonical_url,
        "og_image": social_image_url,
        "og_image_alt": image_alt,
        "og_image_type": "image/png",
        "og_image_width": "1200",
        "og_image_height": "630",
        "twitter_card": "summary_large_image",
        "twitter_title": title,
        "twitter_description": description,
        "twitter_url": canonical_url,
        "twitter_image": social_image_url,
        "twitter_image_alt": image_alt,
        "application_name": SITE_NAME,
        "apple_mobile_web_app_title": SITE_LABEL,
        "theme_color": THEME_COLOR,
        "color_scheme": COLOR_SCHEME,
        "favicon_svg_url": "/static/meta/favicon.svg",
        "favicon_png_url": "/static/meta/favicon-32x32.png",
        "apple_touch_icon_url": "/static/meta/apple-touch-icon.png",
        "manifest_url": "/static/site.webmanifest",
    }


def elapsed_ms(started_at: float) -> float:
    return max((perf_counter() - started_at) * 1000.0, 0.001)


def chat_view(session: dict) -> dict:
    remaining_messages = max(
        CHAT_MAX_USER_MESSAGES - session["user_messages_count"], 0
    )
    return {
        "chat_id": session["chat_id"],
        "chat_token": session["active_form_token"],
        "history": session["history"],
        "remaining_messages": remaining_messages,
        "max_messages": CHAT_MAX_USER_MESSAGES,
        "limit_reached": remaining_messages == 0,
    }


def summary_result_from_session(session: dict) -> dict:
    summary_text = session["summary"]
    return {
        "summary": summary_text,
        "summary_html": render_summary_html(summary_text),
        "transcript": session["transcript"],
        "meta": {
            "video_id": session["video_id"],
            "language": session["language"],
            "mode": normalize_mode(session["mode"]),
            "cached": True,
            "processing_ms": session["processing_ms"],
        },
    }


async def base_index_context(
    request: Request,
    *,
    url: str = "",
    mode: SummaryMode = "veloce",
    auto_start: bool = False,
    result: dict | None = None,
    error: str | None = None,
    chat: dict | None = None,
    chat_error: str | None = None,
) -> dict:
    return {
        "request": request,
        "meta": build_meta("index", "/"),
        "url": url,
        "mode": mode,
        "result": result,
        "error": error,
        "chat": chat,
        "chat_error": chat_error,
        "auto_start": auto_start,
        "metrics": await metrics_service.snapshot(),
    }


async def summarize_video(url: str, mode: SummaryMode) -> dict:
    await metrics_service.record_request(mode)
    started_at = perf_counter()

    try:
        video_id = extract_video_id(url)
        cache_key = make_key(video_id=video_id, mode=mode)
        cache_hit = await summary_cache.get(cache_key)
        if cache_hit:
            cached_summary = cache_hit.get("summary", "")
            cached_language = cache_hit.get("language", "sconosciuto")
            cached_transcript = (cache_hit.get("transcript") or "").strip()
            if not cached_transcript:
                try:
                    transcript_data = fetch_transcript(video_id)
                    cached_transcript = transcript_data["text"]
                    await summary_cache.set(
                        cache_key,
                        {
                            "summary": cached_summary,
                            "language": cached_language,
                            "transcript": cached_transcript,
                        },
                        ttl_seconds=SUMMARY_CACHE_TTL_SECONDS,
                    )
                except TranscriptError as exc:
                    logger.warning(
                        "cache_hit_missing_transcript video_id=%s mode=%s error=%s",
                        video_id,
                        mode,
                        exc,
                    )
            processing_ms = elapsed_ms(started_at)
            await metrics_service.record_success(
                mode=mode,
                processing_ms=processing_ms,
                cached=True,
            )
            return {
                "summary": cached_summary,
                "transcript": cached_transcript,
                "meta": {
                    "video_id": video_id,
                    "language": cached_language,
                    "mode": mode,
                    "cached": True,
                    "processing_ms": processing_ms,
                },
            }

        transcript_data = fetch_transcript(video_id)
        summary = await summarize_text(transcript_data["text"], mode=mode)
        await summary_cache.set(
            cache_key,
            {
                "summary": summary,
                "language": transcript_data["language"],
                "transcript": transcript_data["text"],
            },
            ttl_seconds=SUMMARY_CACHE_TTL_SECONDS,
        )
        processing_ms = elapsed_ms(started_at)
        await metrics_service.record_success(
            mode=mode,
            processing_ms=processing_ms,
            cached=False,
        )
        return {
            "summary": summary,
            "transcript": transcript_data["text"],
            "meta": {
                "video_id": video_id,
                "language": transcript_data["language"],
                "mode": mode,
                "cached": False,
                "processing_ms": processing_ms,
            },
        }
    except Exception:
        await metrics_service.record_failure(
            mode=mode, processing_ms=elapsed_ms(started_at)
        )
        raise


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    url: str | None = None,
    mode: str = "veloce",
):
    selected_mode = normalize_mode(mode)
    has_url = bool((url or "").strip())
    context = await base_index_context(
        request,
        url=url or "",
        mode=selected_mode,
        auto_start=has_url,
    )
    return templates.TemplateResponse("index.html", context)


@app.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request):
    context: dict = {
        "request": request,
        "meta": build_meta("stats", "/stats"),
        "metrics": await metrics_service.snapshot(),
    }
    return templates.TemplateResponse("stats.html", context)


@app.post("/summarize", response_class=HTMLResponse)
async def summarize(
    request: Request,
    url: str = Form(...),
    mode: str = Form("veloce"),
):
    selected_mode = normalize_mode(mode)
    context = await base_index_context(request, url=url, mode=selected_mode)

    try:
        context["result"] = await summarize_video(url=url, mode=selected_mode)
        context["result"]["summary_html"] = render_summary_html(
            context["result"]["summary"]
        )
        chat_session = await chat_sessions.create(
            {
                "video_id": context["result"]["meta"]["video_id"],
                "mode": context["result"]["meta"]["mode"],
                "language": context["result"]["meta"]["language"],
                "summary": context["result"]["summary"],
                "transcript": context["result"]["transcript"],
                "processing_ms": context["result"]["meta"]["processing_ms"],
            },
            ttl_seconds=CHAT_SESSION_TTL_SECONDS,
        )
        context["chat"] = chat_view(chat_session)
    except (TranscriptError, SummarizerError) as exc:
        context["error"] = str(exc)
    except Exception as exc:  # pragma: no cover - defensive fallback
        context["error"] = f"Unexpected error: {exc}"
    context["metrics"] = await metrics_service.snapshot()

    return templates.TemplateResponse("index.html", context)


@app.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    chat_id: str = Form(...),
    chat_token: str = Form(...),
    message: str = Form(...),
):
    session = await chat_sessions.get(chat_id)
    if session is None:
        context = await base_index_context(
            request,
            error="Sessione chat non valida o scaduta.",
        )
        return templates.TemplateResponse("index.html", context)

    cleaned_message = (message or "").strip()
    result = summary_result_from_session(session)
    context = await base_index_context(
        request,
        url=f"https://www.youtube.com/watch?v={session['video_id']}",
        mode=normalize_mode(session["mode"]),
        result=result,
        chat=chat_view(session),
    )

    if not cleaned_message:
        context["chat_error"] = "Inserisci un messaggio prima di inviare."
        return templates.TemplateResponse("index.html", context)

    if session["user_messages_count"] >= CHAT_MAX_USER_MESSAGES:
        context["chat_error"] = (
            f"Hai raggiunto il limite di {CHAT_MAX_USER_MESSAGES} messaggi."
        )
        return templates.TemplateResponse("index.html", context)

    try:
        verified_session = await chat_sessions.verify_form_token(chat_id, chat_token)
        context["chat"] = chat_view(verified_session)
        assistant_answer = await answer_about_transcript(
            transcript=verified_session["transcript"],
            history=verified_session["history"],
            question=cleaned_message,
        )
        updated_session = await chat_sessions.record_exchange(
            chat_id=chat_id,
            user_message=cleaned_message,
            assistant_message=assistant_answer,
            max_user_messages=CHAT_MAX_USER_MESSAGES,
        )
        context["result"] = summary_result_from_session(updated_session)
        context["chat"] = chat_view(updated_session)
    except SummarizerError as exc:
        context["chat_error"] = str(exc)
    except ValueError as exc:
        if "Limite" in str(exc):
            context["chat_error"] = (
                f"Hai raggiunto il limite di {CHAT_MAX_USER_MESSAGES} messaggi."
            )
        else:
            context["chat_error"] = "Messaggio gia inviato o non valido. Riprova."
    except KeyError:
        context = await base_index_context(
            request,
            error="Sessione chat non valida o scaduta.",
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        context["chat_error"] = f"Unexpected error: {exc}"
    context["metrics"] = await metrics_service.snapshot()
    return templates.TemplateResponse("index.html", context)


@app.get("/transcript/{chat_id}.txt")
async def transcript_download(chat_id: str):
    session = await chat_sessions.get(chat_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail="Transcript non disponibile o sessione scaduta."
        )
    filename = f'transcript-{session["video_id"]}.txt'
    return PlainTextResponse(
        content=session["transcript"],
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/summarize", response_model=SummarizeApiResponse)
async def summarize_api(payload: SummarizeApiRequest):
    started_at = perf_counter()
    try:
        result = await summarize_video(url=payload.url, mode=payload.mode)
    except InvalidYouTubeUrlError as exc:
        logger.warning(
            "api_summarize invalid_url mode=%s processing_ms=%.2f error=%s",
            payload.mode,
            elapsed_ms(started_at),
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (TranscriptError, SummarizerError) as exc:
        logger.error(
            "api_summarize service_error mode=%s processing_ms=%.2f error=%s",
            payload.mode,
            elapsed_ms(started_at),
            exc,
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception(
            "api_summarize unexpected_error mode=%s processing_ms=%.2f",
            payload.mode,
            elapsed_ms(started_at),
        )
        raise HTTPException(
            status_code=500, detail="Unexpected internal error."
        ) from exc

    logger.info(
        "api_summarize success video_id=%s mode=%s cached=%s processing_ms=%.2f",
        result["meta"]["video_id"],
        result["meta"]["mode"],
        result["meta"]["cached"],
        result["meta"]["processing_ms"],
    )
    return {"summary": result["summary"], "meta": result["meta"]}


@app.post("/api/summarize/stream")
async def summarize_api_stream(payload: SummarizeApiRequest):
    selected_mode = normalize_mode(payload.mode)

    async def stream():
        started_at = perf_counter()
        await metrics_service.record_request(selected_mode)
        yield ndjson_line({"type": "start"})
        try:
            video_id = extract_video_id(payload.url)
            cache_key = make_key(video_id=video_id, mode=selected_mode)
            cache_hit = await summary_cache.get(cache_key)

            if cache_hit:
                summary_text = (cache_hit.get("summary") or "").strip()
                language = cache_hit.get("language", "sconosciuto")
                transcript = (cache_hit.get("transcript") or "").strip()
                if not transcript:
                    try:
                        transcript_data = fetch_transcript(video_id)
                        transcript = transcript_data["text"]
                        await summary_cache.set(
                            cache_key,
                            {
                                "summary": summary_text,
                                "language": language,
                                "transcript": transcript,
                            },
                            ttl_seconds=SUMMARY_CACHE_TTL_SECONDS,
                        )
                    except TranscriptError as exc:
                        logger.warning(
                            "api_summarize_stream cache_hit_missing_transcript video_id=%s mode=%s error=%s",
                            video_id,
                            selected_mode,
                            exc,
                        )
                processing_ms = elapsed_ms(started_at)
                meta = {
                    "video_id": video_id,
                    "language": language,
                    "mode": selected_mode,
                    "cached": True,
                    "processing_ms": processing_ms,
                }
                await metrics_service.record_success(
                    mode=selected_mode,
                    processing_ms=processing_ms,
                    cached=True,
                )
                chat_session = await chat_sessions.create(
                    {
                        "video_id": video_id,
                        "mode": selected_mode,
                        "language": language,
                        "summary": summary_text,
                        "transcript": transcript,
                        "processing_ms": processing_ms,
                    },
                    ttl_seconds=CHAT_SESSION_TTL_SECONDS,
                )
                yield ndjson_line(
                    {
                        "type": "meta",
                        "video_id": video_id,
                        "language": language,
                        "mode": selected_mode,
                        "cached": True,
                    }
                )
                if summary_text:
                    yield ndjson_line({"type": "chunk", "text": summary_text})
                yield ndjson_line(
                    {
                        "type": "done",
                        "summary": summary_text,
                        "summary_html": render_summary_html(summary_text),
                        "meta": meta,
                        "chat": chat_view(chat_session),
                    }
                )
                return

            transcript_data = fetch_transcript(video_id)
            yield ndjson_line(
                {
                    "type": "meta",
                    "video_id": video_id,
                    "language": transcript_data["language"],
                    "mode": selected_mode,
                    "cached": False,
                }
            )

            chunks: list[str] = []
            async for chunk in stream_summarize_text(
                transcript_data["text"], mode=selected_mode
            ):
                chunks.append(chunk)
                yield ndjson_line({"type": "chunk", "text": chunk})

            summary_text = "".join(chunks).strip()
            if not summary_text:
                raise SummarizerError("Il modello AI non ha restituito alcun riassunto.")

            await summary_cache.set(
                cache_key,
                {
                    "summary": summary_text,
                    "language": transcript_data["language"],
                    "transcript": transcript_data["text"],
                },
                ttl_seconds=SUMMARY_CACHE_TTL_SECONDS,
            )
            processing_ms = elapsed_ms(started_at)
            meta = {
                "video_id": video_id,
                "language": transcript_data["language"],
                "mode": selected_mode,
                "cached": False,
                "processing_ms": processing_ms,
            }
            await metrics_service.record_success(
                mode=selected_mode,
                processing_ms=processing_ms,
                cached=False,
            )
            chat_session = await chat_sessions.create(
                {
                    "video_id": video_id,
                    "mode": selected_mode,
                    "language": transcript_data["language"],
                    "summary": summary_text,
                    "transcript": transcript_data["text"],
                    "processing_ms": processing_ms,
                },
                ttl_seconds=CHAT_SESSION_TTL_SECONDS,
            )
            yield ndjson_line(
                {
                    "type": "done",
                    "summary": summary_text,
                    "summary_html": render_summary_html(summary_text),
                    "meta": meta,
                    "chat": chat_view(chat_session),
                }
            )
        except InvalidYouTubeUrlError as exc:
            await metrics_service.record_failure(
                mode=selected_mode, processing_ms=elapsed_ms(started_at)
            )
            logger.warning(
                "api_summarize_stream invalid_url mode=%s processing_ms=%.2f error=%s",
                selected_mode,
                elapsed_ms(started_at),
                exc,
            )
            yield ndjson_line(
                {
                    "type": "error",
                    "detail": str(exc),
                    "status": 400,
                }
            )
        except (TranscriptError, SummarizerError) as exc:
            await metrics_service.record_failure(
                mode=selected_mode, processing_ms=elapsed_ms(started_at)
            )
            logger.error(
                "api_summarize_stream service_error mode=%s processing_ms=%.2f error=%s",
                selected_mode,
                elapsed_ms(started_at),
                exc,
            )
            yield ndjson_line(
                {
                    "type": "error",
                    "detail": str(exc),
                    "status": 503,
                }
            )
        except Exception:
            await metrics_service.record_failure(
                mode=selected_mode, processing_ms=elapsed_ms(started_at)
            )
            logger.exception(
                "api_summarize_stream unexpected_error mode=%s processing_ms=%.2f",
                selected_mode,
                elapsed_ms(started_at),
            )
            yield ndjson_line(
                {
                    "type": "error",
                    "detail": "Unexpected internal error.",
                    "status": 500,
                }
            )

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/chat/stream")
async def chat_stream_api(payload: ChatStreamApiRequest):
    async def stream():
        cleaned_message = (payload.message or "").strip()
        yield ndjson_line({"type": "start"})

        if not cleaned_message:
            yield ndjson_line(
                {
                    "type": "error",
                    "detail": "Inserisci un messaggio prima di inviare.",
                    "status": 400,
                }
            )
            return

        session = await chat_sessions.get(payload.chat_id)
        if session is None:
            yield ndjson_line(
                {
                    "type": "error",
                    "detail": "Sessione chat non valida o scaduta.",
                    "status": 404,
                }
            )
            return

        if session["user_messages_count"] >= CHAT_MAX_USER_MESSAGES:
            yield ndjson_line(
                {
                    "type": "error",
                    "detail": f"Hai raggiunto il limite di {CHAT_MAX_USER_MESSAGES} messaggi.",
                    "status": 429,
                    "chat": chat_view(session),
                }
            )
            return

        try:
            verified_session = await chat_sessions.verify_form_token(
                payload.chat_id, payload.chat_token
            )
        except ValueError:
            yield ndjson_line(
                {
                    "type": "error",
                    "detail": "Messaggio gia inviato o non valido. Riprova.",
                    "status": 409,
                }
            )
            return
        except KeyError:
            yield ndjson_line(
                {
                    "type": "error",
                    "detail": "Sessione chat non valida o scaduta.",
                    "status": 404,
                }
            )
            return

        yield ndjson_line({"type": "ack", "chat": chat_view(verified_session)})

        chunks: list[str] = []
        try:
            async for chunk in stream_answer_about_transcript(
                transcript=verified_session["transcript"],
                history=verified_session["history"],
                question=cleaned_message,
            ):
                chunks.append(chunk)
                yield ndjson_line({"type": "chunk", "text": chunk})

            assistant_answer = "".join(chunks).strip()
            if not assistant_answer:
                raise SummarizerError(
                    "Il modello AI non ha restituito alcuna risposta utile."
                )

            updated_session = await chat_sessions.record_exchange(
                chat_id=payload.chat_id,
                user_message=cleaned_message,
                assistant_message=assistant_answer,
                max_user_messages=CHAT_MAX_USER_MESSAGES,
            )
            yield ndjson_line(
                {
                    "type": "done",
                    "answer": assistant_answer,
                    "chat": chat_view(updated_session),
                }
            )
        except SummarizerError as exc:
            yield ndjson_line(
                {
                    "type": "error",
                    "detail": str(exc),
                    "status": 503,
                    "chat": chat_view(verified_session),
                }
            )
        except ValueError as exc:
            detail = (
                f"Hai raggiunto il limite di {CHAT_MAX_USER_MESSAGES} messaggi."
                if "Limite" in str(exc)
                else "Messaggio gia inviato o non valido. Riprova."
            )
            yield ndjson_line(
                {
                    "type": "error",
                    "detail": detail,
                    "status": 429 if "Limite" in str(exc) else 409,
                }
            )
        except KeyError:
            yield ndjson_line(
                {
                    "type": "error",
                    "detail": "Sessione chat non valida o scaduta.",
                    "status": 404,
                }
            )
        except Exception:
            logger.exception("api_chat_stream unexpected_error chat_id=%s", payload.chat_id)
            yield ndjson_line(
                {
                    "type": "error",
                    "detail": "Unexpected internal error.",
                    "status": 500,
                    "chat": chat_view(verified_session),
                }
            )

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/metrics", response_model=MetricsResponse)
async def metrics_api():
    return await metrics_service.snapshot()
