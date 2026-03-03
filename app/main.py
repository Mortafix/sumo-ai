import html
import logging
import re
from os import getenv
from time import perf_counter
from typing import Literal

from app.services.cache_service import InMemoryTTLCache, make_key
from app.services.metrics_service import InMemoryMetrics
from app.services.summarizer_service import SummarizerError, summarize_text
from app.services.transcript_service import (InvalidYouTubeUrlError,
                                             TranscriptError, extract_video_id,
                                             fetch_transcript)
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

SummaryMode = Literal["one_line", "veloce", "dettagliato"]
MetaPage = Literal["index", "stats"]
SUMMARY_CACHE_TTL_SECONDS = 3600
SITE_URL = getenv("SITE_URL").rstrip("/")
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


async def summarize_video(url: str, mode: SummaryMode) -> dict:
    await metrics_service.record_request(mode)
    started_at = perf_counter()

    try:
        video_id = extract_video_id(url)
        cache_key = make_key(video_id=video_id, mode=mode)
        cache_hit = await summary_cache.get(cache_key)
        if cache_hit:
            processing_ms = elapsed_ms(started_at)
            await metrics_service.record_success(
                mode=mode,
                processing_ms=processing_ms,
                cached=True,
            )
            return {
                "summary": cache_hit["summary"],
                "meta": {
                    "video_id": video_id,
                    "language": cache_hit["language"],
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
    context: dict = {
        "request": request,
        "meta": build_meta("index", "/"),
        "url": url or "",
        "mode": selected_mode,
        "result": None,
        "error": None,
        "auto_start": has_url,
        "metrics": await metrics_service.snapshot(),
    }

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
    context: dict = {
        "request": request,
        "meta": build_meta("index", "/"),
        "url": url,
        "mode": selected_mode,
        "result": None,
        "error": None,
        "auto_start": False,
        "metrics": await metrics_service.snapshot(),
    }

    try:
        context["result"] = await summarize_video(url=url, mode=selected_mode)
        context["result"]["summary_html"] = render_summary_html(
            context["result"]["summary"]
        )
    except (TranscriptError, SummarizerError) as exc:
        context["error"] = str(exc)
    except Exception as exc:  # pragma: no cover - defensive fallback
        context["error"] = f"Unexpected error: {exc}"
    context["metrics"] = await metrics_service.snapshot()

    return templates.TemplateResponse("index.html", context)


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
    return result


@app.get("/api/metrics", response_model=MetricsResponse)
async def metrics_api():
    return await metrics_service.snapshot()
