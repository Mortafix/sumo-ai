from re import compile
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi

VIDEO_ID_RE = compile(r"^[a-zA-Z0-9_-]{11}$")


class TranscriptError(Exception):
    pass


class InvalidYouTubeUrlError(TranscriptError):
    pass


def _is_valid_video_id(value: str) -> bool:
    return bool(VIDEO_ID_RE.match(value))


def extract_video_id(url_or_id: str) -> str:
    candidate = (url_or_id or "").strip()
    if _is_valid_video_id(candidate):
        return candidate
    parsed = urlparse(candidate)
    hostname = (parsed.hostname or "").lower()
    path_parts = [part for part in parsed.path.split("/") if part]
    if "youtu.be" in hostname and path_parts:
        video_id = path_parts[0]
        if _is_valid_video_id(video_id):
            return video_id
    if "youtube.com" in hostname:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            if video_id and _is_valid_video_id(video_id):
                return video_id
        if (
            path_parts
            and path_parts[0] in {"shorts", "embed", "live"}
            and len(path_parts) > 1
        ):
            video_id = path_parts[1]
            if _is_valid_video_id(video_id):
                return video_id
    raise InvalidYouTubeUrlError("Video di YouTube non trovato.")


def fetch_transcript(url_or_id: str) -> dict:
    video_id = extract_video_id(url_or_id)
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=["it", "en"])
    except Exception as exc:
        raise TranscriptError(
            "Impossibile recuperare la trascrizione. "
            "Verifica che il video abbia sottotitoli disponibili."
        ) from exc
    if not transcript.to_raw_data():
        raise TranscriptError("Trascrizione vuota o non disponibile.")
    parts: list[str] = []
    for row in transcript.to_raw_data():
        text = row.get("text", "").strip()
        if text:
            parts.append(text)
    transcript_text = " ".join(parts).strip()
    if not transcript_text:
        raise TranscriptError("Trascrizione presente ma senza testo utile.")
    return {
        "video_id": video_id,
        "language": transcript.language or "sconosciuto",
        "text": transcript_text,
    }
