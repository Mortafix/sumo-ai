from os import getenv, path

from dotenv import load_dotenv
from httpx import AsyncClient

load_dotenv()
OLLAMA_BASE_URL = getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = getenv("OLLAMA_MODEL")
OPENAI_BASE_URL = getenv("OPENAI_BASE_URL").rstrip("/")
OPENAI_API_KEY = getenv("OPENAI_API_KEY")
OPENAI_MODEL = getenv("OPENAI_MODEL")
AI_PROVIDER = getenv("AI_PROVIDER").strip().lower()
PROMPTS_FOLDER = getenv("FOLDER")


class SummarizerError(Exception):
    pass


def _prompt_template_name(mode: str) -> str:
    template_name = {
        "one_line": "oneline",
        "dettagliato": "detailed",
        "veloce": "short",
    }.get(mode)
    if not template_name:
        raise SummarizerError(f"Modalita riassunto non supportata: {mode}")
    return template_name


def _read_prompt_template(mode: str) -> str:
    template_name = _prompt_template_name(mode)
    prompt_path = path.join(
        PROMPTS_FOLDER, "app", "static", "prompts", f"{template_name}.txt"
    )
    try:
        with open(prompt_path, encoding="utf-8") as file_handle:
            return file_handle.read().strip()
    except FileNotFoundError as exc:
        raise SummarizerError(
            f"Template prompt non trovato: {prompt_path}. "
            "Controlla la variabile FOLDER o la struttura dei file."
        ) from exc


def _build_prompt(text: str, mode: str) -> str:
    summarize_style = _read_prompt_template(mode)
    return (
        "Sei un assistente che riassume trascrizioni video.\n"
        "Usa solo informazioni presenti nella trascrizione.\n"
        "Non inventare dettagli, nomi, date o numeri.\n"
        f"{summarize_style}\n"
        "Mantieni un linguaggio chiaro e professionale.\n\n"
        f"Trascrizione:\n{text}"
    )


def _extract_openai_text(payload: dict) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output") or []
    for item in output:
        if not isinstance(item, dict):
            continue
        for content_item in item.get("content") or []:
            if (
                isinstance(content_item, dict)
                and content_item.get("type") == "output_text"
                and isinstance(content_item.get("text"), str)
                and content_item["text"].strip()
            ):
                return content_item["text"].strip()
    return ""


async def _summarize_with_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    try:
        async with AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate", json=payload
            )
            response.raise_for_status()
    except Exception as exc:
        raise SummarizerError(
            "Modello AI locale non trovato o non raggiungibile."
        ) from exc

    data = response.json()
    summary = (data.get("response") or "").strip()
    if not summary:
        raise SummarizerError("Il modello AI locale non ha restituito alcun riassunto.")
    return summary


async def _summarize_with_openai(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise SummarizerError("OPENAI_API_KEY non impostata.")
    payload = {"model": OPENAI_MODEL, "input": prompt}
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{OPENAI_BASE_URL}/responses",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
    except Exception as exc:
        raise exc
        raise SummarizerError(
            "API OpenAI non raggiungibile o configurazione non valida."
        ) from exc
    data = response.json()
    summary = _extract_openai_text(data)
    if not summary:
        raise SummarizerError("OpenAI non ha restituito alcun testo di riassunto.")
    return summary


async def summarize_text(text: str, mode: str) -> str:
    prompt = _build_prompt(text, mode)
    if AI_PROVIDER not in {"ollama", "openai"}:
        raise SummarizerError(
            f"AI_PROVIDER non supportato: {AI_PROVIDER}. Usa 'ollama' o 'openai'."
        )
    if AI_PROVIDER == "openai":
        return await _summarize_with_openai(prompt)
    return await _summarize_with_ollama(prompt)
