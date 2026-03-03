from os import getenv, path
from typing import Sequence

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


def _build_chat_prompt(
    transcript: str, history: Sequence[dict[str, str]], question: str
) -> str:
    history_lines: list[str] = []
    for item in history:
        role = (item.get("role") or "").strip().lower()
        content = (item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        label = "Utente" if role == "user" else "Assistente"
        history_lines.append(f"{label}: {content}")

    history_text = "\n".join(history_lines) if history_lines else "Nessuno."
    return (
        "Sei un assistente che risponde a domande su una trascrizione video.\n"
        "Usa solo informazioni presenti nella trascrizione.\n"
        "Se l'informazione non e presente, rispondi chiaramente che non e disponibile.\n"
        "Rispondi in italiano in modo chiaro e conciso.\n\n"
        f"Cronologia chat:\n{history_text}\n\n"
        f"Domanda utente:\n{question}\n\n"
        f"Trascrizione:\n{transcript}"
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


async def _generate_with_ollama(prompt: str, empty_text_error: str) -> str:
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
    generated_text = (data.get("response") or "").strip()
    if not generated_text:
        raise SummarizerError(empty_text_error)
    return generated_text


async def _generate_with_openai(prompt: str, empty_text_error: str) -> str:
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
        raise SummarizerError(
            "API OpenAI non raggiungibile o configurazione non valida."
        ) from exc
    data = response.json()
    generated_text = _extract_openai_text(data)
    if not generated_text:
        raise SummarizerError(empty_text_error)
    return generated_text


async def _generate_text(prompt: str, empty_text_error: str) -> str:
    if AI_PROVIDER not in {"ollama", "openai"}:
        raise SummarizerError(
            f"AI_PROVIDER non supportato: {AI_PROVIDER}. Usa 'ollama' o 'openai'."
        )
    if AI_PROVIDER == "openai":
        return await _generate_with_openai(prompt, empty_text_error)
    return await _generate_with_ollama(prompt, empty_text_error)


async def summarize_text(text: str, mode: str) -> str:
    prompt = _build_prompt(text, mode)
    return await _generate_text(
        prompt=prompt,
        empty_text_error="Il modello AI non ha restituito alcun riassunto.",
    )


async def answer_about_transcript(
    transcript: str, history: Sequence[dict[str, str]], question: str
) -> str:
    if not (transcript or "").strip():
        return "Non ho un transcript disponibile, quindi non posso rispondere in modo affidabile."
    prompt = _build_chat_prompt(
        transcript=transcript,
        history=history,
        question=question,
    )
    return await _generate_text(
        prompt=prompt,
        empty_text_error="Il modello AI non ha restituito alcuna risposta utile.",
    )
