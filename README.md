# Sumo AI

Applicazione FastAPI + Jinja che:
- accetta URL YouTube
- recupera trascrizione
- genera riassunto con AI (Ollama o OpenAI)
- offre una mini-chat sul transcript (max 3 messaggi utente)
- permette download del transcript in `.txt`
- espone metriche runtime e dashboard `/stats`

## Requisiti

- Python 3.10+
- [Ollama](https://ollama.com/) in esecuzione locale
- OpenAI API key

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:1b
```

## Avvio

```bash
uvicorn app.main:app --reload
```

Apri: `http://127.0.0.1:8000`

## Query Params supportati

- `url`: link YouTube
- `mode`: `one_line`, `veloce` oppure `dettagliato`
- alias accettati lato pagina HTML: `oneline`, `one-line`, `short`, `fast`, `long`, `detailed`

Esempio:

`http://127.0.0.1:8000/?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ&mode=veloce`

Con `url` nei query params, la pagina viene renderizzata subito (form + preview + bottone) e poi avvia automaticamente il submit verso `POST /summarize` lato client.

## Variabili ambiente opzionali

- `AI_PROVIDER` valori supportati: `ollama`, `openai`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)
- `OPENAI_API_KEY` 
- `OPENAI_MODEL` 
- `SITE_URL` per canonical/meta URL
- `FOLDER` root da cui leggere i prompt in `app/static/prompts/`

## Prompt Modes

I template prompt sono file statici:
- `app/static/prompts/oneline.txt`
- `app/static/prompts/short.txt`
- `app/static/prompts/detailed.txt`

Nota: con `FOLDER=.` l'app si aspetta di essere avviata dalla root del progetto.

## API JSON

Endpoint:

`POST /api/summarize`

Request:

```json
{
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "mode": "veloce"
}
```

Valori `mode` accettati da API: `one_line`, `veloce`, `dettagliato`.

Response:

```json
{
  "summary": "....",
  "meta": {
    "video_id": "dQw4w9WgXcQ",
    "language": "it",
    "mode": "veloce",
    "cached": false,
    "processing_ms": 842.7
  }
}
```

Errori principali:
- `400`: URL YouTube non valido
- `422`: payload JSON non valido
- `503`: transcript non disponibile o errore modello locale

Nota: il contratto di `POST /api/summarize` resta invariato (nessun campo chat/transcript aggiuntivo nella risposta pubblica).

## Chat sul transcript (UI web)

Dopo la generazione del riassunto (route HTML `POST /summarize`), la pagina mostra:
- una chat contestuale al transcript del video
- limite massimo: `3` messaggi utente
- storico conversazione nella stessa sessione in-memory

Route HTML:
- `POST /chat` (form data: `chat_id`, `message`)

Comportamento:
- se la sessione chat non esiste o e scaduta, viene mostrato errore in pagina
- al quarto messaggio utente la richiesta viene rifiutata

## Download transcript `.txt` (UI web)

Route:
- `GET /transcript/{chat_id}.txt`

Comportamento:
- scarica il transcript in formato testo (`text/plain`) usato per il riassunto
- filename: `transcript-<video_id>.txt`
- restituisce `404` se la sessione non esiste o e scaduta

Endpoint metriche:

`GET /api/metrics`

Response:

```json
{
  "since_start": {
    "requests_total": 3,
    "success_total": 3,
    "failure_total": 0,
    "cache_hits_total": 1,
    "cache_misses_total": 2,
    "cache_hit_rate": 0.3333,
    "error_rate": 0.0,
    "avg_processing_ms": 421.73
  },
  "per_mode": {
    "one_line": {
      "requests_total": 2,
      "success_total": 2,
      "failure_total": 0,
      "cache_hits_total": 1,
      "cache_misses_total": 1,
      "cache_hit_rate": 0.5,
      "error_rate": 0.0,
      "avg_processing_ms": 257.1
    },
    "veloce": {
      "requests_total": 1,
      "success_total": 1,
      "failure_total": 0,
      "cache_hits_total": 0,
      "cache_misses_total": 1,
      "cache_hit_rate": 0.0,
      "error_rate": 0.0,
      "avg_processing_ms": 751.0
    },
    "dettagliato": {
      "requests_total": 0,
      "success_total": 0,
      "failure_total": 0,
      "cache_hits_total": 0,
      "cache_misses_total": 0,
      "cache_hit_rate": 0.0,
      "error_rate": 0.0,
      "avg_processing_ms": 0.0
    }
  }
}
```

Dashboard metriche HTML:

`GET /stats`

## Cache

- Tipo: in-memory (process local)
- Chiave: `video_id:mode`
- TTL: `3600` secondi (1 ora)
- Invalidazione: lazy alla lettura
- Payload cache summary: `summary`, `language`, `transcript`

Sessione chat:
- store in-memory separato
- TTL: `3600` secondi (1 ora)
- contiene `chat_id`, metadati video, `summary`, `transcript`, storico chat, contatore messaggi utente
