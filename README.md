# Sumo AI

Applicazione FastAPI + Jinja che:
- accetta URL YouTube
- recupera trascrizione
- genera riassunto con AI (Ollama o OpenAI)
- offre una mini-chat sul transcript (max 3 messaggi utente)
- permette download del transcript in `.txt`
- analizza un campione bilanciato dei commenti pubblici YouTube
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

Con `url` nei query params, la pagina viene renderizzata subito (form + preview + bottone) e poi avvia automaticamente il submit lato client in modalita streaming.

## Variabili ambiente opzionali

- `AI_PROVIDER` valori supportati: `ollama`, `openai`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `OPENAI_BASE_URL` (default: `https://api.openai.com/v1`)
- `OPENAI_API_KEY` 
- `OPENAI_MODEL` 
- `YOUTUBE_API_KEY` chiave server-side per leggere i commenti tramite YouTube Data API v3
- `SITE_URL` per canonical/meta URL
- `FOLDER` root da cui leggere i prompt in `app/static/prompts/`

## Prompt Modes

I template prompt sono file statici:
- `app/static/prompts/oneline.txt`
- `app/static/prompts/short.txt`
- `app/static/prompts/detailed.txt`
- `app/static/prompts/comments.txt` (analisi commenti, con placeholder
  `{comments_text}`, `{found_count}` e `{analyzed_count}`)

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

## API Streaming (NDJSON)

Endpoint:

- `POST /api/summarize/stream`
- `POST /api/chat/stream`

Formato risposta: `application/x-ndjson` (una riga JSON per evento).

Eventi riassunto (`/api/summarize/stream`):

- `start`: inizio elaborazione
- `meta`: metadati iniziali (`video_id`, `language`, `mode`, `cached`)
- `chunk`: pezzo di testo del riassunto
- `done`: risultato finale (`summary`, `summary_html`, `transcript`, `meta`, `chat`)
- `error`: errore (`detail`, `status`)

Eventi chat (`/api/chat/stream`):

- `start`: inizio elaborazione
- `ack`: token chat aggiornato (`chat`)
- `chunk`: pezzo di testo della risposta assistente
- `done`: risposta finale (`answer`, `chat`)
- `error`: errore (`detail`, `status`, opzionale `chat`)

## Analisi commenti YouTube

Endpoint streaming:

`POST /api/comments/analyze/stream`

Request:

```json
{
  "video_id": "dQw4w9WgXcQ"
}
```

La risposta usa `application/x-ndjson` con eventi `start`, `meta`, `chunk`,
`done` ed `error`. Gli eventi finali includono il numero di commenti trovati e
analizzati, la composizione del campione rilevanti/recenti e lo stato della cache.

L'analisi parte quando viene aperto il tab `Commenti`, usa fino a 150 commenti
rilevanti e 150 recenti e viene conservata in-memory per un'ora. Sono analizzati
solo i commenti principali pubblicamente accessibili, non le risposte. Se la
trascrizione non è disponibile, il tab resta comunque utilizzabile.

## Chat sul transcript (UI web)

Dopo la generazione del riassunto, la pagina mostra:
- due tab risultato: `Riassunto` e `Trascrizione`
- nel tab `Trascrizione`, il transcript grezzo con copia e download `.txt`
- una chat contestuale al transcript del video
- limite massimo: `3` messaggi utente
- storico conversazione nella stessa sessione in-memory

Route HTML:
- `POST /chat` (form data: `chat_id`, `chat_token`, `message`)

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

## Estensione locale Chrome/Brave/Firefox

La cartella `extension/` contiene l'estensione privata **Sumo for YouTube**.
Sulle pagine YouTube aggiunge:

- un pulsante Sumo sulle miniature, visibile al passaggio del mouse o tramite
  focus da tastiera
- un pulsante Sumo stabile nella barra azioni della pagina video
- l'apertura del riassunto in una nuova scheda su `https://sumo.moris.dev`

L'estensione invia soltanto l'URL canonico del video. Non imposta la modalità,
quindi Sumo usa il proprio default (`veloce`).

Lo stesso pacchetto WebExtensions Manifest V3 funziona su Chrome, Brave e
Firefox 140 o successivo. Firefox mostra in fase di installazione che
l'estensione trasmette l'URL del video scelto a Sumo; non raccoglie la cronologia
in background.

### Installazione su Chrome o Brave

1. Apri `chrome://extensions` in Chrome oppure Brave.
2. Attiva **Modalità sviluppatore**.
3. Seleziona **Carica estensione non pacchettizzata**.
4. Scegli la cartella `extension/` di questo repository.

Dopo una modifica ai file dell'estensione, usa il pulsante **Ricarica** nella
scheda delle estensioni e aggiorna le pagine YouTube già aperte.

### Installazione locale su Firefox

1. Apri `about:debugging#/runtime/this-firefox` in Firefox.
2. Seleziona **Carica componente aggiuntivo temporaneo**.
3. Apri la cartella `extension/` di questo repository.
4. Seleziona il file `manifest.json`.
5. Apri o ricarica una pagina YouTube.

L'installazione temporanea resta attiva fino al riavvio di Firefox. Dopo una
modifica, torna in `about:debugging`, premi **Ricarica** sulla scheda di Sumo e
ricarica anche YouTube.

Firefox stabile richiede che i componenti aggiuntivi persistenti siano firmati
da Mozilla. Per uso privato senza firma si puo usare l'installazione temporanea;
Firefox Developer Edition, Nightly ed ESR permettono inoltre l'installazione di
pacchetti non firmati disattivando `xpinstall.signatures.required` in
`about:config`.

### Installazione permanente e privata su Firefox stabile

Per non dover ricaricare l'estensione dopo ogni riavvio, falla firmare da Mozilla
come componente **non elencato** (self-distributed). Non verra pubblicata nello
store e il file firmato resta privato.

1. Accedi al Developer Hub di `addons.mozilla.org` con un account Mozilla.
2. Crea un nuovo componente aggiuntivo e scegli la distribuzione autonoma/non
   elencata.
3. Carica un archivio ZIP che contenga direttamente i file della cartella
   `extension/`, con `manifest.json` alla radice dell'archivio.
4. Completa la validazione e scarica il file `.xpi` firmato.
5. Apri `about:addons`, premi l'ingranaggio, scegli **Installa componente
   aggiuntivo da file** e seleziona lo `.xpi`.

In alternativa, dopo aver creato le credenziali API AMO, firma dalla root del
repository con `web-ext`:

```bash
npx web-ext sign \
  --source-dir extension \
  --artifacts-dir web-ext-artifacts \
  --channel unlisted \
  --api-key "$AMO_JWT_ISSUER" \
  --api-secret "$AMO_JWT_SECRET"
```

Il file firmato viene salvato in `web-ext-artifacts/`. Per pubblicare un
aggiornamento bisogna aumentare `version`, firmare nuovamente e installare il
nuovo `.xpi`.

### Test

I test dell'estensione non richiedono dipendenze npm:

```bash
node --test tests/extension.test.cjs
```
