# Sanket — ML Service: Perception Pipeline + Orchestration

## Role
Own the `/analyze` FastAPI endpoint end to end. Run Whisper, PANNs, SpeechBrain in parallel, call Jay's trained model as a pipeline stage, call the LLM reasoning layer, fuse everything into the frozen contract JSON.

## What you own
- `ml-service/app/main.py` — FastAPI app, route definitions
- `ml-service/app/pipeline/perception.py` — Whisper + PANNs + SpeechBrain, run concurrently (asyncio.gather or threadpool)
- `ml-service/app/pipeline/fusion.py` — merges perception + Jay's model output + reasoning output into one response
- `ml-service/app/pipeline/llm_reasoning.py` — Ollama call (Phi-3.5-mini default), Groq fallback, timeout handling
- `ml-service/app/schemas.py` — pydantic models matching the contract exactly

## What you DO NOT touch
- `ml-service/app/models/` (Jay's training code + artifact) — you only ever import `infer.py` and call `predict()`. Never edit training logic.

## Your contract — inbound (what Backend sends you)

```
POST /analyze
Content-Type: multipart/form-data

audio_file: binary (.wav/.mp3)
session_id: string (passthrough — you do NOT store this, just echo it back)
language_hint (optional): string, e.g. "auto", "hi", "en"
```

## Your contract — outbound (what you must return)

```json
{
  "session_id": "sess_abc123",
  "duration_sec": 8.4,
  "transcript": {
    "text": "Someone help, there's smoke coming from the kitchen!",
    "language": "en",
    "confidence": 0.91
  },
  "sound_events": [
    { "label": "smoke_alarm", "confidence": 0.87, "start_sec": 1.2, "end_sec": 3.0 },
    { "label": "shouting", "confidence": 0.78, "start_sec": 0.0, "end_sec": 2.1 }
  ],
  "emotion": {
    "primary": "fear",
    "confidence": 0.82,
    "arousal": "high"
  },
  "speakers": {
    "count": 1,
    "diarization": [
      { "speaker_id": "spk_1", "start_sec": 0.0, "end_sec": 8.4 }
    ]
  },
  "model_insight": {
    "label": "fire_emergency",
    "confidence": 0.89
  },
  "reasoning": {
    "trace": "High-arousal fear speech co-occurs with a smoke alarm...",
    "summary": "A person is shouting for help in a high-stress, fearful tone while a smoke alarm is audible — likely a fire emergency.",
    "severity_hint": "critical"
  },
  "processing_ms": 1840
}
```

- `model_insight` = Jay's trained model output. Call contract: `from app.models.infer import predict; result = predict(perception_output)` → `{"label": str, "confidence": float}`. If Jay's model isn't ready yet, stub this with a hardcoded value — do not block your integration on it.
- `reasoning` = null-safe. If the LLM call times out (>3s), return `reasoning: null` and everything else populated. Never fail the whole request because reasoning was slow.
- **422**: malformed/unsupported audio. **500**: model inference failure (backend retries once, then marks session failed).

## Non-negotiable
Return this full shape — including `model_insight` and `reasoning` fields, even as dummy/mocked values — starting hour 2, so Backend can build against it immediately. Fill in real values incrementally without ever renaming a field.

## File structure (your working folder)

```
ml-service/app/pipeline/
├── perception.py       # whisper_transcribe(), detect_sound_events(), analyze_emotion_speakers()
├── fusion.py            # fuse(perception, model_insight, reasoning) -> final dict
└── llm_reasoning.py     # reason(perception_json) -> {trace, summary, severity_hint} | None
```

## Local setup

```bash
cd ml-service
python -m venv venv && source venv/bin/activate     # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# pull local LLM
ollama pull phi3.5:3.8b
uvicorn app.main:app --reload --port 8000
```

Test with a sample clip:
```bash
curl -X POST http://localhost:8000/analyze \
  -F "audio_file=@sample.wav" \
  -F "session_id=test123"
```

## Dependencies on others
- **Waits on:** nothing to start — build perception pipeline on your own sample clips immediately.
- **Jay hands you:** `app/models/infer.py` with a working `predict()` function, ideally by hour 14–16.
- **Backend (Anwin) calls you:** his session endpoint forwards audio to your `/analyze` — coordinate the exact base URL (`http://ml-service:8000` in docker-compose) at hour 0–2.

## Timeline checkpoints
| Hour | You should have |
|---|---|
| 2–10 | Whisper + PANNs + SpeechBrain running individually on sample clips |
| 10–16 | All 3 fused into one JSON; `model_insight` stubbed until Jay's model lands |
| 16–22 | LLM reasoning wired (Ollama local, Groq fallback tested) |
| 22–24 | Integration pass — real end-to-end request from Backend works |
| 30–38 | Swap stub for Jay's real trained model; error handling for silence/noise/no-speech |
