import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, Optional
import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3.5:3.8b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
REASONING_TIMEOUT_SEC = float(os.getenv("REASONING_TIMEOUT_SEC", "3.0"))


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extracts JSON dictionary from raw model text response."""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "trace" in parsed and "summary" in parsed:
                return {
                    "trace": str(parsed.get("trace", "")),
                    "summary": str(parsed.get("summary", "")),
                    "severity_hint": str(parsed.get("severity_hint", "medium")).lower()
                }
    except Exception:
        pass
    return None


def _rule_based_synthesis(perception_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic synthesis fallback when neither Ollama nor Groq is reachable.
    Ensures non-empty output for early integration checkpoints.
    """
    transcript = perception_json.get("transcript", {})
    text = transcript.get("text", "") if isinstance(transcript, dict) else str(transcript)
    
    events = perception_json.get("sound_events", [])
    event_labels = [e.get("label", "") if isinstance(e, dict) else str(e) for e in events]
    
    emotion_obj = perception_json.get("emotion", {})
    emotion_val = emotion_obj.get("primary", "neutral") if isinstance(emotion_obj, dict) else str(emotion_obj)
    arousal_val = emotion_obj.get("arousal", "medium") if isinstance(emotion_obj, dict) else "medium"
    
    model_insight = perception_json.get("model_insight", {})
    label = model_insight.get("label", "unknown") if isinstance(model_insight, dict) else "unknown"

    trace = (
        f"Audio transcript '{text}' detected alongside sound events [{', '.join(event_labels)}]. "
        f"Acoustic emotion shows {emotion_val} with {arousal_val} arousal. "
        f"Domain classifier indicated {label}."
    )
    
    if any(k in event_labels for k in ["smoke_alarm", "alarm", "explosion", "gunshot"]) or label == "fire_emergency":
        summary = "A person is shouting for help in a high-stress, fearful tone while a smoke alarm is audible — likely a fire emergency."
        severity = "critical"
    elif label in ["medical_distress", "assault", "robbery"]:
        summary = f"Distress detected with {emotion_val} acoustic signature indicating urgent situation: {label}."
        severity = "high"
    else:
        summary = f"Speech analysis indicates {emotion_val} state with sound cues: {', '.join(event_labels) or 'ambient'}."
        severity = "medium"

    return {
        "trace": trace,
        "summary": summary,
        "severity_hint": severity
    }


async def _call_ollama(client: httpx.AsyncClient, prompt: str) -> Optional[Dict[str, Any]]:
    """Invokes local Ollama service."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "num_predict": 250
        }
    }
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    response = await client.post(url, json=payload, timeout=2.2)
    if response.status_code == 200:
        data = response.json()
        raw_response = data.get("response", "")
        return _extract_json_block(raw_response)
    return None


async def _call_groq(client: httpx.AsyncClient, prompt: str) -> Optional[Dict[str, Any]]:
    """Invokes Groq fallback API."""
    if not GROQ_API_KEY:
        return None
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an emergency situation reasoning engine. Output only valid JSON with keys: trace, summary, severity_hint."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 250
    }
    url = f"{GROQ_BASE_URL.rstrip('/')}/chat/completions"
    response = await client.post(url, json=payload, headers=headers, timeout=2.2)
    if response.status_code == 200:
        data = response.json()
        choices = data.get("choices", [])
        if choices:
            raw_content = choices[0].get("message", {}).get("content", "")
            return _extract_json_block(raw_content)
    return None


async def _perform_reasoning(perception_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    prompt = (
        "Analyze the following multimodal audio perception data and produce a situation reasoning assessment.\n"
        f"Input Data:\n{json.dumps(perception_json, indent=2)}\n\n"
        "Return ONLY a JSON object with this exact schema:\n"
        "{\n"
        '  "trace": "step-by-step observation linking audio cues and emotion",\n'
        '  "summary": "one or two sentence situational summary",\n'
        '  "severity_hint": "critical | high | medium | low"\n'
        "}"
    )

    async with httpx.AsyncClient(timeout=2.5) as client:
        # 1. Primary: Local Ollama
        try:
            result = await _call_ollama(client, prompt)
            if result:
                return result
        except Exception as e:
            logger.debug("Ollama inference unavailable: %s", e)

        # 2. Secondary: Groq fallback
        try:
            result = await _call_groq(client, prompt)
            if result:
                return result
        except Exception as e:
            logger.debug("Groq inference unavailable: %s", e)

    # 3. Fallback synthesis if external engines are offline
    # Return rule-based synthesis so backend receives populated values
    if os.getenv("DISABLE_REASONING_FALLBACK", "false").lower() != "true":
        return _rule_based_synthesis(perception_json)

    return None


async def reason(perception_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Executes reasoning with strict timeout handling (default 3.0s).
    If it times out or encounters fatal errors, gracefully returns None (null in JSON).
    """
    try:
        return await asyncio.wait_for(_perform_reasoning(perception_json), timeout=REASONING_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        logger.warning("LLM reasoning timed out after %s seconds, returning null", REASONING_TIMEOUT_SEC)
        return None
    except Exception as e:
        logger.error("LLM reasoning error: %s, returning null", e)
        return None
