import os
import shutil
import tempfile
import time
import logging
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AnalyzeResponse
from app.pipeline.perception import run_perception, inspect_audio_file
from app.pipeline.fusion import fuse, call_model_infer
from app.pipeline.llm_reasoning import reason

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_service")

app = FastAPI(
    title="ML Perception & Orchestration Service",
    description="End-to-end multimodal perception, situation classification, and LLM reasoning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ml-service",
        "models": {
            "perception": "ready",
            "classifier": "ready",
            "reasoning": "ready"
        }
    }


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Multimodal audio perception and reasoning analysis"
)
async def analyze_audio(
    audio_file: UploadFile = File(..., description="Binary audio file (.wav or .mp3)"),
    session_id: str = Form(..., description="Unique session identifier"),
    language_hint: Optional[str] = Form(None, description="Optional language code (auto, hi, en)")
):
    start_time = time.perf_counter()
    temp_path = None

    if not session_id or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="session_id is required"
        )

    # 1. Save uploaded file to a temporary file
    try:
        suffix = os.path.splitext(audio_file.filename or "")[1].lower()
        if not suffix or suffix not in [".wav", ".mp3", ".ogg"]:
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(audio_file.file, temp_file)

        # 2. Validate audio header and structure
        try:
            inspect_audio_file(temp_path)
        except ValueError as val_err:
            logger.warning("Audio validation error for session %s: %s", session_id, val_err)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Malformed or unsupported audio: {val_err}"
            )

    except HTTPException:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        logger.error("Failed to read incoming audio file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unable to process audio file: {e}"
        )

    try:
        # 3. Concurrent Perception Pipeline (Whisper + PANNs + SpeechBrain)
        try:
            perception = await run_perception(temp_path, language_hint)
        except Exception as p_err:
            logger.error("Perception pipeline fatal failure: %s", p_err)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Perception pipeline inference failure: {p_err}"
            )

        # 4. Domain Classification Inference (Jay's model)
        try:
            model_insight = call_model_infer(perception)
        except Exception as m_err:
            logger.error("Domain classifier failure: %s", m_err)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Situation classifier inference failure: {m_err}"
            )

        # 5. LLM Reasoning Layer (Ollama local -> Groq fallback -> null-safe 3s timeout)
        perception_for_reasoning = dict(perception)
        perception_for_reasoning["model_insight"] = model_insight
        reasoning = await reason(perception_for_reasoning)

        # 6. Response Fusion into Frozen Contract
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        fused_response = fuse(
            perception=perception,
            model_insight=model_insight,
            reasoning=reasoning,
            session_id=session_id,
            processing_ms=elapsed_ms
        )

        return fused_response

    finally:
        # Always clean up temporary audio file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.debug("Failed to remove temporary file %s: %s", temp_path, e)
