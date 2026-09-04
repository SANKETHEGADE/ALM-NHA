import os
import sys
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# Ensure repository root and ml-service are on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.alm.inference import ALMInferencePipeline
from src.api.schemas import AnalyzeResponse

pipeline_instance: Optional[ALMInferencePipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance
    ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "core_alm_latest.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(BASE_DIR, "checkpoints", "core_alm_latest.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = None

    print(f"[ML Service] Initializing Core ALM Pipeline ONCE on startup. Checkpoint: {ckpt_path}")
    try:
        pipeline_instance = ALMInferencePipeline(checkpoint_path=ckpt_path)
        print("[ML Service] Pipeline initialized successfully and ready for inference.")
    except Exception as e:
        print(f"[ML Service] Error initializing pipeline: {e}")
        pipeline_instance = None
    yield
    print("[ML Service] Shutting down Core ALM Service.")

app = FastAPI(
    title="Smart Horizon 2026 - Core ALM ML Service",
    description="Core Audio Language Model & Multimodal Temporal Fusion Engine API",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
frontend_origin_env = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173,http://localhost:3000,http://localhost:4000,http://127.0.0.1:5173,http://127.0.0.1:3000,http://127.0.0.1:4000")
origins = [o.strip() for o in frontend_origin_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB max audio upload limit

@app.get("/health")
async def health_check():
    loaded = pipeline_instance is not None and getattr(pipeline_instance, "model", None) is not None
    return {
        "status": "ok" if loaded else "degraded",
        "service": "Core Audio Language Model (Core ALM)",
        "model_loaded": loaded,
        "team": "SH-DST-02 (Team ID: SHIH26-TID-320)"
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(
    audio_file: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
    question: Optional[str] = Form(""),
    language_hint: Optional[str] = Form("hi"),
    spoken_transcript: Optional[str] = Form(None),
    llm_model: Optional[str] = Form("gpt-4o-mini")
):
    """
    POST /analyze
    Accepts multipart/form-data with binary audio buffer or audio file and text question.
    Runs canonical Core ALM inference pipeline and returns validated AnalyzeResponse JSON schema.
    """
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Core ALM model service is not initialized or unavailable.")

    # Sanitize and default question
    final_question = question.strip() if question and question.strip() else ""
    if len(final_question) > 500:
        raise HTTPException(status_code=400, detail="Question string exceeds maximum allowed length (500 characters).")

    temp_audio_path = None
    try:
        audio_source = None

        if audio_file is not None:
            contents = await audio_file.read()
            if len(contents) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(status_code=413, detail="Uploaded audio file exceeds maximum limit of 50 MB.")

            if len(contents) > 0:
                ext = os.path.splitext(audio_file.filename)[1].lower() if (audio_file and audio_file.filename) else ".webm"
                if not ext or len(ext) > 6:
                    ext = ".webm"
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(contents)
                    temp_audio_path = tmp.name
                audio_source = temp_audio_path

        print(f"[ML Service] Processing POST /analyze. Question: '{final_question}', Audio file: {audio_file.filename if audio_file else 'None'}, Spoken: '{spoken_transcript}', LLM: '{llm_model}'")

        # Run canonical Core ALM inference
        analysis_result = pipeline_instance.analyze(
            audio_source=audio_source,
            question=final_question,
            language_hint=language_hint or "hi",
            spoken_transcript=spoken_transcript,
            llm_model=llm_model or "gpt-4o-mini"
        )

        return analysis_result

        return analysis_result

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ML Service] Exception during /analyze processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Core ALM Inference Error: {str(e)}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception as cleanup_err:
                print(f"[ML Service] Temporary file cleanup warning: {cleanup_err}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
