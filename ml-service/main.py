import os
import sys
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.alm.inference import ALMInferencePipeline

app = FastAPI(
    title="Smart Horizon 2026 - Core ALM ML Service",
    description="Core Audio Language Model & Multimodal Temporal Fusion Engine API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core ALM Pipeline instance
pipeline = ALMInferencePipeline(
    checkpoint_path="checkpoints/best_checkpoint.pt" if os.path.exists("checkpoints/best_checkpoint.pt") else None
)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "Core Audio Language Model (Core ALM)",
        "team": "SH-DST-02 (Team ID: SHIH26-TID-320)"
    }

@app.post("/analyze")
async def analyze_audio(
    audio_file: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
    question: Optional[str] = Form("Where is the speaker likely to be?"),
    language_hint: Optional[str] = Form("hi")
):
    """
    POST /analyze
    Input: audio file binary buffer, question, optional language hint
    Returns exact Core ALM contract:
    {
      "answer": "...",
      "confidence": 0.87,
      "speech": {},
      "speakers": [],
      "audio_events": [],
      "paralinguistic": {},
      "scene": {},
      "evidence": []
    }
    """
    try:
        audio_buffer = None
        if audio_file is not None:
            audio_buffer = await audio_file.read()
        
        # Default question fallback if empty string passed
        if not question or not question.strip():
            question = "Where is the speaker likely to be?"

        print(f"[ML Service] Processing POST /analyze request. Question: '{question}', Audio Size: {len(audio_buffer) if audio_buffer else 0} bytes")

        analysis_result = pipeline.analyze(
            audio_source=audio_buffer,
            question=question,
            language_hint=language_hint or "hi"
        )

        return analysis_result

    except Exception as e:
        print(f"[ML Service] Error processing /analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Core ALM Analysis Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
