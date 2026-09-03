from typing import List, Optional
from pydantic import BaseModel, Field


class Transcript(BaseModel):
    text: str = Field(..., description="Transcribed speech text")
    language: str = Field(default="en", description="Detected or specified language code")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")


class SoundEvent(BaseModel):
    label: str = Field(..., description="Sound event classification label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    start_sec: float = Field(..., ge=0.0, description="Event start timestamp in seconds")
    end_sec: float = Field(..., ge=0.0, description="Event end timestamp in seconds")


class Emotion(BaseModel):
    primary: str = Field(..., description="Primary detected emotion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    arousal: str = Field(default="medium", description="Arousal level: high, medium, low")


class SpeakerDiarization(BaseModel):
    speaker_id: str = Field(..., description="Speaker identifier e.g. spk_1")
    start_sec: float = Field(..., ge=0.0, description="Segment start timestamp in seconds")
    end_sec: float = Field(..., ge=0.0, description="Segment end timestamp in seconds")


class Speakers(BaseModel):
    count: int = Field(default=1, ge=0, description="Number of distinct detected speakers")
    diarization: List[SpeakerDiarization] = Field(default_factory=list, description="Speaker segmentation intervals")


class ModelInsight(BaseModel):
    label: str = Field(..., description="Situation category predicted by trained model")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")


class Reasoning(BaseModel):
    trace: str = Field(..., description="Chain of thought reasoning trace")
    summary: str = Field(..., description="Human readable synthesis summary")
    severity_hint: str = Field(default="medium", description="Severity assessment: critical, high, medium, low, informational")


class AnalyzeResponse(BaseModel):
    session_id: str = Field(..., description="Passthrough session identifier")
    duration_sec: float = Field(..., ge=0.0, description="Audio duration in seconds")
    transcript: Transcript = Field(..., description="Speech-to-text perception output")
    sound_events: List[SoundEvent] = Field(default_factory=list, description="Detected acoustic events")
    emotion: Emotion = Field(..., description="Emotion recognition analysis")
    speakers: Speakers = Field(..., description="Speaker diarization and count")
    model_insight: ModelInsight = Field(..., description="Domain situation classifier output")
    reasoning: Optional[Reasoning] = Field(default=None, description="LLM synthesis reasoning output, null-safe")
    processing_ms: int = Field(..., ge=0, description="Total server processing time in milliseconds")
