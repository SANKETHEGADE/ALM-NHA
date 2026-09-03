import io
import math
import os
import struct
import sys
import unittest
import wave

from fastapi.testclient import TestClient

# Ensure app package is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.schemas import AnalyzeResponse
from app.pipeline.perception import inspect_audio_file, run_perception
from app.pipeline.fusion import call_model_infer, fuse
from app.pipeline.llm_reasoning import reason, _rule_based_synthesis


def create_synthetic_wav(duration_sec: float = 2.0, sample_rate: int = 16000, freq: float = 440.0) -> bytes:
    """Generates a valid in-memory PCM 16-bit mono WAV file."""
    buf = io.BytesIO()
    total_frames = int(duration_sec * sample_rate)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(total_frames):
            t = float(i) / sample_rate
            value = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * freq * t))
            frames.extend(struct.pack("<h", value))
        wf.writeframes(frames)
    return buf.getvalue()


class TestMLPipeline(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.wav_bytes = create_synthetic_wav(duration_sec=2.5)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "ml-service")

    async def test_perception_pipeline(self):
        test_wav_path = "test_clip.wav"
        with open(test_wav_path, "wb") as f:
            f.write(self.wav_bytes)
        try:
            meta = inspect_audio_file(test_wav_path)
            self.assertGreater(meta["duration_sec"], 2.0)
            self.assertEqual(meta["format"], "wav")

            perception = await run_perception(test_wav_path, language_hint="en")
            self.assertIn("transcript", perception)
            self.assertIn("sound_events", perception)
            self.assertIn("emotion", perception)
            self.assertIn("speakers", perception)
            self.assertEqual(perception["transcript"]["language"], "en")
        finally:
            if os.path.exists(test_wav_path):
                os.remove(test_wav_path)

    def test_fusion_with_jays_model(self):
        mock_perception = {
            "duration_sec": 8.4,
            "transcript": {
                "text": "Someone help, there's smoke coming from the kitchen!",
                "language": "en",
                "confidence": 0.91
            },
            "sound_events": [
                {"label": "smoke_alarm", "confidence": 0.87, "start_sec": 1.2, "end_sec": 3.0},
                {"label": "shouting", "confidence": 0.78, "start_sec": 0.0, "end_sec": 2.1}
            ],
            "emotion": {
                "primary": "fear",
                "confidence": 0.82,
                "arousal": "high"
            },
            "speakers": {
                "count": 1,
                "diarization": [{"speaker_id": "spk_1", "start_sec": 0.0, "end_sec": 8.4}]
            }
        }

        # Call Jay's infer through call_model_infer
        model_insight = call_model_infer(mock_perception)
        self.assertIn("label", model_insight)
        self.assertIn("confidence", model_insight)
        self.assertIsInstance(model_insight["label"], str)
        self.assertIsInstance(model_insight["confidence"], float)

        # Fused response
        fused = fuse(
            perception=mock_perception,
            model_insight=model_insight,
            reasoning={
                "trace": "test trace",
                "summary": "test summary",
                "severity_hint": "critical"
            },
            session_id="test_sess_001",
            processing_ms=120
        )

        # Validate strictly against schema
        validated = AnalyzeResponse(**fused)
        self.assertEqual(validated.session_id, "test_sess_001")
        self.assertEqual(validated.transcript.confidence, 0.91)
        self.assertEqual(validated.model_insight.label, model_insight["label"])

    async def test_reasoning_fallback_and_null_safety(self):
        perception_data = {
            "transcript": {"text": "Fire in the building!"},
            "sound_events": [{"label": "alarm"}],
            "emotion": {"primary": "fear", "arousal": "high"},
            "speakers": {"count": 1}
        }
        res = await reason(perception_data)
        # Should return a valid reasoning dict with trace, summary, severity_hint or None
        if res is not None:
            self.assertIn("trace", res)
            self.assertIn("summary", res)
            self.assertIn("severity_hint", res)

    def test_analyze_endpoint_success(self):
        files = {"audio_file": ("sample.wav", self.wav_bytes, "audio/wav")}
        data = {"session_id": "sess_unit_test_123", "language_hint": "en"}

        response = self.client.post("/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 200, response.text)

        result = response.json()
        # Verify frozen contract fields
        self.assertEqual(result["session_id"], "sess_unit_test_123")
        self.assertIn("duration_sec", result)
        self.assertIn("transcript", result)
        self.assertIn("sound_events", result)
        self.assertIn("emotion", result)
        self.assertIn("speakers", result)
        self.assertIn("model_insight", result)
        self.assertIn("reasoning", result)
        self.assertIn("processing_ms", result)

        # Confirm shape against schema
        validated = AnalyzeResponse(**result)
        self.assertEqual(validated.session_id, "sess_unit_test_123")

    def test_analyze_endpoint_malformed_audio_422(self):
        # Sending random invalid text as audio file
        corrupted_bytes = b"NOT_A_VALID_WAV_OR_MP3_DATA_STREAM"
        files = {"audio_file": ("bad_audio.wav", corrupted_bytes, "audio/wav")}
        data = {"session_id": "sess_bad_audio"}

        response = self.client.post("/analyze", files=files, data=data)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
