import os
import io
import time
from typing import Union, Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.asr.preprocessing import (
    load_and_preprocess_audio,
    normalize_language_code,
    SUPPORTED_LANGUAGES,
    TARGET_SAMPLE_RATE,
)
from src.asr.tokenizer import MultilingualTokenizer
from src.asr.encoder import SpeechEncoder
from src.asr.decoder import CTCDecoder


class ASRModel(nn.Module):
    """
    Multilingual Non-Whisper ASR Model supporting 7 Target Languages:
    Hindi, Telugu, Tamil, Bengali, Urdu, Mandarin, and English.

    Primary Interface for ALM Multimodal Fusion:
        - encode(audio): Extracts speech representations for Jay's temporal fusion.
        - transcribe(audio, language): Produces multilingual transcription & token alignments.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        hidden_dim: int = 512,
        vocab_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        super().__init__()
        
        # 1. Device Setup
        if device is None:
            self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_type = str(device)
        self.target_device = torch.device(self.device_type)

        # 2. Multilingual Tokenizer
        self.tokenizer = MultilingualTokenizer(vocab_path=vocab_path)
        self.vocab_size = self.tokenizer.vocab_size()

        # 3. Speech Representation Encoder (Non-Whisper)
        self.encoder = SpeechEncoder(
            pretrained_name_or_path=model_name_or_path,
            hidden_dim=hidden_dim
        )
        self.hidden_dim = self.encoder.hidden_dim

        # 4. CTC Classification & Decoding Head
        self.decoder = CTCDecoder(
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size,
            blank_id=self.tokenizer.blank_id
        )

        self.to(self.target_device)

    @property
    def embedding_dim(self) -> int:
        """Returns the speech embedding dimension (e.g. 512 or 768)."""
        return self.hidden_dim

    # =========================================================================
    # CORE ALM CONTRACT: SPEECH REPRESENTATION ENCODER
    # =========================================================================

    def encode(
        self,
        audio: Union[torch.Tensor, np.ndarray, str, bytes, bytearray, io.BytesIO],
        pool: bool = False
    ) -> torch.Tensor:
        """
        Extract high-resolution speech representations for Core ALM Temporal Fusion.
        This is the primary endpoint consumed by Jay's multimodal fusion pipeline.

        Args:
            audio: Audio file path, raw bytes, numpy waveform, or PyTorch tensor.
            pool: If True, returns global pooled embedding (B, D). If False, returns (B, T, D) frame embeddings.

        Returns:
            torch.Tensor: Frame-level embeddings of shape (B, T, D) or pooled (B, D).
        """
        self.eval()
        with torch.no_grad():
            waveform, _ = self._prepare_audio_tensor(audio)
            frame_embeddings, pooled_embedding = self.encoder(waveform)
            if pool:
                return pooled_embedding
            return frame_embeddings

    def get_speech_embeddings(
        self,
        audio: Union[torch.Tensor, np.ndarray, str, bytes, bytearray, io.BytesIO]
    ) -> Dict[str, torch.Tensor]:
        """
        Rich dictionary output for ALM multimodal fusion pipeline containing both
        frame-level representations and pooled sentence embeddings.
        """
        self.eval()
        with torch.no_grad():
            waveform, sr = self._prepare_audio_tensor(audio)
            frame_embeddings, pooled_embedding = self.encoder(waveform)
            return {
                "frame_embeddings": frame_embeddings,
                "pooled_embedding": pooled_embedding,
                "embedding_dim": torch.tensor(self.hidden_dim, device=frame_embeddings.device),
                "num_frames": torch.tensor(frame_embeddings.size(1), device=frame_embeddings.device),
                "sample_rate": torch.tensor(sr, device=frame_embeddings.device)
            }

    # =========================================================================
    # MULTILINGUAL TRANSCRIPTION INTERFACE
    # =========================================================================

    def transcribe(
        self,
        audio: Union[torch.Tensor, np.ndarray, str, bytes, bytearray, io.BytesIO],
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe multilingual audio in any of the 7 supported languages.

        Args:
            audio: Audio input (file path, raw bytes, numpy array, or tensor).
            language: Optional language hint ('hi', 'te', 'ta', 'bn', 'ur', 'zh', 'en').

        Returns:
            Dict containing:
                - text: Decoded transcript string
                - language: Detected or requested language code
                - confidence: Average token confidence score (0.0 to 1.0)
                - duration_seconds: Audio duration in seconds
                - embedding_shape: Shape of the generated frame embeddings
        """
        start_time = time.perf_counter()
        lang_code = normalize_language_code(language) if language else "en"

        self.eval()
        with torch.no_grad():
            waveform, sr = self._prepare_audio_tensor(audio)
            duration = float(waveform.size(-1)) / float(sr)

            # 1. Acoustic encoding
            frame_embeddings, _ = self.encoder(waveform)

            # 2. CTC projection & greedy decode
            logits = self.decoder(frame_embeddings)
            decode_results = self.decoder.decode_greedy(logits, self.tokenizer)

            result = decode_results[0] if decode_results else {"text": "", "token_ids": [], "confidence": 0.0}
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)

            return {
                "text": result["text"],
                "language": lang_code,
                "confidence": result["confidence"],
                "tokens": [self.tokenizer.id_to_token.get(tid, "") for tid in result["token_ids"]],
                "duration_seconds": round(duration, 3),
                "latency_ms": elapsed_ms,
                "embedding_shape": list(frame_embeddings.shape)
            }

    # =========================================================================
    # TRAINING FORWARD PASS
    # =========================================================================

    def forward(
        self,
        audio: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass returning logits, frame embeddings, and CTC loss if targets are given.
        """
        frame_embeddings, pooled_embedding = self.encoder(audio)
        logits = self.decoder(frame_embeddings)

        outputs = {
            "logits": logits,
            "frame_embeddings": frame_embeddings,
            "pooled_embedding": pooled_embedding,
        }

        if targets is not None and target_lengths is not None:
            # CTC input length is number of time frames in frame_embeddings
            if input_lengths is None:
                input_lengths = torch.full(
                    (audio.size(0),),
                    frame_embeddings.size(1),
                    dtype=torch.long,
                    device=audio.device
                )
            loss = self.decoder.compute_loss(logits, targets, input_lengths, target_lengths)
            outputs["loss"] = loss

        return outputs

    # =========================================================================
    # HELPERS & SERIALIZATION
    # =========================================================================

    def _prepare_audio_tensor(
        self,
        audio: Union[torch.Tensor, np.ndarray, str, bytes, bytearray, io.BytesIO]
    ) -> Tuple[torch.Tensor, int]:
        """Standardize any audio input to 16kHz tensor on self.target_device."""
        if isinstance(audio, torch.Tensor) and audio.ndim in (1, 2) and audio.size(-1) > 0:
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            return audio.to(self.target_device, dtype=torch.float32), TARGET_SAMPLE_RATE

        waveform, sr = load_and_preprocess_audio(audio, target_sr=TARGET_SAMPLE_RATE)
        return waveform.to(self.target_device, dtype=torch.float32), sr

    def save_checkpoint(self, path: str):
        """Save model state dict and tokenizer vocabulary."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "hidden_dim": self.hidden_dim,
            "vocab": self.tokenizer.token_to_id
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model state dict and tokenizer vocabulary from checkpoint."""
        checkpoint = torch.load(path, map_location=self.target_device)
        self.load_state_dict(checkpoint["state_dict"])
        if "vocab" in checkpoint:
            self.tokenizer.token_to_id = checkpoint["vocab"]
            self.tokenizer.id_to_token = {int(v): k for k, v in self.tokenizer.token_to_id.items()}
