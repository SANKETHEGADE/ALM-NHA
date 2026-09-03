import json
import os
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np

from src.asr.preprocessing import load_and_preprocess_audio, TARGET_SAMPLE_RATE, normalize_language_code
from src.asr.tokenizer import MultilingualTokenizer


class MultilingualASRDataset(Dataset):
    """
    Multilingual ASR Dataset loader reading from JSON manifests across 7 languages.
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer: MultilingualTokenizer,
        max_samples: Optional[int] = None,
        target_sr: int = TARGET_SAMPLE_RATE,
        synthetic_fallback: bool = True
    ):
        self.tokenizer = tokenizer
        self.target_sr = target_sr
        self.synthetic_fallback = synthetic_fallback
        self.samples: List[Dict[str, Any]] = []

        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.samples = data
                elif isinstance(data, dict) and "samples" in data:
                    self.samples = data["samples"]
        elif synthetic_fallback:
            # Generate synthetic dummy samples if manifest file does not exist
            self.samples = self._generate_synthetic_samples()

        if max_samples and max_samples > 0:
            self.samples = self.samples[:max_samples]

    def _generate_synthetic_samples(self) -> List[Dict[str, Any]]:
        """Fallback synthetic dataset for tests and pipeline validation."""
        sample_texts = [
            ("नमस्ते भारत", "hi"),
            ("నమస్కారం ప్రపంచం", "te"),
            ("வணக்கம் உலகம்", "ta"),
            ("নমস্কার বিশ্ব", "bn"),
            ("سلام دنیا", "ur"),
            ("你好世界", "zh"),
            ("hello world emergency alert", "en"),
        ]
        items = []
        for text, lang in sample_texts:
            items.append({
                "audio_filepath": f"synthetic_{lang}.wav",
                "text": text,
                "language": lang,
                "duration": 2.0
            })
        return items

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        audio_path = item.get("audio_filepath", "")
        text = item.get("text", "")
        lang = normalize_language_code(item.get("language", "en"))

        # Load audio or generate dummy tone
        if os.path.exists(audio_path):
            waveform, _ = load_and_preprocess_audio(audio_path, target_sr=self.target_sr)
        else:
            # Generate 1.5s dummy sine audio for missing files/tests
            duration = float(item.get("duration", 1.5))
            num_samples = int(duration * self.target_sr)
            t = np.linspace(0, duration, num_samples, endpoint=False)
            sine_wave = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            waveform = torch.from_numpy(sine_wave).unsqueeze(0)

        token_ids = self.tokenizer.encode(text, language=lang, add_special_tokens=False)
        if not token_ids:
            token_ids = [self.tokenizer.space_id]

        return {
            "waveform": waveform.squeeze(0),  # (T,)
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "text": text,
            "language": lang
        }


def collate_asr_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate variable length audio and token sequences into padded tensors for CTC training.
    """
    waveforms = [item["waveform"] for item in batch]
    token_seqs = [item["token_ids"] for item in batch]
    texts = [item["text"] for item in batch]
    languages = [item["language"] for item in batch]

    # Pad audio waveforms
    audio_lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
    max_audio_len = max(len(w) for w in waveforms)
    padded_audio = torch.zeros(len(waveforms), max_audio_len, dtype=torch.float32)
    for i, w in enumerate(waveforms):
        padded_audio[i, :len(w)] = w

    # Pad target token IDs (use blank_id=2 or pad_id=0 for padding)
    target_lengths = torch.tensor([len(t) for t in token_seqs], dtype=torch.long)
    max_target_len = max(len(t) for t in token_seqs)
    padded_targets = torch.zeros(len(token_seqs), max_target_len, dtype=torch.long)
    for i, t in enumerate(token_seqs):
        padded_targets[i, :len(t)] = t

    return {
        "audio": padded_audio,               # (Batch, Max_Samples)
        "audio_lengths": audio_lengths,       # (Batch,)
        "targets": padded_targets,           # (Batch, Max_Tokens)
        "target_lengths": target_lengths,     # (Batch,)
        "texts": texts,
        "languages": languages
    }
