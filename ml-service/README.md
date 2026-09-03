# ML Service

Audio Language Model (ALM) Service architecture baseline for Smart Horizon 2026 (SH-DST-02, Team SHIH26-TID-320).

## Directory Structure

```
ml-service/
├── README.md
├── requirements.txt
├── configs/
├── datasets/
│   └── manifests/
├── models/
├── checkpoints/
├── scripts/
├── evaluation/
├── shared/
├── src/
│   ├── audio/
│   ├── asr/                 # Lahari: Multilingual Non-Whisper ASR (7 Languages + Speech Embeddings)
│   ├── events/
│   ├── speaker/
│   ├── paralinguistic/
│   ├── fusion/              # Jay: Temporal Fusion & Multimodal Fusion
│   ├── training/            # ASR & Multimodal Training Subsystem
│   └── alm/
└── tests/
```

## Subsystems

### 🎙️ Multilingual Non-Whisper ASR (Lahari)
- **Zero Whisper Dependency:** Pure non-Whisper architecture supporting 7 languages (Hindi, Telugu, Tamil, Bengali, Urdu, Mandarin, English).
- **Core ALM Contract:** High-resolution speech embeddings extraction via `ASRModel.encode(audio)` feeding directly into Jay's temporal fusion.
- **Documentation:** See [`src/asr/README.md`](src/asr/README.md) for API usage, CLI commands, and training instructions.

## Setup & Guidelines

- **Do NOT commit large datasets or model checkpoints to GitHub.**
- Keep model weights (`*.pt`, `*.pth`, `*.safetensors`, `*.bin`) and dataset archives locally or stored in designated cloud storage.
- All code modules should be placed in `src/` or `shared/`.

