# ML Service

Audio Language Model (ALM) Service architecture baseline.

## Directory Structure

```
ml-service/
├── README.md
├── requirements.txt
├── configs/
├── datasets/
├── models/
├── checkpoints/
├── scripts/
├── evaluation/
├── shared/
└── src/
    ├── audio/
    ├── asr/
    ├── events/
    ├── speaker/
    ├── paralinguistic/
    ├── fusion/
    └── alm/
```

## Setup & Guidelines

- **Do NOT commit large datasets or model checkpoints to GitHub.**
- Keep model weights (`*.pt`, `*.pth`, `*.safetensors`, `*.bin`) and dataset archives locally or stored in designated cloud storage.
- All code modules should be placed in `src/` or `shared/`.
