# Jay — ML Service: Trained Model Workstream

## Role
Source/curate a dataset, extract features, train and evaluate a real model, and package it as a self-contained, pluggable inference function that Sanket's pipeline calls as one more stage. This is the artifact that answers "what did you actually train" in front of the jury.

## What you own
- `ml-service/app/models/train.py` — training script
- `ml-service/app/models/features.py` — feature extraction from audio / perception output
- `ml-service/app/models/artifact/` — the saved trained model file (`.pkl`, `.pt`, whatever your framework produces)
- `ml-service/app/models/infer.py` — the ONLY file Sanket imports from you
- `ml-service/app/models/eval/` — confusion matrix, metrics report, notebook or script

## What you DO NOT touch
- Anything in `ml-service/app/pipeline/` — that's Sanket's. You hand him a function, not a service.

## Your contract — what you must deliver to Sanket

A single function, importable, no side effects on his code:

```python
# ml-service/app/models/infer.py

def predict(perception_output: dict) -> dict:
    """
    Input: the perception-stage dict Sanket already has
           (transcript, sound_events, emotion, speakers — same shape as the SRS contract)
    Output MUST be exactly:
    {
        "label": str,          # e.g. "fire_emergency", "assault", "normal_conversation"
        "confidence": float    # 0.0-1.0
    }
    """
```

If your track ends up being audio-feature-based (e.g. a replay/provenance classifier) instead of a situation classifier, the contract shape stays the same — `predict(input) -> {"label": ..., "confidence": ...}` — only the input and label vocabulary change. Confirm which track with the team before hour 10 so Sanket's `fusion.py` field name (`model_insight`) matches your actual output semantics.

## Dataset options (pick one at hour 2, don't relitigate later)

| Track | Dataset | Metric to report |
|---|---|---|
| Situation classifier | Built from your own pipeline outputs (AudioSet/ESC-50 sound labels + CREMA-D/RAVDESS emotion labels, hand-combined into labeled rows) | F1 per class, confusion matrix |
| Emotion fine-tune | RAVDESS + CREMA-D (+ IEMOCAP if time allows) | UAR (unweighted average recall), per-class F1 |
| Sound event fine-tune | AudioSet subset (danger + ambient classes) | mAP |
| Replay/provenance detector | ASVspoof 2019/2021 Physical Access subset | EER |

## File structure (your working folder)

```
ml-service/app/models/
├── train.py             # loads dataset, extracts features, trains, saves artifact
├── features.py           # shared feature extraction (also importable by infer.py)
├── infer.py               # predict() — the only file Sanket touches
├── artifact/
│   └── model.pkl          # or model.pt
└── eval/
    ├── evaluate.py         # runs on held-out set, prints metrics
    └── confusion_matrix.png
```

## Local setup

```bash
cd ml-service
source venv/bin/activate           # shared venv with Sanket, same requirements.txt
pip install scikit-learn librosa pandas   # add to requirements.txt if not present

# training
python app/models/train.py --data ./data/ --out app/models/artifact/model.pkl

# evaluation
python app/models/eval/evaluate.py --model app/models/artifact/model.pkl
```

## Dependencies on others
- **Waits on:** nothing to start — dataset sourcing and feature extraction can begin at hour 0, fully independent of Sanket's pipeline being ready.
- **You hand off to Sanket:** a working `infer.py` with `predict()` by **hour 14–16 latest** — this is the one hard sync point between you two. If you're behind, tell Sanket immediately so he keeps using a stub rather than blocking on you.
- **Coordinate with Lahari:** if your model's `label` output feeds into her alert severity logic, confirm the exact label vocabulary (e.g. `fire_emergency`, `assault`, `medical_distress`, `normal_conversation`) with her before finalizing training labels — she consumes this field in the backend.

## Timeline checkpoints
| Hour | You should have |
|---|---|
| 0–2 | Track chosen (table above), dataset download started |
| 2–10 | Dataset labeled/curated, feature extraction script working |
| 10–14 | Model trained, first evaluation run |
| 14–16 | `infer.py` handed to Sanket, integrated into his pipeline |
| 16–30 | Iterate on model quality if time allows; finalize confusion matrix / metrics slide |
| 30–38 | Prep the "what did you train and why" explanation — dataset, approach, metrics, one slide |
