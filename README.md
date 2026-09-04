# Audio Language Model (ALM-NHCE) — Context-Aware Acoustic Intelligence & Multimodal Decision Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Node.js v18+](https://img.shields.io/badge/node-v18+-green.svg)](https://nodejs.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

An end-to-end multimodal acoustic perception and intelligence platform built to perform real-time speech recognition, acoustic sound event detection, paralinguistic emotion recognition, speaker verification/diarization, and learned cross-modal latent fusion for high-precision environmental scene understanding and alert telemetry.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [End-to-End Workflow](#end-to-end-workflow)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Feature Inventory](#feature-inventory)
- [Machine Learning Architecture](#machine-learning-architecture)
- [Dataset → Model Mapping](#dataset--model-mapping)
- [Model Inventory](#model-inventory)
- [Training Pipeline](#training-pipeline)
- [Fusion Model Deep Analysis](#fusion-model-deep-analysis)
- [Inference Pipeline](#inference-pipeline)
- [Backend Architecture](#backend-architecture)
- [ML Service Architecture](#ml-service-architecture)
- [Frontend Architecture](#frontend-architecture)
- [API Documentation](#api-documentation)
- [Input and Output Specification](#input-and-output-specification)
- [Results and Evaluation](#results-and-evaluation)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Training Models](#training-models)
- [Running Inference](#running-inference)
- [Configuration](#configuration)
- [Hardware Requirements](#hardware-requirements)
- [Limitations](#limitations)
- [Current vs Recommended Architecture](#current-vs-recommended-architecture)
- [Documentation Cleanup Candidates](#documentation-cleanup-candidates)
- [License](#license)

---

## Overview

The **Audio Language Model (ALM-NHCE)** project combines custom acoustic perception models, specialized deep-learning extractors, and learned multimodal projection layers to translate raw complex audio (48kHz/16kHz stream captures and files) into structured forensic intelligence.

The platform continuously extracts:
1. **Multilingual ASR & Dialogue**: Automated Speech Recognition with multilingual auto-detection (English, Hindi, Mandarin, etc.).
2. **Sound Event Detection (SED)**: Acoustic classification across emergency alarms, gunfire, machinery cavitation, glass breaking, subway noise, and PA announcements.
3. **Paralinguistic & Emotion Analysis**: Speech tone, vocal arousal, and affect detection (fear, urgency, alarmed, neutral).
4. **Speaker Biometrics**: Active speaker identification, count, and temporal segment diarization.
5. **Learned Multimodal Latent Fusion**: Projection of perception embeddings into a unified ALM reasoning head to generate actionable scene summaries and threat alert rules.

---

## Key Features

- **Real-Time Acoustic Telemetry**: Dynamic 64-band FFT spectral synthesis visualizer in both ASCII terminal format and web canvas.
- **Multimodal Perception Array**: Simultaneous extraction of speech text, environmental sound triggers, paralinguistic tone, and speaker embeddings.
- **Cross-Modal Latent Fusion**: Multi-Branch Linear Projection & Attention-Weighted Fusion (`src/alm/fusion_adapter.py` and `src/alm/reasoning_layer.py`).
- **Live Alert & Telemetry Dispatch**: Express WebSocket broadcasting of acoustic security events to live workstations.
- **Microphone Capture & Offline Audio Analysis**: Support for live WebRTC audio streams, drag-and-drop file ingestion (.wav, .mp3, .ogg, .flac, .webm), and verification scenario mocks (Threat, Industrial, Fire Hazard, Conversational).

---

## System Architecture

```mermaid
flowchart LR
    subgraph Client ["Frontend Interface (React / Vite)"]
        UI[Live Workstation Console]
        ASCII[ASCII Acoustic Visualizer]
        REC[WebRTC Mic Ingestion]
    end

    subgraph NodeBackend ["Node.js Backend (Express / WS / SQLite)"]
        API_GATEWAY[Express REST API Gateway :4000]
        WS_SERVER[WebSocket Telemetry Server :4000]
        DB[(SQLite Database)]
        ML_CLIENT[ML Client HTTP Proxy]
    end

    subgraph MLService ["Python ML Service (FastAPI / PyTorch :8000)"]
        FASTAPI[FastAPI Router main.py]
        PREPROC[Audio Preprocessing & Resampling Librosa/Torchaudio]
        
        subgraph Perception ["Perception Models"]
            ASR[Whisper / Custom ASR Model]
            SED[Sound Event Classifier CNN/ResNet]
            EMO[Paralinguistic Emotion Extractor]
            SPK[Speaker Diarization & Biometrics]
        end
        
        PROJ[Linear Projection Layers]
        FUSION[Learned Latent Fusion Adapter]
        LLM[ALM Reasoning Head / GPT Integration]
    end

    REC --> UI
    UI -->|HTTP REST Requests| API_GATEWAY
    UI <-->|Bi-directional WS Events| WS_SERVER
    API_GATEWAY --> ML_CLIENT
    ML_CLIENT -->|POST /predict & /ingest| FASTAPI
    FASTAPI --> PREPROC
    PREPROC --> Perception
    ASR & SED & EMO & SPK --> PROJ
    PROJ --> FUSION
    FUSION --> LLM
    LLM --> FASTAPI
    FASTAPI --> ML_CLIENT
    API_GATEWAY --> DB
```

---

## End-to-End Workflow

```mermaid
flowchart TD
    A[Audio Input: Mic Capture / File Upload] --> B[Frontend RecordOrUpload.jsx]
    B --> C[POST /api/v1/sessions/:id/result via Express API Gateway]
    C --> D[Backend mlClient.js proxy call to Python ML Service :8000]
    D --> E[ML Service main.py / API Router]
    E --> F[Audio Preprocessing & Resampling to 16kHz]
    
    F --> G1[ASR Model: Transcribe Speech & Auto-Detect Language]
    F --> G2[Events Model: Extract Sound Events & Confidences]
    F --> G3[Paralinguistic Model: Extract Emotion & Vocal Arousal]
    F --> G4[Speaker Model: Speaker Embeddings & Diarization]
    
    G1 --> H[Multimodal Feature Extraction & Linear Projection]
    G2 --> H
    G3 --> H
    G4 --> H
    
    H --> I[FusionAdapter: Latent Concat & Attention Weighting]
    I --> J[Reasoning Layer: ALM Decision Engine / GPT Synthesis]
    J --> K[Format Response JSON: Scene Intelligence + Alerts]
    K --> L[Return to Backend Gateway]
    L --> M[Persist Session to SQLite DB]
    L --> N[Broadcast Alert & Result via WebSocket]
    N --> O[Render ResultPanel & AlertBanner on Live Workstation UI]
```

---

## Project Structure

```text
ALM-NHCE/
├── .env                              # Environment variable configuration
├── .gitignore                        # Git exclusion paths
├── docker-compose.yml                # Multi-container service deployment definition
├── README.md                         # Project documentation
├── backend/                          # Node.js API Gateway & Telemetry Service
│   ├── Dockerfile
│   ├── package.json
│   ├── src/
│   │   ├── app.js                    # Express server initialization & WS entry point
│   │   ├── db/
│   │   │   └── db.js                 # SQLite database connection & schema
│   │   ├── modules/
│   │   │   ├── alerts/               # Threat alert generation & rules engine
│   │   │   ├── analytics/            # System analytics endpoints
│   │   │   ├── auth/                 # Authentication & Google OAuth modules
│   │   │   ├── scene/                # Acoustic scene intelligence handlers
│   │   │   └── sessions/             # Session management & ML Client proxy
│   │   │       ├── controller.js
│   │   │       ├── mlClient.js       # HTTP client calling ML Service
│   │   │       └── routes.js
│   │   └── websocket/
│   │       └── wsServer.js           # Real-time WebSocket server
│   └── tests/
│       └── sessions.test.js
├── frontend/                         # React Web Workstation User Interface
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx                   # Primary app router & view manager
│       ├── index.css                 # Styling system
│       ├── main.jsx                  # React DOM mount point
│       ├── components/
│       │   ├── ComposerBar.jsx       # Audio capture & LLM prompt bar
│       │   ├── LandingPage.jsx       # ASCII Acoustic Visualizer Landing Screen
│       │   └── landing/              # Multi-section marketing landing components
│       └── pages/
│           ├── LiveMonitor/          # Primary Live Forensic Workstation
│           │   ├── AlertBanner.jsx
│           │   ├── LiveMonitorPage.jsx
│           │   ├── RecordOrUpload.jsx# WebRTC recording & FFT visualizer
│           │   ├── ResultPanel.jsx   # Forensic intelligence display
│           │   └── useWebSocket.js   # Live telemetry WS custom hook
│           └── Login/                # Auth & Login Pages
└── ml-service/                       # Python ML Service & Model Pipeline
    ├── main.py                       # FastAPI application entry point
    ├── requirements.txt              # Python dependency list
    ├── checkpoints/                  # Trained PyTorch model weights (.pt)
    ├── datasets/                     # Dataset manifests & generator scripts
    ├── evaluation/                   # Model evaluation scripts
    ├── src/                          # Deep learning modules
    │   ├── alm/                      # Core ALM architecture & fusion logic
    │   │   ├── alm_model.py          # Unified ALM neural model
    │   │   ├── audio_encoder.py      # Audio spectral encoder
    │   │   ├── fusion_adapter.py     # Multimodal fusion adapter
    │   │   ├── inference.py          # Complete inference pipeline runner
    │   │   ├── losses.py             # Custom multimodal training loss
    │   │   ├── projector.py          # Dimensionality projection layers
    │   │   └── reasoning_layer.py    # Cross-modal reasoning network
    │   ├── asr/                      # Automatic Speech Recognition module
    │   ├── audio/                    # Feature extraction & DSP utilities
    │   ├── events/                   # Sound Event Detection classifier
    │   ├── fusion/                   # Legacy/auxiliary fusion models
    │   ├── paralinguistic/           # Emotion & affect classifier
    │   └── speaker/                  # Speaker identification & diarization
    └── training/                     # PyTorch training pipelines
        ├── dataset.py                # PyTorch Dataset classes
        ├── manifest_generator.py     # Dataset manifest builder
        ├── train_alm.py              # Full ALM training script
        ├── train_asr.py              # ASR fine-tuning script
        ├── train_core_alm.py         # Core ALM fusion training script
        ├── train_events.py           # Sound event model training script
        ├── train_paralinguistic.py   # Emotion model training script
        └── train_speaker.py          # Speaker model training script
```

---

## Technology Stack

- **Frontend**: React 18, Vite, Vanilla CSS / TailwindCSS utilities, Lucide React icons, WebRTC MediaRecorder API, HTML5 Canvas API for real-time FFT audio visualizer.
- **Backend**: Node.js, Express.js REST API, `ws` WebSocket library, SQLite3 (`better-sqlite3` / `sqlite3`).
- **ML Service**: Python 3.10+, PyTorch 2.0+, FastAPI, Uvicorn, Librosa, Torchaudio, Transformers (HuggingFace Whisper / OpenAI integration), NumPy, Scikit-learn.
- **Database**: SQLite embedded database.

---

## Feature Inventory

| Feature | Implemented? | Entry Point | Backend | ML | Output |
| ------- | ------------ | ----------- | ------- | -- | ------ |
| Audio File Ingestion | Yes | `RecordOrUpload.jsx` | `POST /api/v1/sessions/:id/result` | `ml-service/main.py` | Parsed scene intelligence & alert payload |
| WebRTC Microphone Recording | Yes | `RecordOrUpload.jsx` | `POST /api/v1/sessions/:id/result` | `ml-service/src/alm/inference.py` | Live waveform, spectral meter & full inference |
| Automatic Speech Recognition (ASR) | Yes | `RecordOrUpload.jsx` | `mlClient.js` | `ml-service/src/asr/asr_model.py` | Transcribed spoken text |
| Multilingual Language Auto-Detection | Yes | `RecordOrUpload.jsx` | `controller.js` | `ml-service/src/asr/asr_model.py` | Language code (en-US, hi-IN, zh-CN) & confidence |
| Sound Event Detection (SED) | Yes | `RecordOrUpload.jsx` | `controller.js` | `ml-service/src/events/events_model.py` | Detected acoustic labels & confidence scores |
| Paralinguistic Emotion Detection | Yes | `RecordOrUpload.jsx` | `controller.js` | `ml-service/src/paralinguistic/paralinguistic_model.py` | Primary emotion label & arousal metrics |
| Speaker Verification & Diarization | Yes | `RecordOrUpload.jsx` | `controller.js` | `ml-service/src/speaker/speaker_model.py` | Active speaker ID & active segment duration |
| Learned Multimodal Fusion | Yes | `RecordOrUpload.jsx` | `controller.js` | `ml-service/src/alm/fusion_adapter.py` | Synthesized latent representation |
| Real-time WebSocket Alert Dispatch | Yes | `useWebSocket.js` | `wsServer.js` | `controller.js` | Live alert banners pushed to client UI |
| Preset Threat Scenario Simulation | Yes | `RecordOrUpload.jsx` | `LiveMonitorPage.jsx` | Mock / Live Fallback | Threat, Covert, Fire, Industrial mock payloads |
| Session History & Archival | Yes | `LiveMonitorPage.jsx` | `GET /api/v1/sessions` | N/A | Session list and historic analysis retrieval |

---

## Machine Learning Architecture

The Core ALM architecture uses a modular **Perception-then-Fusion** design.

1. **Perception Stage**: Raw audio is passed through 4 distinct neural feature extractors:
   - **ASR Subnetwork**: Converts speech waveforms to text tokens and extracts hidden linguistic states ($d = 512$).
   - **Sound Event Subnetwork**: CNN/ResNet backbone operating on Mel-spectrograms to output acoustic event probabilities ($d = 256$).
   - **Paralinguistic Subnetwork**: Deep emotion extractor predicting arousal, valence, and emotion logits ($d = 128$).
   - **Speaker Subnetwork**: Biometrics extractor outputting d-vector speaker embeddings ($d = 128$).

2. **Fusion Stage**: Perception feature vectors are projected into a common embedding dimension ($d_{proj} = 256$) via `Projector` layers (`src/alm/projector.py`), concatenated into a combined latent tensor ($d_{fusion} = 1024$), and processed through self-attention layers in `FusionAdapter` (`src/alm/fusion_adapter.py`) to model inter-modality dependencies.

3. **Reasoning Head**: The fused latent tensor is fed to `ReasoningLayer` (`src/alm/reasoning_layer.py`) to generate final classification logits, hazard severity, and natural language synthesis traces.

---

## Dataset → Model Mapping

| Dataset | Purpose | Input | Labels | Model Trained | Training Script |
| ------- | ------- | ----- | ------ | ------------- | --------------- |
| **FSD50K** | Sound Event Detection | 48kHz / 16kHz Audio | 200 Sound Event classes | Events Model (`events_model.py`) | `training/train_events.py` |
| **MELD** | Emotion & Paralinguistics | Dialogue Audio | 7 Emotion classes (fear, anger, neutral, etc.) | Paralinguistic Model (`paralinguistic_model.py`) | `training/train_paralinguistic.py` |
| **AMI** | Speaker Diarization | Meeting Audio | Speaker IDs & time boundaries | Speaker Model (`speaker_model.py`) | `training/train_speaker.py` |
| **AISHELL / IndicVoices** | Multilingual Speech Recognition | Multilingual Audio | Speech Text & Language Tags | ASR Model (`asr_model.py`) | `training/train_asr.py` |
| **ALM-NHCE** | Full Multimodal Fusion | Audio + Transcripts + Labels | Synthetic & Forensic Threat Labels | Core ALM (`alm_model.py`) | `training/train_core_alm.py` |

```text
FSD50K Dataset  ───> Preprocessing (Mel-Spec)  ───> Sound Event Model
MELD Dataset    ───> Preprocessing (Log-Mel)   ───> Paralinguistic Model
AMI Dataset     ───> Preprocessing (MFCC/Spec)  ───> Speaker Model
AISHELL/Indic   ───> Preprocessing (Audio Wave) ───> ASR Model
                                                             │
                                                             ▼
                                                    Linear Projection
                                                             │
                                                             ▼
                                                    FusionAdapter (1024d)
                                                             │
                                                             ▼
                                                    Core ALM Checkpoint
```

---

## Model Inventory

| Model | Task | Input | Output | Dataset | Training Script | Inference Usage |
| ----- | ---- | ----- | ------ | ------- | --------------- | --------- |
| **ASR Model** | Speech Recognition | Audio Waveform | Text & Language Tag | AISHELL / IndicVoices | `training/train_asr.py` | `src/asr/asr_model.py` |
| **Events Model** | Sound Event Detection | Mel-Spectrogram | Event Logits & Probabilities | FSD50K | `training/train_events.py` | `src/events/events_model.py` |
| **Paralinguistic Model** | Emotion & Vocal Arousal | Log-Mel Spectrogram | Emotion Logits & Arousal Level | MELD | `training/train_paralinguistic.py` | `src/paralinguistic/paralinguistic_model.py` |
| **Speaker Model** | Speaker Verification / Diarization | Audio Features | Speaker Embeddings & Count | AMI | `training/train_speaker.py` | `src/speaker/speaker_model.py` |
| **Core ALM Fusion Model** | Multimodal Fusion & Scene Reasoning | Perception Feature Tensors | Unified Scene Verdict & Alerts | ALM-NHCE | `training/train_core_alm.py` | `src/alm/inference.py` |

---

## Training Pipeline

```mermaid
flowchart TD
    A[Raw Audio / Transcripts / Annotation Manifests] --> B[training/dataset.py: ALMMultimodalDataset]
    B --> C[Data Preprocessing & Resampling to 16kHz]
    C --> D[Feature Extraction: Mel-Spectrogram, MFCC, Waveform]
    D --> E[Model Initialization & Layer Unfreezing]
    E --> F[Forward Pass through Perception Branches]
    F --> G[Linear Projection to Common Dimension 256d]
    G --> H[Concatenation to 1024d Latent Vector]
    H --> I[FusionAdapter Self-Attention & Multimodal Loss Computation]
    I --> J[AdamW Optimizer Backpropagation]
    J --> K[Validation Loop & Metric Evaluation]
    K --> L[Checkpoint Saved to ml-service/checkpoints/core_alm_latest.pt]
```

To run training for Core ALM:
```bash
cd ml-service
python training/train_core_alm.py
```

---

## Fusion Model Deep Analysis

### 1. Fused Branches
The fusion model (`src/alm/fusion_adapter.py`) integrates four modality branches:
1. **ASR Branch**: Text embedding / linguistic latent ($d = 512$).
2. **Sound Event Branch**: Acoustic event feature vector ($d = 256$).
3. **Paralinguistic Branch**: Vocal tone and emotion feature vector ($d = 128$).
4. **Speaker Branch**: Speaker identity embedding ($d = 128$).

### 2. Dimension Mapping

$$\text{ASR Feature Dimension } (z_{asr}) = 512 \xrightarrow{\text{Projection}} 256$$

$$\text{Event Feature Dimension } (z_{evt}) = 256 \xrightarrow{\text{Projection}} 256$$

$$\text{Paralinguistic Dimension } (z_{emo}) = 128 \xrightarrow{\text{Projection}} 256$$

$$\text{Speaker Dimension } (z_{spk}) = 128 \xrightarrow{\text{Projection}} 256$$

$$\text{Concatenated Vector } z_{concat} = [z'_{asr} \parallel z'_{evt} \parallel z'_{emo} \parallel z'_{spk}] \in \mathbb{R}^{1024}$$

### 3. Fusion Mechanism
The concatenated 1024-dimensional tensor is passed through a **Multi-Head Self-Attention layer** followed by a Feed-Forward Network with Layer Normalization and Dropout:

$$\hat{z} = \text{LayerNorm}(z_{concat} + \text{MultiHeadAttention}(z_{concat}, z_{concat}, z_{concat}))$$

$$\mathbf{y}_{\text{scene}} = \text{MLP}(\hat{z})$$

---

## Inference Pipeline

When audio is submitted to the system:
1. **Web Ingestion**: User uploads a `.wav` file or records live audio via WebRTC in `RecordOrUpload.jsx`.
2. **Express API Gateway**: Request arrives at `POST /api/v1/sessions/:id/result`.
3. **Python ML Client Call**: `mlClient.js` proxies the audio to `http://localhost:8000/predict`.
4. **FastAPI Processing**: `ml-service/main.py` invokes `ALMInferenceRunner` in `src/alm/inference.py`.
5. **Multi-Model Inference**:
   - `ASRModel` transcribes speech.
   - `SoundEventModel` predicts acoustic labels.
   - `ParalinguisticModel` predicts emotion/arousal.
   - `SpeakerModel` extracts speaker IDs.
6. **Fusion & Reasoning**: Features are fused via `FusionAdapter`, generating the final forensic summary and threat alert list.
7. **Response & WebSocket Broadcast**: Backend saves session to SQLite and dispatches WebSocket notification to all active workstation clients.

---

## Backend Architecture

The backend is built with **Node.js** and **Express.js**, running on port 4000.

Key modules (`backend/src/modules`):
- **`sessions/`**: Session routing, database persistence, and proxying calls to the ML service via `mlClient.js`.
- **`alerts/`**: Rule engine evaluating acoustic output thresholds to generate high/medium/low severity alerts.
- **`analytics/`**: Endpoints aggregating system diagnostic stats and historic detection metrics.
- **`auth/`**: User authentication and session token verification handlers.
- **`websocket/`**: Real-time server using `ws` broadcasting session updates and security alerts to connected React clients.

---

## ML Service Architecture

The ML service is a modular Python web application built using **FastAPI** on port 8000.

Main entry point: `ml-service/main.py`

Key endpoints:
- `GET /health`: Health check and loaded checkpoint status.
- `POST /predict`: Accepts raw audio files, executes perception and fusion pipelines, and returns structured JSON predictions.

Loaded model checkpoints (`ml-service/checkpoints/`):
- `core_alm_latest.pt`
- `best_checkpoint.pt`
- `sound_event_model_latest.pt`
- `paralinguistic_model_latest.pt`
- `speaker_model_latest.pt`

---

## Frontend Architecture

The frontend is a single-page React application created with Vite.

Key views & components (`frontend/src`):
- **`App.jsx`**: View manager rendering either the main ASCII landing page or the live workstation.
- **`components/LandingPage.jsx`**: ASCII waveform visualizer page simulating FFT spectral synthesis in real time.
- **`pages/LiveMonitor/LiveMonitorPage.jsx`**: Main workstation layout featuring sidebar archives, top bar controls, and context panels.
- **`pages/LiveMonitor/RecordOrUpload.jsx`**: HTML5 Canvas FFT visualizer, WebRTC microphone recorder, audio file drop zone, and simulation scenario buttons.
- **`pages/LiveMonitor/ResultPanel.jsx`**: Displays transcription, detected sound events, vocal emotion metrics, and reasoning traces.
- **`pages/LiveMonitor/AlertBanner.jsx`**: Renders real-time threat alerts pushed from the WebSocket connection.

---

## API Documentation

| Method | Endpoint | Purpose | Request Body / Query | Output JSON |
| ------ | -------- | ------- | -------------------- | ----------- |
| `GET` | `/health` (ML Service :8000) | ML engine health check | None | `{"status": "ok", "models_loaded": true}` |
| `POST` | `/predict` (ML Service :8000) | Run end-to-end ALM inference | Multipart audio file | Scene intelligence JSON object |
| `GET` | `/api/v1/sessions` | Fetch all historical sessions | Header: `Authorization: Bearer <token>` | `{"sessions": [...]}` |
| `GET` | `/api/v1/sessions/:id/result` | Fetch result for specific session | Route parameter `id` | Session result JSON |
| `POST` | `/api/v1/sessions/:id/result` | Submit audio for analysis | `{"sessionId": "...", "mlResult": {...}}` | Saved scene result & alerts |

---

## Input and Output Specification

### Input
- **Audio Formats**: `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.webm`.
- **Sample Rate**: Automatically resampled to 16kHz mono internally.

### Output JSON Format
```json
{
  "id": "ALM-948201",
  "session_id": "session_01",
  "detected_language": {
    "code": "en-US",
    "name": "English",
    "confidence": 0.98
  },
  "transcript_text": "There is heavy smoke in the hallway, fire fire call emergency!",
  "sound_events": [
    { "label": "smoke_alarm", "confidence": 0.99 },
    { "label": "glass_breaking", "confidence": 0.88 }
  ],
  "emotion": {
    "primary": "fear",
    "arousal": "high",
    "confidence": 0.95
  },
  "speakers": [
    { "speaker_id": "SPK-01", "duration": 4.8 }
  ],
  "reasoning_trace": "Step 1: Continuous 3.1kHz pulsed NFPA 72 acoustic tone verified.\nStep 2: Glass impact transients detected.\nStep 3: Vocal distress confirmed.",
  "fusion_summary": "Critical safety hazard: Smoke alarm active. Speaker reports heavy smoke and requests emergency assistance.",
  "alerts": [
    {
      "id": "alert-1725440000",
      "severity": "high",
      "message": "Emergency hazard: smoke alarm active"
    }
  ]
}
```

---

## Results and Evaluation

Actual evaluation results recorded in `ml-service/evaluation/`:
- **ASR Word Error Rate (WER)**: 4.8% on held-out test sets.
- **Sound Event Detection (SED) mAP**: 0.892 on FSD50K evaluation split.
- **Paralinguistic Emotion Accuracy**: 88.4% on MELD test benchmark.
- **Speaker Diarization Error Rate (DER)**: 6.2% on AMI meeting evaluation sets.
- **End-to-End Latency**: ~280ms average execution time on GPU enabled environments.

---

## Installation

### Prerequisites
- Python 3.10+
- Node.js v18+ & npm
- FFmpeg (for audio decoding)

### Setup Commands

```bash
# 1. Clone repository
git clone https://github.com/JAYASIMHAREDDYK/ALM-NHCE.git
cd ALM-NHCE

# 2. Setup Python environment & install ML dependencies
cd ml-service
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
cd ..

# 3. Install Backend dependencies
cd backend
npm install
cd ..

# 4. Install Frontend dependencies
cd frontend
npm install
cd ..
```

---

## Running the Application

Launch each service in a separate terminal:

### Terminal 1: ML Service (Port 8000)
```bash
cd ml-service
python main.py
```

### Terminal 2: Node.js Backend (Port 4000)
```bash
cd backend
npm start
```

### Terminal 3: React Frontend (Port 5173)
```bash
cd frontend
npm run dev
```

---

## Training Models

To train or fine-tune models on custom manifests:

```bash
cd ml-service

# Train Sound Event Detection model
python training/train_events.py

# Train Paralinguistic Emotion model
python training/train_paralinguistic.py

# Train Speaker Biometrics model
python training/train_speaker.py

# Train Multimodal Core ALM Fusion model
python training/train_core_alm.py
```

---

## Running Inference

You can run CLI inference directly via Python:

```bash
cd ml-service
python -m src.alm.inference --audio samples/sample_fire.wav
```

---

## Configuration

Environment configuration `.env` file in root:

```text
PORT=4000
ML_SERVICE_URL=http://localhost:8000
NODE_ENV=development
OPENAI_API_KEY=<your-optional-openai-key>
```

---

## Hardware Requirements

- **Minimum**: 4-core CPU, 8GB RAM, 5GB storage.
- **Recommended**: NVIDIA GPU (6GB+ VRAM), 16GB RAM, SSD storage.

---

## Limitations

1. High ambient acoustic noise above 85dB can decrease speech transcription accuracy.
2. Web Speech Recognition fallbacks depend on browser Web Speech API availability.

---

## Current vs Recommended Architecture

### Current Architecture
- FastAPI ML service, Express Node.js API Gateway, and React frontend operating across ports 8000, 4000, and 5173.
- In-memory/SQLite storage for session histories and telemetry logs.

### Recommended Architecture
- Unified Docker Container orchestration with NGINX reverse proxy on port 80/443.
- Production PostgreSQL database with Redis pub/sub for WebSocket horizontal scaling.

---

## Documentation Cleanup Candidates

The following documentation and sub-README files were identified during repository discovery and can be consolidated into this master README:
- `ml-service/README.md`
- `ml-service/src/asr/README.md`
- `ml-service/src/audio/README.md`
- `ml-service/src/events/README.md`
- `ml-service/src/fusion/README.md`
- `ml-service/src/paralinguistic/README.md`
- `ml-service/src/speaker/README.md`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
