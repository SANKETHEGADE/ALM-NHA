import torch
import torch.nn as nn
from src.alm.audio_encoder import AudioEncoder
from src.alm.projector import ModalityProjector
from src.alm.fusion_adapter import ALMFusionAdapter
from src.alm.reasoning_model import CoreALMReasoningModel
from src.fusion.temporal_alignment import TemporalAligner
from src.fusion.multimodal_projection import MultimodalProjector
from src.fusion.temporal_fusion import TemporalFusionAdapter

from src.asr.asr_model import ASRModel
from src.events.event_model import SoundEventModel
from src.speaker.speaker_model import SpeakerModel
from src.paralinguistic.paralinguistic_model import ParalinguisticModel

class CoreALM(nn.Module):
    """
    Core Audio Language Model (Core ALM).
    Accepts:
        - raw audio / mel-spectrogram
        - user question
    Or feeds representations from specialized submodels:
        - ASR Model (Lahari)
        - Sound Event Model (Anoop)
        - Speaker Model (Sanket)
        - Paralinguistic Model
    Pipes representations into:
        1. Temporal Alignment
        2. Multimodal Fusion Adapter
        3. Core ALM Question-Conditioned Reasoning Head
    Outputs:
        - Natural language answer logits / tokens
        - Evidence temporal grounding
        - Scalar confidence score
    """
    def __init__(self, embed_dim=256, vocab_size=5000):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Base Specialized Audio Modality Encoders
        self.audio_encoder = AudioEncoder(embed_dim=embed_dim)
        self.asr_model = ASRModel(embed_dim=embed_dim)
        self.event_model = SoundEventModel(embed_dim=embed_dim)
        self.speaker_model = SpeakerModel(embed_dim=embed_dim)
        self.para_model = ParalinguisticModel(embed_dim=embed_dim)

        # 2. Temporal Alignment & Multimodal Fusion Pipeline
        self.temporal_aligner = TemporalAligner(target_len=64, embed_dim=embed_dim)
        self.multimodal_projector = MultimodalProjector(fusion_dim=embed_dim)
        self.temporal_fusion_adapter = TemporalFusionAdapter(embed_dim=embed_dim)

        # 3. Core ALM Projector & Question Fusion Adapter
        self.modality_projector = ModalityProjector(fused_dim=embed_dim, question_dim=embed_dim, model_dim=embed_dim)
        self.alm_fusion_adapter = ALMFusionAdapter(embed_dim=embed_dim)

        # 4. Core ALM Question-Conditioned Reasoning Head
        self.reasoning_model = CoreALMReasoningModel(embed_dim=embed_dim, vocab_size=vocab_size)

    def forward(self, audio_tensor: torch.Tensor, question_text: str = "Where is the speaker likely to be?") -> dict:
        """
        Full End-to-End Forward Pass.
        Args:
            audio_tensor: (B, 1, F, T) or (B, T_samples)
            question_text: user question string
        Returns:
            dict containing reasoning output (answer_logits, pooled_rep, evidence_scores, confidence)
            plus submodel predictions (transcript, events, diarization, paralinguistics).
        """
        device = audio_tensor.device

        # Step 1: Extract Specialized Submodel Embeddings
        speech_emb = self.asr_model.encode(audio_tensor)       # (B, T_s, D)
        event_emb = self.event_model.encode(audio_tensor)       # (B, T_e, D)
        speaker_emb = self.speaker_model.encode(audio_tensor)   # (B, T_sp, D)
        para_emb = self.para_model.encode(audio_tensor)         # (B, T_p, D)
        audio_emb = self.audio_encoder(audio_tensor)            # (B, T_a, D)

        # Step 2: Temporal Alignment across all modalities + timestamp encodings
        aligned_dict = self.temporal_aligner(
            speech_emb=speech_emb,
            event_emb=event_emb,
            speaker_emb=speaker_emb,
            para_emb=para_emb,
            audio_emb=audio_emb
        )

        # Step 3: Multimodal Projection
        projected_dict = self.multimodal_projector(
            speech_emb=aligned_dict["speech"],
            event_emb=aligned_dict["events"],
            speaker_emb=aligned_dict["speakers"],
            para_emb=aligned_dict["paralinguistic"],
            audio_emb=aligned_dict["audio"]
        )

        # Step 4: Temporal Multimodal Fusion
        fused_multimodal = self.temporal_fusion_adapter(projected_dict) # (B, T_aligned, D)

        # Step 5: Question Embedding & Core ALM Projection
        q_emb = self.modality_projector.embed_question_text(question_text, device=device) # (1, L_q, D)
        if audio_tensor.shape[0] > 1:
            q_emb = q_emb.repeat(audio_tensor.shape[0], 1, 1)

        proj_fused, proj_question = self.modality_projector(fused_multimodal, q_emb)

        # Step 6: Question-Conditioned Cross-Attention Fusion
        joint_context = self.alm_fusion_adapter(proj_fused, proj_question) # (B, L_q + T_aligned, D)

        # Step 7: Reasoning & Answer Generation
        reasoning_output = self.reasoning_model(joint_context)

        # Submodel structured extractions
        asr_out = self.asr_model.transcribe(audio_tensor)
        events_out = self.event_model.detect(audio_tensor)
        speaker_out = self.speaker_model.diarize(audio_tensor)
        para_out = self.para_model.analyze(audio_tensor)

        reasoning_output.update({
            "speech": asr_out,
            "audio_events": events_out,
            "speakers": speaker_out,
            "paralinguistic": para_out
        })

        return reasoning_output
