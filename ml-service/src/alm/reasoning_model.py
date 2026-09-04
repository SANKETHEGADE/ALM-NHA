import torch
import torch.nn as nn
import torch.nn.functional as F

class CoreALMReasoningModel(nn.Module):
    """
    Core ALM Question-Conditioned Multimodal Reasoning Engine.
    Processes joint spatio-temporal audio representations + question embedding sequence.
    Predicts:
    1. Natural language answer representation / scene synthesis classification
    2. Evidence item grounding weights
    3. Scalar confidence score (calibrated 0.0 - 1.0)
    """
    def __init__(self, embed_dim=256, vocab_size=5000, num_layers=3, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # Deep Multimodal Decoder / Reasoning Transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.reasoning_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 1. Answer Output Head
        self.answer_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, vocab_size)
        )
        
        # 2. Evidence Grounding Head (Frame-level relevance / modality attribution)
        self.evidence_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 3. Confidence Head (Calibrated scalar predictor)
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, joint_context: torch.Tensor) -> dict:
        """
        Args:
            joint_context: (B, Seq_len, embed_dim)
        Returns:
            dict containing:
                - answer_logits: (B, Seq_len, vocab_size)
                - pooled_rep: (B, embed_dim)
                - evidence_scores: (B, Seq_len)
                - confidence: (B, 1)
        """
        # Pass dummy target or self-attention sequence
        tgt = joint_context
        reasoned = self.reasoning_decoder(tgt, joint_context)
        
        # Global pooled representation (mean over tokens)
        pooled_rep = torch.mean(reasoned, dim=1) # (B, embed_dim)

        answer_logits = self.answer_head(reasoned) # (B, Seq_len, vocab_size)
        evidence_scores = self.evidence_head(reasoned).squeeze(-1) # (B, Seq_len)
        confidence = self.confidence_head(pooled_rep) # (B, 1)

        return {
            "answer_logits": answer_logits,
            "pooled_rep": pooled_rep,
            "evidence_scores": evidence_scores,
            "confidence": confidence
        }
