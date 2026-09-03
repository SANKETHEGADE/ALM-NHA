import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, Wav2Vec2Model, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ConvFeatureExtractor(nn.Module):
    """
    1D Convolutional temporal frontend to extract acoustic features from raw 16kHz audio.
    Downsamples 16000 Hz raw audio to ~50 Hz frame rate (stride 320 total).
    """

    def __init__(self, in_channels: int = 1, out_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=10, stride=5, padding=3)
        self.gn1 = nn.GroupNorm(16, 128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=6, stride=4, padding=2)
        self.gn2 = nn.GroupNorm(32, 256)
        self.conv3 = nn.Conv1d(256, 384, kernel_size=4, stride=4, padding=1)
        self.gn3 = nn.GroupNorm(48, 384)
        self.conv4 = nn.Conv1d(384, out_dim, kernel_size=4, stride=4, padding=1)
        self.gn4 = nn.GroupNorm(64, out_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, T) or (B, T)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        
        x = self.activation(self.gn1(self.conv1(x)))
        x = self.activation(self.gn2(self.conv2(x)))
        x = self.activation(self.gn3(self.conv3(x)))
        x = self.activation(self.gn4(self.conv4(x)))
        # Output shape: (B, out_dim, T_frames)
        # Permute to (B, T_frames, out_dim)
        return x.transpose(1, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            return x
        return x + self.pe[:, :seq_len, :].to(x.device)


class StandaloneSpeechEncoder(nn.Module):
    """
    Lightweight Conformer/Transformer Non-Whisper Speech Encoder.
    Operates offline without requiring external network access.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.frontend = ConvFeatureExtractor(in_channels=input_dim, out_dim=hidden_dim)
        self.pos_enc = PositionalEncoding(d_model=hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Temporal attention pooling for global representation
        self.pool_att = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            audio: (Batch, Num_Samples) or (Batch, 1, Num_Samples)
        Returns:
            frame_embeddings: (Batch, Time_Frames, hidden_dim)
            pooled_embedding: (Batch, hidden_dim)
        """
        feats = self.frontend(audio)
        feats = self.pos_enc(feats)
        frame_embeddings = self.transformer(feats)
        frame_embeddings = self.layer_norm(frame_embeddings)

        # Compute attentive global pooling
        att_weights = F.softmax(self.pool_att(frame_embeddings), dim=1)  # (B, T, 1)
        pooled_embedding = torch.sum(frame_embeddings * att_weights, dim=1)  # (B, D)

        return frame_embeddings, pooled_embedding


class SpeechEncoder(nn.Module):
    """
    Unified Non-Whisper Speech Encoder supporting both MMS/Wav2Vec2 backbones
    and self-contained Standalone Transformer architectures.
    """

    def __init__(
        self,
        pretrained_name_or_path: Optional[str] = None,
        hidden_dim: int = 512,
        freeze_feature_extractor: bool = False
    ):
        super().__init__()
        self.pretrained_name = pretrained_name_or_path
        self.use_hf = False
        self.hidden_dim = hidden_dim

        if pretrained_name_or_path and TRANSFORMERS_AVAILABLE:
            try:
                self.hf_model = AutoModel.from_pretrained(pretrained_name_or_path)
                self.hidden_dim = getattr(self.hf_model.config, "hidden_size", hidden_dim)
                self.use_hf = True
                if freeze_feature_extractor and hasattr(self.hf_model, "freeze_feature_encoder"):
                    self.hf_model.freeze_feature_encoder()
            except Exception:
                # Fall back gracefully to standalone encoder
                self.use_hf = False

        if not self.use_hf:
            self.backbone = StandaloneSpeechEncoder(hidden_dim=self.hidden_dim)

    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract speech representations.
        Returns:
            frame_embeddings: (Batch, T, Hidden_Dim)
            pooled_embedding: (Batch, Hidden_Dim)
        """
        if self.use_hf:
            # HuggingFace Wav2Vec2/MMS encoder
            if audio.ndim == 3 and audio.shape[1] == 1:
                audio = audio.squeeze(1)
            outputs = self.hf_model(audio, attention_mask=attention_mask)
            frame_embeddings = outputs.last_hidden_state
            # Mean pooling over time dimension
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(frame_embeddings)
                sum_embeddings = torch.sum(frame_embeddings * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled_embedding = sum_embeddings / sum_mask
            else:
                pooled_embedding = torch.mean(frame_embeddings, dim=1)
            return frame_embeddings, pooled_embedding
        else:
            return self.backbone(audio, attention_mask=attention_mask)
