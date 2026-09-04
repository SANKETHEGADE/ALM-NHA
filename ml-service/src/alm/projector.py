import torch
import torch.nn as nn

class ModalityProjector(nn.Module):
    """
    Core ALM Modality Projector.
    Projects fused multimodal sequence representations and question embeddings into the reasoning space.
    """
    def __init__(self, fused_dim=256, question_dim=256, model_dim=256):
        super().__init__()
        self.multimodal_proj = nn.Sequential(
            nn.Linear(fused_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU()
        )
        self.question_proj = nn.Sequential(
            nn.Linear(question_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU()
        )
        # Text embedding lookup for question tokenization
        self.vocab_size = 5000
        self.text_embedding = nn.Embedding(self.vocab_size, question_dim)

    def embed_question_text(self, question_text: str, device=None) -> torch.Tensor:
        """
        Simple lightweight character/word-hash tokenizer for question text.
        """
        words = question_text.lower().split()
        token_ids = [hash(w) % (self.vocab_size - 1) + 1 for w in words]
        if not token_ids:
            token_ids = [0]
        ids_tensor = torch.tensor([token_ids], device=device or torch.device('cpu'))
        embeds = self.text_embedding(ids_tensor) # (1, L_q, D)
        return embeds

    def forward(self, fused_tensor: torch.Tensor, question_tensor: torch.Tensor) -> tuple:
        """
        Args:
            fused_tensor: (B, T_fused, fused_dim)
            question_tensor: (B, L_q, question_dim)
        Returns:
            proj_fused: (B, T_fused, model_dim)
            proj_question: (B, L_q, model_dim)
        """
        proj_fused = self.multimodal_proj(fused_tensor)
        proj_question = self.question_proj(question_tensor)
        return proj_fused, proj_question
