from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.asr.tokenizer import MultilingualTokenizer, BLANK_TOKEN


class CTCDecoder(nn.Module):
    """
    Multilingual CTC Projection and Decoding Head.
    Maps acoustic frame embeddings to multilingual vocabulary probabilities.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        dropout: float = 0.1,
        blank_id: int = 2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.blank_id = blank_id

        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        # PyTorch CTC Loss
        self.ctc_loss = nn.CTCLoss(
            blank=self.blank_id,
            reduction="mean",
            zero_infinity=True
        )

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute logits over vocabulary.
        Args:
            frame_embeddings: (Batch, Time_Frames, Hidden_Dim)
        Returns:
            logits: (Batch, Time_Frames, Vocab_Size)
        """
        x = self.dropout(frame_embeddings)
        logits = self.lm_head(x)
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Connectionist Temporal Classification (CTC) loss.
        """
        # PyTorch CTC loss expects log_probs shape: (Time, Batch, Vocab_Size)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

    def decode_greedy(
        self,
        logits: torch.Tensor,
        tokenizer: MultilingualTokenizer
    ) -> List[Dict[str, Any]]:
        """
        Greedy CTC decoding on logits:
        1. argmax over vocab dimension
        2. collapse repeated tokens
        3. remove blank tokens
        4. compute frame confidence scores
        """
        probs = F.softmax(logits, dim=-1)  # (Batch, T, Vocab_Size)
        max_probs, preds = torch.max(probs, dim=-1)  # (Batch, T), (Batch, T)

        batch_results = []
        batch_size = logits.size(0)

        for b in range(batch_size):
            seq_preds = preds[b].tolist()
            seq_probs = max_probs[b].tolist()

            collapsed_ids: List[int] = []
            collapsed_probs: List[float] = []
            prev_token = None

            for token_id, p in zip(seq_preds, seq_probs):
                if token_id != prev_token:
                    if token_id != self.blank_id:
                        collapsed_ids.append(token_id)
                        collapsed_probs.append(p)
                    prev_token = token_id

            text = tokenizer.decode(collapsed_ids, skip_special_tokens=True)
            avg_conf = float(sum(collapsed_probs) / max(1, len(collapsed_probs))) if collapsed_probs else 0.0

            batch_results.append({
                "text": text,
                "token_ids": collapsed_ids,
                "confidence": round(avg_conf, 4),
            })

        return batch_results
