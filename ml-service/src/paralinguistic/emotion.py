import torch
import torch.nn as nn
import torch.nn.functional as F

MELD_EMOTION_CLASSES = ["neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted"]

class EmotionClassifier(nn.Module):
    """
    Paralinguistic Emotion Classifier (trained on MELD audio subset).
    Classifies emotions strictly into official MELD classes:
    ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised', 'disgusted'].
    """
    def __init__(self, embed_dim=256, num_classes=len(MELD_EMOTION_CLASSES)):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.classes = MELD_EMOTION_CLASSES
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        self.arousal_head = nn.Linear(embed_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> dict:
        """
        Args:
            embeddings: (B, T, embed_dim) continuous paralinguistic embeddings
        Returns:
            dict containing emotion_logits, emotion_probs, arousal
        """
        pooled = torch.mean(embeddings, dim=1) # (B, embed_dim)
        logits = self.classifier(pooled) # (B, num_classes)
        probs = F.softmax(logits, dim=-1)
        arousal = torch.sigmoid(self.arousal_head(pooled))

        return {
            "logits": logits,
            "probs": probs,
            "arousal": arousal
        }

    def predict(self, embeddings: torch.Tensor) -> dict:
        out = self.forward(embeddings)
        probs = out["probs"][0] # (num_classes,)
        top_idx = torch.argmax(probs).item()
        
        emotion = self.classes[top_idx]
        confidence = float(probs[top_idx].cpu().item())
        arousal_val = float(out["arousal"][0].cpu().item())

        return {
            "emotion": emotion,
            "confidence": round(max(0.50, min(0.99, confidence)), 2),
            "arousal": "high" if arousal_val > 0.70 else ("medium" if arousal_val > 0.40 else "low"),
            "speaking_style": "rapid" if emotion in ["angry", "fearful"] else ("deliberate" if emotion == "sad" else "conversational"),
            "urgency": "high" if emotion in ["fearful", "angry"] else "low"
        }
