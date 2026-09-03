import torch
import os
import sys

# Add parent directory to path to import features
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import PerceptionPreprocessor, SITUATION_CLASSES
from train import SituationClassifierMLP

class SituationPredictor:
    """
    Singleton wrapper for the trained PyTorch model.
    Loads the model weights and preprocessor into memory once, providing a fast, thread-safe inference interface.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "artifact", "model.pt")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.preprocessor = checkpoint['preprocessor']
            self.input_dim = checkpoint['input_dim']
            self.num_classes = checkpoint['num_classes']
            
            self.model = SituationClassifierMLP(self.input_dim, self.num_classes)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.to(self.device)
            self.model.eval()
            self.is_ready = True
        except Exception as e:
            print(f"Warning: Failed to load model from {model_path}. Error: {e}")
            self.is_ready = False

    def predict(self, perception_output: dict) -> dict:
        """
        Contract input:
        {
            "transcript": str,
            "sound_events": list[str],
            "emotion": str,
            "speakers": int
        }
        """
        if not self.is_ready:
            return {"label": "unknown", "confidence": 0.0}
            
        transcript = [perception_output.get("transcript", "")]
        sound_events = [perception_output.get("sound_events", [])]
        emotion = [perception_output.get("emotion", "neutral")]
        speakers = [perception_output.get("speakers", 1)]
        
        try:
            X = self.preprocessor.transform(transcript, sound_events, emotion, speakers)
            X = X.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(X)
                probabilities = torch.softmax(outputs, dim=1)
                
                confidence, pred_class = torch.max(probabilities, dim=1)
                
                label = SITUATION_CLASSES[pred_class.item()]
                conf = confidence.item()
                
            return {
                "label": label,
                "confidence": round(conf, 4)
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return {"label": "unknown", "confidence": 0.0}

# Pre-initialize global instance for fast downstream inference
_predictor = SituationPredictor()

def predict(perception_output: dict) -> dict:
    return _predictor.predict(perception_output)
