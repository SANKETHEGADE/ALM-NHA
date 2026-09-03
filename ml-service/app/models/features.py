import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import numpy as np
import pickle

SITUATION_CLASSES = [
    "normal_conversation", "basic_fight", "medical_distress", "assault",
    "fire_emergency", "robbery", "car_accident", "domestic_violence",
    "kidnapping", "vandalism", "public_disturbance", "drowning",
    "active_shooter", "bomb_threat", "earthquake", "flood",
    "tornado", "gas_leak", "power_outage", "animal_attack",
    "riot", "protest", "drug_overdose", "suicide_attempt",
    "trespassing", "burglary", "explosion", "stalking",
    "harassment", "structure_collapse"
]

class PerceptionPreprocessor:
    """
    Transforms raw perception metadata into a unified feature vector suitable for neural network consumption.
    It combines TF-IDF for text transcripts, MultiLabelBinarizer for sound events, and One-Hot Encoding for emotions.
    """
    def __init__(self, max_features=1000):
        self.text_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.sound_binarizer = MultiLabelBinarizer()
        self.emotion_encoder = LabelEncoder()
        
        self.is_fitted = False
        
        self.known_emotions = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
        self.known_sounds = ["siren", "glass_shatter", "gunshot", "scream", "explosion", "crying", "dog_bark", "engine", "alarm", "footsteps", "silence", "traffic", "music"]

    def fit(self, transcripts, sound_events_list, emotions):
        self.text_vectorizer.fit(transcripts)
        # Fit on known classes so it handles unseen data gracefully
        self.sound_binarizer.fit([self.known_sounds])
        self.emotion_encoder.fit(self.known_emotions)
        self.is_fitted = True

    def transform(self, transcripts, sound_events_list, emotions, speakers_list):
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted.")
            
        text_features = self.text_vectorizer.transform(transcripts).toarray()
        
        # Filter unknown sounds to prevent errors
        filtered_sounds = [[s for s in sounds if s in self.known_sounds] for sounds in sound_events_list]
        sound_features = self.sound_binarizer.transform(filtered_sounds)
        
        # Handle unknown emotions by mapping to 'neutral'
        safe_emotions = [e if e in self.known_emotions else "neutral" for e in emotions]
        emotion_idx = self.emotion_encoder.transform(safe_emotions)
        emotion_features = np.zeros((len(safe_emotions), len(self.known_emotions)))
        emotion_features[np.arange(len(safe_emotions)), emotion_idx] = 1.0
        
        # Speakers as a 2D array
        speaker_features = np.array(speakers_list).reshape(-1, 1)
        
        combined = np.hstack([text_features, sound_features, emotion_features, speaker_features])
        return torch.tensor(combined, dtype=torch.float32)

    def get_feature_dim(self):
        return len(self.text_vectorizer.get_feature_names_out()) + len(self.known_sounds) + len(self.known_emotions) + 1

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
