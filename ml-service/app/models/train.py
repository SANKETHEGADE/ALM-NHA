import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import os
import random
from features import PerceptionPreprocessor, SITUATION_CLASSES

class SituationClassifierMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) architecture used to classify emergency situations.
    It takes a unified feature vector from the PerceptionPreprocessor and outputs logits for each of the 30 classes.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def synthesize_dataset(num_samples=5000):
    print("Generating top-down deterministic synthetic dataset...")
    transcripts = []
    sound_events_list = []
    emotions = []
    speakers_list = []
    y_labels = []

    # Deterministic mapping to simulate correlated dataset distributions
    SITUATION_MAPPING = {
        "normal_conversation": (["how are you", "what's for dinner", "I'll be there soon", "just chatting"], ["footsteps", "none", "music"], ["neutral", "joy"]),
        "basic_fight": (["stop it", "get away from me", "I'll hit you", "calm down"], ["shouting", "thud"], ["anger", "annoyance"]),
        "medical_distress": (["my chest hurts", "call an ambulance", "I can't breathe", "she passed out"], ["siren", "crying", "none"], ["fear", "sadness"]),
        "assault": (["help me", "stop hitting me", "he has a knife", "get off me"], ["scream", "thud", "shouting"], ["fear", "panic"]),
        "fire_emergency": (["fire", "it's burning", "smoke everywhere", "call the fire department"], ["alarm", "siren", "crackling"], ["fear", "panic"]),
        "car_accident": (["we crashed", "call a tow truck", "is everyone okay", "rear ended"], ["glass_shatter", "siren", "traffic"], ["surprise", "nervousness"]),
        "domestic_violence": (["don't hit me", "put that down", "please stop", "get out of the house"], ["scream", "crying", "thud"], ["fear", "sadness"]),
        "robbery": (["give me the money", "put your hands up", "empty the register", "he has a gun"], ["scream", "footsteps"], ["fear", "panic"]),
        "kidnapping": (["get in the car", "let me go", "where are you taking me", "help"], ["scream", "engine", "crying"], ["fear", "panic"]),
        "vandalism": (["they're breaking windows", "graffiti", "smashing the car", "stop ruining it"], ["glass_shatter", "shouting"], ["anger", "annoyance"]),
        "public_disturbance": (["keep it down", "you're being too loud", "stop fighting in the street"], ["shouting", "music", "traffic"], ["annoyance", "anger"]),
        "drowning": (["help I can't swim", "he's going under", "throw a rope"], ["splashing", "scream", "crying"], ["panic", "fear"]),
        "active_shooter": (["he has a gun", "shots fired", "get down", "run away"], ["gunshot", "scream", "siren"], ["fear", "panic"]),
        "bomb_threat": (["there is a bomb", "evacuate the building", "it's going to explode", "suspicious package"], ["siren", "alarm"], ["fear", "nervousness"]),
        "explosion": (["it blew up", "huge explosion", "duck and cover"], ["explosion", "siren", "glass_shatter"], ["panic", "surprise"]),
        "earthquake": (["the ground is shaking", "get under the table", "it's an earthquake"], ["rumbling", "glass_shatter", "scream"], ["fear", "surprise"]),
        "flood": (["the water is rising", "get to higher ground", "it's flooded"], ["splashing", "siren", "rain"], ["fear", "sadness"]),
        "tornado": (["tornado warning", "head to the basement", "funnel cloud"], ["siren", "wind", "rumbling"], ["fear", "panic"]),
        "gas_leak": (["smells like gas", "don't light a match", "gas line broke"], ["hissing", "siren", "alarm"], ["nervousness", "fear"]),
        "power_outage": (["the lights went out", "we lost power", "where is the flashlight"], ["silence", "footsteps"], ["confusion", "annoyance"]),
        "animal_attack": (["it's biting me", "get the dog off", "bear attack"], ["dog_bark", "scream", "growling"], ["fear", "panic"]),
        "structure_collapse": (["the roof caved in", "building is falling", "trapped under rubble"], ["rumbling", "thud", "siren"], ["panic", "fear"]),
        "riot": (["they're rioting", "throw tear gas", "form a line", "push back"], ["shouting", "glass_shatter", "siren"], ["anger", "panic"]),
        "protest": (["peaceful protest", "march for our rights", "no justice no peace"], ["shouting", "chanting", "traffic"], ["anger", "neutral"]),
        "drug_overdose": (["he took too much", "he's not waking up", "where is the narcan"], ["siren", "crying"], ["fear", "sadness"]),
        "suicide_attempt": (["don't jump", "put the gun down", "it's not worth it"], ["crying", "siren"], ["sadness", "panic"]),
        "trespassing": (["you're on private property", "get out of my yard", "calling the cops"], ["footsteps", "dog_bark"], ["anger", "annoyance"]),
        "burglary": (["someone broke in", "the window is broken", "my tv is missing"], ["glass_shatter", "footsteps"], ["surprise", "sadness"]),
        "stalking": (["he's following me", "that car is still behind me", "I think I'm being followed"], ["footsteps", "engine"], ["fear", "nervousness"]),
        "harassment": (["leave me alone", "stop following me", "don't talk to me like that"], ["shouting", "footsteps"], ["anger", "annoyance"])
    }

    # Ensure all situations are covered
    for situation in SITUATION_CLASSES:
        if situation not in SITUATION_MAPPING:
            SITUATION_MAPPING[situation] = (["emergency"], ["siren"], ["panic"])

    for _ in range(num_samples):
        situation = random.choice(SITUATION_CLASSES)
        phrases, possible_sounds, possible_emotions = SITUATION_MAPPING[situation]
        
        transcript = random.choice(phrases)
        emotion = random.choice(possible_emotions)
        sounds = random.sample(possible_sounds, k=min(len(possible_sounds), random.randint(1, 2)))
        
        transcripts.append(transcript)
        emotions.append(emotion)
        sound_events_list.append(sounds)
        speakers_list.append(random.randint(1, 2))
        y_labels.append(SITUATION_CLASSES.index(situation))

    return transcripts, sound_events_list, emotions, speakers_list, y_labels

def main():
    os.makedirs(r"u:\Hackathons\ALM-NHCE\ml-service\app\models\artifact", exist_ok=True)
    
    transcripts, sound_events_list, emotions, speakers_list, y_labels = synthesize_dataset()
    
    print("Fitting preprocessor...")
    preprocessor = PerceptionPreprocessor(max_features=1000)
    preprocessor.fit(transcripts, sound_events_list, emotions)
    
    print("Transforming data to tensors...")
    X = preprocessor.transform(transcripts, sound_events_list, emotions, speakers_list)
    y = torch.tensor(y_labels, dtype=torch.long)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create DataLoader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize Model
    input_dim = preprocessor.get_feature_dim()
    num_classes = len(SITUATION_CLASSES)
    model = SituationClassifierMLP(input_dim, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")
        
    # Save Artifacts
    print(r"Saving model and preprocessor to artifact\model.pt...")
    torch.save({
        'model_state': model.state_dict(),
        'input_dim': input_dim,
        'num_classes': num_classes,
        'preprocessor': preprocessor
    }, r"u:\Hackathons\ALM-NHCE\ml-service\app\models\artifact\model.pt")
    print("Training complete.")

if __name__ == "__main__":
    main()
