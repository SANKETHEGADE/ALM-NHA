import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
import os
import sys

# Add parent directory to path to import features and model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import PerceptionPreprocessor, SITUATION_CLASSES
from train import SituationClassifierMLP, synthesize_dataset

def main():
    artifact_path = r"u:\Hackathons\ALM-NHCE\ml-service\app\models\artifact\model.pt"
    if not os.path.exists(artifact_path):
        print("Model artifact not found. Please run train.py first.")
        return
        
    print(f"Loading model from {artifact_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(artifact_path, map_location=device, weights_only=False)
    preprocessor = checkpoint['preprocessor']
    input_dim = checkpoint['input_dim']
    num_classes = checkpoint['num_classes']
    
    model = SituationClassifierMLP(input_dim, num_classes)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    # Generate evaluation set using the same deterministic mapping
    print("Generating evaluation dataset...")
    transcripts, sound_events_list, emotions, speakers_list, y_true = synthesize_dataset()
    
    print("Preprocessing evaluation data...")
    X = preprocessor.transform(transcripts, sound_events_list, emotions, speakers_list)
    y = torch.tensor(y_true, dtype=torch.long)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    all_preds = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            
    # Calculate Metrics
    f1_per_class = f1_score(y_true, all_preds, average=None, zero_division=0)
    f1_macro = f1_score(y_true, all_preds, average='macro', zero_division=0)
    
    print(f"\n--- Evaluation Results ---")
    print(f"Macro F1 Score: {f1_macro:.4f}")
    print("\nPer-class F1 Scores:")
    for i, score in enumerate(f1_per_class):
        if score > 0:
            print(f" - {SITUATION_CLASSES[i]}: {score:.4f}")
            
    # Confusion Matrix
    cm = confusion_matrix(y_true, all_preds)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=SITUATION_CLASSES, yticklabels=SITUATION_CLASSES)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    eval_dir = r"u:\Hackathons\ALM-NHCE\ml-service\app\models\eval"
    os.makedirs(eval_dir, exist_ok=True)
    cm_path = os.path.join(eval_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
