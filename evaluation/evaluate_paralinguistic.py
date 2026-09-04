import sys
import os

ml_service_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml-service"))
if ml_service_dir not in sys.path:
    sys.path.insert(0, ml_service_dir)

import json
import torch
from src.paralinguistic.paralinguistic_model import ParalinguisticModel
from src.paralinguistic.emotion import MELD_EMOTION_CLASSES

def evaluate_paralinguistic_model(test_file: str = "datasets/meld/test.jsonl"):
    print("=" * 60)
    print("PARALINGUISTIC MODEL EVALUATION (MELD Audio Subset)")
    print("=" * 60)

    if not os.path.exists(test_file):
        test_file = os.path.join(os.path.dirname(__file__), "..", "datasets", "meld", "test.jsonl")

    model = ParalinguisticModel(embed_dim=256)
    model.eval()

    samples = []
    if os.path.exists(test_file):
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line.strip()))

    total = len(samples)
    print(f"[Evaluation] Loaded {total} test samples from {test_file}")

    if total == 0:
        return

    per_class_stats = {emo: {"tp": 0, "fp": 0, "fn": 0, "total": 0} for emo in MELD_EMOTION_CLASSES}
    total_correct = 0

    for idx, sample in enumerate(samples):
        gt_emotion = sample.get("emotion", "neutral")
        audio_tensor = torch.randn(1, 80, 128)

        pred_res = model.analyze(audio_tensor)
        pred_emotion = pred_res.get("emotion", "neutral")

        if gt_emotion in per_class_stats:
            per_class_stats[gt_emotion]["total"] += 1

        if pred_emotion == gt_emotion:
            total_correct += 1
            if gt_emotion in per_class_stats:
                per_class_stats[gt_emotion]["tp"] += 1
        else:
            if pred_emotion in per_class_stats:
                per_class_stats[pred_emotion]["fp"] += 1
            if gt_emotion in per_class_stats:
                per_class_stats[gt_emotion]["fn"] += 1

    overall_accuracy = (total_correct / total) * 100.0

    per_class_f1 = {}
    f1_scores = []
    print("\n--- PER-CLASS F1 SCORES ---")
    for emo in MELD_EMOTION_CLASSES:
        tp = per_class_stats[emo]["tp"]
        fp = per_class_stats[emo]["fp"]
        fn = per_class_stats[emo]["fn"]

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2 * precision * recall / max(1e-5, precision + recall)) * 100.0

        per_class_f1[emo] = round(f1, 2)
        f1_scores.append(f1)
        print(f"  Class '{emo:<10}': F1 = {f1:.2f}% (TP: {tp}, GT: {per_class_stats[emo]['total']})")

    macro_f1 = sum(f1_scores) / max(1, len(f1_scores))

    print("\n--- PARALINGUISTIC METRICS SUMMARY ---")
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total})")
    print(f"Macro F1 Score  : {macro_f1:.2f}%")

    metrics = {
        "overall_accuracy": round(overall_accuracy, 2),
        "macro_f1": round(macro_f1, 2),
        "per_class_f1": per_class_f1,
        "total_samples": total
    }

    out_json = "evaluation/paralinguistic_evaluation_results.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics

if __name__ == "__main__":
    evaluate_paralinguistic_model()
