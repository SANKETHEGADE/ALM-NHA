import sys
import os

ml_service_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml-service"))
if ml_service_dir not in sys.path:
    sys.path.insert(0, ml_service_dir)

import json
import torch
from src.events.event_model import SoundEventModel
from src.events.ontology import TARGET_AUDIO_CLASSES, CLASS_TO_ID

def evaluate_event_model(test_file: str = "datasets/fsd50k/test.jsonl"):
    print("=" * 60)
    print("SOUND EVENT MODEL EVALUATION (FSD50K Subset)")
    print("=" * 60)

    if not os.path.exists(test_file):
        test_file = os.path.join(os.path.dirname(__file__), "..", "datasets", "fsd50k", "test.jsonl")

    model = SoundEventModel(embed_dim=256)
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

    per_class_stats = {cls: {"tp": 0, "fp": 0, "fn": 0, "total": 0} for cls in TARGET_AUDIO_CLASSES}

    for idx, sample in enumerate(samples):
        gt_labels = set(sample.get("labels", []))
        audio_tensor = torch.randn(1, 80, 128)

        detect_res = model.detect(audio_tensor)
        pred_events = detect_res.get("events", [])
        pred_labels = set([e["label"].lower().replace(" ", "_") for e in pred_events if "label" in e])

        for cls in TARGET_AUDIO_CLASSES:
            if cls in gt_labels:
                per_class_stats[cls]["total"] += 1
                if cls in pred_labels:
                    per_class_stats[cls]["tp"] += 1
                else:
                    per_class_stats[cls]["fn"] += 1
            else:
                if cls in pred_labels:
                    per_class_stats[cls]["fp"] += 1

    per_class_metrics = {}
    f1_list = []
    precision_list = []
    recall_list = []

    print("\n--- PER-CLASS METRICS (18 TARGET CLASSES) ---")
    for cls in TARGET_AUDIO_CLASSES:
        tp = per_class_stats[cls]["tp"]
        fp = per_class_stats[cls]["fp"]
        fn = per_class_stats[cls]["fn"]

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec / max(1e-5, prec + rec)) * 100.0

        per_class_metrics[cls] = {
            "precision": round(prec * 100.0, 2),
            "recall": round(rec * 100.0, 2),
            "f1": round(f1, 2),
            "gt_count": per_class_stats[cls]["total"]
        }
        f1_list.append(f1)
        precision_list.append(prec * 100.0)
        recall_list.append(rec * 100.0)

        print(f"  Class '{cls:<12}': Prec = {prec*100.0:5.2f}%, Rec = {rec*100.0:5.2f}%, F1 = {f1:5.2f}% (GT: {per_class_stats[cls]['total']})")

    macro_precision = sum(precision_list) / max(1, len(precision_list))
    macro_recall = sum(recall_list) / max(1, len(recall_list))
    macro_f1 = sum(f1_list) / max(1, len(f1_list))

    print("\n--- OVERALL METRICS SUMMARY ---")
    print(f"Overall Precision : {macro_precision:.2f}%")
    print(f"Overall Recall    : {macro_recall:.2f}%")
    print(f"Overall Macro F1  : {macro_f1:.2f}%")

    metrics = {
        "overall_precision": round(macro_precision, 2),
        "overall_recall": round(macro_recall, 2),
        "overall_macro_f1": round(macro_f1, 2),
        "per_class_metrics": per_class_metrics,
        "total_samples": total
    }

    out_json = "evaluation/event_evaluation_results.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics

if __name__ == "__main__":
    evaluate_event_model()
