import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.alm.alm_model import CoreALM
from src.alm.losses import ALMTaskLoss

class ALMJointDataset(Dataset):
    """
    PyTorch Dataset wrapper for ALM Joint JSONL files.
    """
    def __init__(self, jsonl_path: str):
        self.samples = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line.strip()))
        else:
            print(f"[ALMJointDataset] Warning: File {jsonl_path} not found. Creating dummy dataset.")
            self.samples = [{
                "question": "Where is the speaker likely to be?",
                "answer": "Airport terminal",
                "scene": "airport"
            }]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Generate audio representation tensor (1, 80, 128)
        audio_tensor = torch.randn(1, 80, 128)
        question = sample.get("question", "Where is the speaker likely to be?")
        
        # Build synthetic target answer token IDs
        answer_ids = torch.randint(1, 1000, (64,))
        evidence_targets = torch.randint(0, 2, (64,)).float()
        target_confidence = torch.tensor([0.90])

        return {
            "audio": audio_tensor,
            "question": question,
            "answer_ids": answer_ids,
            "evidence_targets": evidence_targets,
            "target_confidence": target_confidence
        }

def train_alm(args):
    print("=" * 60)
    print("CORE ALM TRAINING PIPELINE - SMART HORIZON 2026")
    print("=" * 60)
    print(f"Dataset path   : {args.dataset}")
    print(f"Epochs         : {args.epochs}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Learning rate  : {args.learning_rate}")
    print(f"Device         : {args.device}")
    print(f"Output dir     : {args.output}")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Load dataset
    dataset = ALMJointDataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize Core ALM Model
    model = CoreALM(embed_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = ALMTaskLoss()

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"[Train] Resumed from checkpoint {args.resume} at epoch {start_epoch}")

    model.train()
    grad_accum_steps = 2

    for epoch in range(start_epoch, start_epoch + args.epochs):
        total_epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            audio = batch["audio"].to(device)
            question = batch["question"][0] # batch size 1 text string or list
            
            targets = {
                "answer_ids": batch["answer_ids"].to(device),
                "evidence_targets": batch["evidence_targets"].to(device),
                "target_confidence": batch["target_confidence"].to(device)
            }

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(audio, question_text=question)
                loss_dict = loss_fn(outputs, targets)
                loss = loss_dict["loss"] / grad_accum_steps

            if device.type == 'cuda':
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                if device.type == 'cuda':
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_epoch_loss += loss.item() * grad_accum_steps

        avg_loss = total_epoch_loss / max(1, len(dataloader))
        print(f"Epoch [{epoch}/{start_epoch + args.epochs - 1}] - Avg Loss: {avg_loss:.4f}")

        # Save Checkpoint
        ckpt_path = os.path.join(args.output, f"alm_checkpoint_epoch_{epoch}.pt")
        latest_path = os.path.join(args.output, "best_checkpoint.pt")
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }
        torch.save(ckpt_data, ckpt_path)
        torch.save(ckpt_data, latest_path)

    print(f"[Train] Training complete. Checkpoints saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Core ALM Model")
    parser.add_argument("--dataset", type=str, default="datasets/alm_nhce/train.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--output", type=str, default="ml-service/checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint file path to resume from")

    args = parser.parse_args()
    train_alm(args)
