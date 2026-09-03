import argparse
import os
import sys
import time
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.asr.multilingual_asr import ASRModel
from src.asr.tokenizer import MultilingualTokenizer
from src.training.dataset import MultilingualASRDataset, collate_asr_batch
from src.asr.preprocessing import TARGET_SAMPLE_RATE


def train(
    dataset_manifest: str,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str = "auto",
    output_dir: str = "checkpoints/asr_model",
    max_samples: Optional[int] = None,
    grad_accum_steps: int = 1,
    use_fp16: bool = True,
    resume_checkpoint: Optional[str] = None
):
    """
    Main training and fine-tuning loop for Multilingual Non-Whisper ASR.
    """
    # 1. Device Setup
    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)
    
    use_cuda = target_device.type == "cuda"
    print(f"[ASR Training] Using device: {target_device} (CUDA: {use_cuda})")

    # 2. Output directory
    os.makedirs(output_dir, exist_ok=True)

    # 3. Tokenizer and Model
    tokenizer = MultilingualTokenizer()
    model = ASRModel(device=target_device)
    model.to(target_device)

    # Resume checkpoint if provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[ASR Training] Resuming checkpoint from: {resume_checkpoint}")
        model.load_checkpoint(resume_checkpoint)

    # 4. Dataset & DataLoader
    dataset = MultilingualASRDataset(
        manifest_path=dataset_manifest,
        tokenizer=tokenizer,
        max_samples=max_samples,
        target_sr=TARGET_SAMPLE_RATE,
        synthetic_fallback=True
    )
    print(f"[ASR Training] Dataset initialized with {len(dataset)} samples.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_asr_batch,
        drop_last=False
    )

    # 5. Optimizer & Scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-2,
        betas=(0.9, 0.98),
        eps=1e-8
    )

    # Mixed precision scaler (CUDA AMP)
    scaler = torch.amp.GradScaler('cuda') if (use_cuda and use_fp16) else None

    # 6. Training Loop
    print(f"[ASR Training] Starting training for {epochs} epochs (grad_accum={grad_accum_steps})...")
    total_steps = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        epoch_start = time.perf_counter()

        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            audio = batch["audio"].to(target_device)
            targets = batch["targets"].to(target_device)
            target_lengths = batch["target_lengths"].to(target_device)

            # Mixed precision context
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(
                        audio=audio,
                        targets=targets,
                        input_lengths=None,
                        target_lengths=target_lengths
                    )
                    loss = outputs.get("loss")
                    if loss is not None:
                        loss = loss / grad_accum_steps
                
                if loss is not None:
                    scaler.scale(loss).backward()
            else:
                outputs = model(
                    audio=audio,
                    targets=targets,
                    input_lengths=None,
                    target_lengths=target_lengths
                )
                loss = outputs.get("loss")
                if loss is not None:
                    loss = loss / grad_accum_steps
                    loss.backward()

            if loss is not None:
                epoch_loss += loss.item() * grad_accum_steps
                batch_count += 1

            # Gradient accumulation step
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                total_steps += 1

        avg_loss = epoch_loss / max(1, batch_count)
        elapsed = time.perf_counter() - epoch_start
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} - Time: {elapsed:.2f}s - Steps: {total_steps}")

        # Save checkpoint each epoch
        ckpt_path = os.path.join(output_dir, f"asr_model_epoch_{epoch}.pt")
        model.save_checkpoint(ckpt_path)

    # Save final best model
    final_path = os.path.join(output_dir, "asr_model_final.pt")
    model.save_checkpoint(final_path)
    print(f"[ASR Training] Training complete. Final model saved at: {final_path}")
    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Lahari Multilingual Non-Whisper ASR Training CLI (7 Languages)"
    )
    parser.add_argument("--dataset", "-d", type=str, default="datasets/manifests/sample_multilingual_manifest.json", help="Path to dataset JSON manifest")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=2, help="Batch size per step")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Training device")
    parser.add_argument("--output", "-o", type=str, default="checkpoints/asr_model", help="Directory to save model checkpoints")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to load from dataset")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable mixed precision AMP on CUDA")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from existing checkpoint (.pt)")

    args = parser.parse_args()

    train(
        dataset_manifest=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        output_dir=args.output,
        max_samples=args.max_samples,
        grad_accum_steps=args.grad_accum,
        use_fp16=args.fp16,
        resume_checkpoint=args.resume
    )


if __name__ == "__main__":
    main()
