import math
import os
import time
import json
import torch
import torch.nn as nn
from pathlib import Path
import sentencepiece as spm

from starter.dataset import make_loader
from starter.embeddings import TokenEmbedding, InputLayer
from starter.tokenizer import PAD_ID
from model.transformer import Seq2SeqTransformer


class NoamLRScheduler:
    """
    Learning rate schedule from Vaswani et al. (Equation 3):
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    def __init__(self, optimizer, d_model=256, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self):
        if self.step_num == 0:
            return 0.0
        return (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )


def train_one_epoch(model, dataloader, criterion, optimizer, lr_scheduler, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing:
        # Decoder input: tgt[:, :-1]
        # Target labels: tgt[:, 1:]
        dec_input = tgt[:, :-1]
        labels = tgt[:, 1:]

        optimizer.zero_grad()
        logits, _ = model(src, dec_input)

        # Flatten for CrossEntropyLoss
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Count non-pad tokens for normalized loss reporting
        non_pad = (labels != PAD_ID).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        dec_input = tgt[:, :-1]
        labels = tgt[:, 1:]

        logits, _ = model(src, dec_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        non_pad = (labels != PAD_ID).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / max(total_tokens, 1)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check for sentencepiece model
    sp_path = "sql_sp.model"
    if not os.path.exists(sp_path):
        raise FileNotFoundError(f"{sp_path} not found. Run starter/tokenizer.py first.")

    sp = spm.SentencePieceProcessor(model_file=sp_path)
    vocab_size = sp.get_piece_size()

    # Loaders: train=True drops over-long pairs (max_src=160, max_tgt=64)
    train_dl = make_loader("train_pairs.jsonl", sp, train=True, batch_size=64)
    dev_dl = make_loader("dev_pairs.jsonl", sp, train=False, batch_size=64)

    # Model configuration
    d_model = 256
    shared_emb = TokenEmbedding(vocab_size, d_model, pad_id=PAD_ID)
    enc_in = InputLayer(shared_emb, d_model=d_model, dropout=0.1)
    dec_in = InputLayer(shared_emb, d_model=d_model, dropout=0.1)

    model = Seq2SeqTransformer(
        enc_input_layer=enc_in,
        dec_input_layer=dec_in,
        shared_embedding=shared_emb,
        d_model=d_model,
        h=4,
        d_ff=1024,
        num_layers=3,
        dropout=0.1,
        pad_id=PAD_ID
    ).to(device)

    # Loss: Cross-entropy with label smoothing 0.1, ignoring PAD_ID
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=PAD_ID)

    # Optimizer: Adam with beta1=0.9, beta2=0.98, eps=1e-9
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9
    )

    # Learning rate schedule with 4000 warmup steps
    lr_scheduler = NoamLRScheduler(optimizer, d_model=d_model, warmup_steps=4000)

    best_dev_loss = float("inf")
    os.makedirs("results", exist_ok=True)
    history = []

    print("\nStarting Training (20 Epochs)...")
    for epoch in range(1, 21):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, lr_scheduler, device)
        dev_loss = evaluate_loss(model, dev_dl, criterion, device)
        curr_lr = lr_scheduler.get_lr()
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/20 | "
            f"Train Loss: {train_loss:.4f} | "
            f"Dev Loss: {dev_loss:.4f} | "
            f"LR: {curr_lr:.6e} | "
            f"Time: {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "lr": curr_lr
        })

        # Save checkpoint with lowest dev loss
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_loss": dev_loss,
                "vocab_size": vocab_size,
            }, "best_model.pt")
            print(f"  --> Saved new best checkpoint (Dev Loss: {best_dev_loss:.4f})")

    # Save training loss history for Figure 2
    with open("results/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete! Best checkpoint saved to best_model.pt")


if __name__ == "__main__":
    train()