import math
import os
import time
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import sentencepiece as spm

from starter.dataset import make_loader
from starter.embeddings import TokenEmbedding, InputLayer
from starter.tokenizer import PAD_ID
from model.transformer import Seq2SeqTransformer


def get_secret(key_name):
    """
    Retrieves secret from environment variables, Kaggle Secrets,
    or Google Colab userdata in that order.
    """
    # 1. Standard Environment Variables
    val = os.environ.get(key_name, None)
    if val:
        return val

    # 2. Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        val = UserSecretsClient().get_secret(key_name)
        if val:
            return val
    except Exception:
        pass

    # 3. Google Colab Secrets
    try:
        from google.colab import userdata
        val = userdata.get(key_name)
        if val:
            return val
    except Exception:
        pass

    return None


class NoamLRScheduler:
    """
    Equation 3 from Vaswani et al. (Attention Is All You Need):
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


def plot_lr_schedule(d_model=256, warmup_steps=4000, total_steps=20000, save_path="results/figures/figure3_lr_schedule.png"):
    """
    Section 3.2 Correctness Check:
    Plots the schedule for the first 20,000 steps showing linear rise for 4000 steps
    and subsequent decay as step^(-0.5).
    """
    steps = list(range(1, total_steps + 1))
    lrs = [
        (d_model ** -0.5) * min(s ** -0.5, s * (warmup_steps ** -1.5))
        for s in steps
    ]
    plt.figure(figsize=(8, 4))
    plt.plot(steps, lrs, label="Noam Schedule (d_model=256, warmup=4000)", color="blue")
    plt.axvline(x=4000, color="red", linestyle="--", label="Warmup peak (step 4000)")
    plt.title("Figure 3: Learning-Rate Schedule (First 20,000 Steps)")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def sync_to_hf(repo_id, token, file_path, path_in_repo=None):
    """
    Streams saved model weights and figures directly to Hugging Face Model Hub.
    """
    if not repo_id or not token:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo or os.path.basename(file_path),
            repo_id=repo_id,
            token=token,
            commit_message=f"Sync {os.path.basename(file_path)}"
        )
        print(f"  [HF Hub] Uploaded {file_path} -> {repo_id}")
    except Exception as e:
        print(f"  [HF Hub Warning] Could not push to HF: {e}")


def train_one_epoch(model, dataloader, criterion, optimizer, lr_scheduler, device):
    model.train()
    total_loss, total_tokens = 0.0, 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        # Teacher forcing:
        # Decoder input is tgt[:, :-1], prediction target is tgt[:, 1:]
        dec_input = tgt[:, :-1]
        labels = tgt[:, 1:]

        optimizer.zero_grad()
        logits, _ = model(src, dec_input)

        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        non_pad = (labels != PAD_ID).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

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
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Using device: {device} ({gpu_name})")

    # Output directory tree
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    # Section 3.2 Check: Generate Figure 3 (LR schedule for 20k steps)
    plot_lr_schedule()

    # Load HF credentials safely
    hf_repo = get_secret("HF_REPO_ID")
    hf_token = get_secret("HF_TOKEN")
    if hf_repo and hf_token:
        print(f"Hugging Face sync enabled for target repo: {hf_repo}")
    else:
        print("Hugging Face credentials not found. Local training checkpoints will still be saved to results/.")

    # Load SentencePiece tokenizer
    sp_path = "sql_sp.model"
    if not os.path.exists(sp_path):
        raise FileNotFoundError(f"{sp_path} missing. Run starter/tokenizer.py first.")

    sp = spm.SentencePieceProcessor(model_file=sp_path)
    vocab_size = sp.get_piece_size()

    # Dataloaders: train=True drops over-long pairs (max_src=160, max_tgt=64)
    train_dl = make_loader("train_pairs.jsonl", sp, train=True, batch_size=64)
    dev_dl = make_loader("dev_pairs.jsonl", sp, train=False, batch_size=64)

    # Model configuration strictly as specified
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

    # Section 3.2 Weight sharing validation
    assert model.generator.weight is shared_emb.emb.weight, "Weight sharing check failed!"

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {trainable_params:,}")

    # Section 2.3 Loss, Optimizer, and LR Schedule
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = NoamLRScheduler(optimizer, d_model=d_model, warmup_steps=4000)

    best_dev_loss = float("inf")
    best_epoch = 1
    history = []
    total_training_start = time.time()

    print("\nStarting 20-Epoch Training Loop...")
    for epoch in range(1, 21):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_dl, criterion, optimizer, lr_scheduler, device)
        dv_loss = evaluate_loss(model, dev_dl, criterion, device)
        curr_lr = lr_scheduler.get_lr()
        dur = time.time() - t0

        print(
            f"Epoch {epoch:02d}/20 | "
            f"Train Loss: {tr_loss:.4f} | "
            f"Dev Loss: {dv_loss:.4f} | "
            f"LR: {curr_lr:.6e} | "
            f"{dur:.1f}s"
        )
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "dev_loss": dv_loss,
            "lr": curr_lr
        })

        # Save latest epoch checkpoint locally
        latest_ckpt = f"results/checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dev_loss": dv_loss,
            "train_loss": tr_loss
        }, latest_ckpt)

        # Sync latest checkpoint to Hugging Face
        sync_to_hf(hf_repo, hf_token, latest_ckpt)

        # Save and sync checkpoint with lowest dev loss
        if dv_loss < best_dev_loss:
            best_dev_loss = dv_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_loss": best_dev_loss,
                "vocab_size": vocab_size
            }, "best_model.pt")
            print(f"  --> Saved new best checkpoint to best_model.pt (Dev Loss: {best_dev_loss:.4f})")
            sync_to_hf(hf_repo, hf_token, "best_model.pt")

    total_training_time = time.time() - total_training_start

    # Save training history JSON
    with open("results/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Generate Figure 2: Training and Dev Loss per Epoch
    epochs_range = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    dev_losses = [h["dev_loss"] for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs_range, dev_losses, label="Dev Loss", marker="s")
    plt.title("Figure 2: Training and Dev Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/figure2_loss_curve.png", dpi=300)
    plt.close()

    # Generate Table 2: Model and Training Summary
    t2_content = f"""# Table 2 – Model and training

| Metric | Value |
| :--- | :--- |
| Trainable parameters | {trainable_params:,} |
| Epochs trained / best epoch | 20 / {best_epoch} |
| Best dev loss | {best_dev_loss:.4f} |
| Training time and GPU | {total_training_time / 60:.2f} mins ({gpu_name}) |
"""
    with open("results/tables/table2_model_training.md", "w") as f:
        f.write(t2_content)

    # Sync summary deliverables to Hugging Face
    sync_to_hf(hf_repo, hf_token, "results/figures/figure2_loss_curve.png")
    sync_to_hf(hf_repo, hf_token, "results/figures/figure3_lr_schedule.png")
    sync_to_hf(hf_repo, hf_token, "results/tables/table2_model_training.md")
    sync_to_hf(hf_repo, hf_token, "results/training_history.json")

    print("\nTraining and report generation completed successfully.")


if __name__ == "__main__":
    train()