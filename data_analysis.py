import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sentencepiece as spm
from starter.embeddings import PositionalEncoding
from starter.tokenizer import PAD_ID, BOS_ID, EOS_ID

def compute_data_statistics():
    print("Generating Table 1: Dataset Statistics...")
    sp = spm.SentencePieceProcessor(model_file="sql_sp.model")
    splits = ["train", "dev", "test"]
    stats = {}

    for split in splits:
        with open(f"{split}_pairs.jsonl", "r", encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]

        src_lens = [len(sp.encode(p["src"])) + 1 for p in pairs]  # +1 for EOS
        tgt_lens = [len(sp.encode(p["tgt"])) + 2 for p in pairs]  # +2 for BOS, EOS

        dropped = 0
        if split == "train":
            for sl, tl in zip(src_lens, tgt_lens):
                if sl > 160 or tl > 64:
                    dropped += 1

        stats[split] = {
            "pairs": len(pairs),
            "mean_src": np.mean(src_lens),
            "max_src": np.max(src_lens),
            "mean_tgt": np.mean(tgt_lens),
            "max_tgt": np.max(tgt_lens),
            "dropped": dropped if split == "train" else "-"
        }

    # Format Table 1 Markdown
    table1 = f"""# Table 1 – Data

| Metric | Train | Dev | Test |
| :--- | :--- | :--- | :--- |
| Pairs | {stats['train']['pairs']:,} | {stats['dev']['pairs']:,} | {stats['test']['pairs']:,} |
| Mean / max source length (tokens) | {stats['train']['mean_src']:.1f} / {stats['train']['max_src']} | {stats['dev']['mean_src']:.1f} / {stats['dev']['max_src']} | {stats['test']['mean_src']:.1f} / {stats['test']['max_src']} |
| Mean / max target length (tokens) | {stats['train']['mean_tgt']:.1f} / {stats['train']['max_tgt']} | {stats['dev']['mean_tgt']:.1f} / {stats['dev']['max_tgt']} | {stats['test']['mean_tgt']:.1f} / {stats['test']['max_tgt']} |
| Pairs dropped as too long | {stats['train']['dropped']} | - | - |
"""
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    with open("results/tables/table1_data.md", "w") as f:
        f.write(table1)
    print("Saved: results/tables/table1_data.md")


def plot_positional_encoding():
    print("Generating Figure 1: Positional Encoding Heatmap...")
    pe_layer = PositionalEncoding(d_model=256, max_len=512, dropout=0.0)
    pe_matrix = pe_layer.pe.squeeze(0)[:100, :].numpy()  # (100, 256)

    plt.figure(figsize=(10, 6))
    plt.imshow(pe_matrix, cmap="RdBu", aspect="auto")
    plt.title("Figure 1: Sinusoidal Positional Encoding (First 100 Positions × 256 Dimensions)")
    plt.xlabel("Embedding Dimension (0 to 255)")
    plt.ylabel("Sequence Position (0 to 99)")
    plt.colorbar(label="Value")
    plt.tight_layout()

    Path("results/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/figures/figure1_positional_encoding.png", dpi=300)
    plt.close()
    print("Saved: results/figures/figure1_positional_encoding.png")


if __name__ == "__main__":
    compute_data_statistics()
    plot_positional_encoding()