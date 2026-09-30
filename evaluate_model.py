import json
import os
import subprocess
import torch
import matplotlib.pyplot as plt
import sentencepiece as spm
from pathlib import Path

from starter.data_prep import load_split, encode_source, encode_target
from starter.embeddings import TokenEmbedding, InputLayer
from starter.tokenizer import PAD_ID, BOS_ID, EOS_ID
from model.transformer import Seq2SeqTransformer
from decode import greedy_decode, beam_search_decode, parse_sql_string, format_readable_sql


def load_model(checkpoint_path="best_model.pt", sp_path="sql_sp.model", device="cpu"):
    sp = spm.SentencePieceProcessor(model_file=sp_path)
    vocab_size = sp.get_piece_size()
    d_model = 256

    shared_emb = TokenEmbedding(vocab_size, d_model, pad_id=PAD_ID)
    enc_in = InputLayer(shared_emb, d_model=d_model, dropout=0.0)
    dec_in = InputLayer(shared_emb, d_model=d_model, dropout=0.0)

    model = Seq2SeqTransformer(
        enc_input_layer=enc_in,
        dec_input_layer=dec_in,
        shared_embedding=shared_emb,
        d_model=d_model,
        h=4,
        d_ff=1024,
        num_layers=3,
        dropout=0.0,
        pad_id=PAD_ID
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, sp


def gold_roundtrip_check(split="dev"):
    """Correctness Check: Gold roundtrip accuracy must exceed 99%."""
    print("Running Gold Round-Trip Check...")
    examples, _ = load_split(split)
    parsed_gold_file = f"results/gold_{split}_parsed.jsonl"
    os.makedirs("results", exist_ok=True)

    with open(parsed_gold_file, "w", encoding="utf-8") as out_f:
        for ex in examples:
            tgt_text = encode_target(ex["sql"])
            parsed = parse_sql_string(tgt_text)
            if parsed is not None:
                out_f.write(json.dumps({"query": parsed}) + "\n")
            else:
                out_f.write(json.dumps({"error": "parse"}) + "\n")

    cmd = f"python WikiSQL/evaluate.py WikiSQL/data/{split}.jsonl WikiSQL/data/{split}.db {parsed_gold_file}"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(res.stdout)
    return parsed_gold_file


def run_official_evaluator(split, pred_file):
    """Invokes WikiSQL/evaluate.py and parses LF and Ex accuracy."""
    cmd = f"python WikiSQL/evaluate.py WikiSQL/data/{split}.jsonl WikiSQL/data/{split}.db {pred_file}"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    lf_acc, ex_acc = 0.0, 0.0
    for line in res.stdout.splitlines():
        if "ex" in line.lower() or "execution" in line.lower() or "{" in line:
            try:
                data = json.loads(line)
                return data.get("lf", 0.0) * 100, data.get("ex", 0.0) * 100
            except Exception:
                pass
    return lf_acc, ex_acc


def generate_predictions_and_evaluate(model, sp, split="dev", method="greedy", output_path=None, device="cpu"):
    examples, tables = load_split(split)
    total = len(examples)
    parse_failures = 0
    predictions = []

    print(f"Generating predictions ({split}, {method})...")
    for ex in examples:
        header = tables[ex["table_id"]]["header"]
        src_text = encode_source(ex["question"], header)
        src_ids = torch.tensor([sp.encode(src_text) + [EOS_ID]], dtype=torch.long)

        if method == "beam":
            token_ids, _ = beam_search_decode(model, src_ids, beam_size=4, max_len=64, device=device)
        else:
            token_ids, _ = greedy_decode(model, src_ids, max_len=64, device=device)

        cleaned_ids = [t for t in token_ids if t not in (BOS_ID, EOS_ID)]
        pred_text = sp.decode(cleaned_ids)
        parsed = parse_sql_string(pred_text)

        if parsed is not None:
            predictions.append({"query": parsed})
        else:
            predictions.append({"error": "parse"})
            parse_failures += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    fail_rate = (parse_failures / total) * 100
    lf_acc, ex_acc = run_official_evaluator(split, output_path)
    return lf_acc, ex_acc, fail_rate


def generate_samples_md(model, sp, device="cpu"):
    """Generates results/samples.md: 5 correct and 5 wrong predictions."""
    examples, tables = load_split("dev")
    correct_samples = []
    wrong_samples = []

    for ex in examples:
        if len(correct_samples) >= 5 and len(wrong_samples) >= 5:
            break

        header = tables[ex["table_id"]]["header"]
        src_text = encode_source(ex["question"], header)
        src_ids = torch.tensor([sp.encode(src_text) + [EOS_ID]], dtype=torch.long)
        token_ids, _ = greedy_decode(model, src_ids, max_len=64, device=device)

        cleaned_ids = [t for t in token_ids if t not in (BOS_ID, EOS_ID)]
        pred_text = sp.decode(cleaned_ids)
        parsed = parse_sql_string(pred_text)

        gold_sql = format_readable_sql({"query": ex["sql"]}, header)
        pred_sql = format_readable_sql({"query": parsed} if parsed else None, header)

        is_correct = (parsed is not None and parsed == ex["sql"])

        if is_correct and len(correct_samples) < 5:
            correct_samples.append({
                "question": ex["question"],
                "gold": gold_sql,
                "pred": pred_sql
            })
        elif not is_correct and len(wrong_samples) < 5:
            # Diagnose failure mode
            if parsed is None:
                failure_type = "Parse failure"
            elif parsed["sel"] != ex["sql"]["sel"]:
                failure_type = "Wrong column (sel)"
            elif parsed["agg"] != ex["sql"]["agg"]:
                failure_type = "Wrong aggregation"
            elif len(parsed["conds"]) != len(ex["sql"]["conds"]):
                failure_type = "Missing or extra condition"
            else:
                failure_type = "Wrong condition value or operator"

            wrong_samples.append({
                "question": ex["question"],
                "gold": gold_sql,
                "pred": pred_sql,
                "failure": failure_type
            })

    # Write results/samples.md
    with open("results/samples.md", "w", encoding="utf-8") as f:
        f.write("# Qualitative Samples (10 Dev Examples)\n\n## Correct Predictions (5 Examples)\n\n")
        for i, s in enumerate(correct_samples, 1):
            f.write(f"### Example {i}\n- **Question:** {s['question']}\n- **Gold SQL:** `{s['gold']}`\n- **Predicted SQL:** `{s['pred']}`\n\n")

        f.write("## Failure Cases (5 Examples)\n\n")
        for i, s in enumerate(wrong_samples, 1):
            f.write(f"### Failure Example {i}\n- **Question:** {s['question']}\n- **Gold SQL:** `{s['gold']}`\n- **Predicted SQL:** `{s['pred']}`\n- **Failure Mode:** {s['failure']}\n\n")

    print("Saved: results/samples.md")


def plot_attention_map(model, sp, example_idx=0, device="cpu"):
    """Figure 4 / Task 5.3: Decoder cross-attention map."""
    examples, tables = load_split("dev")
    ex = examples[example_idx]
    header = tables[ex["table_id"]]["header"]
    src_text = encode_source(ex["question"], header)

    src_tokens = [sp.id_to_piece(idx) for idx in sp.encode(src_text) + [EOS_ID]]
    src_ids = torch.tensor([sp.encode(src_text) + [EOS_ID]], dtype=torch.long)

    token_ids, cross_attn = greedy_decode(model, src_ids, max_len=64, device=device)
    tgt_tokens = [sp.id_to_piece(idx) for idx in token_ids]

    attn_avg = cross_attn.squeeze(0).mean(dim=0).cpu().numpy()
    attn_matrix = attn_avg[:len(tgt_tokens), :len(src_tokens)]

    plt.figure(figsize=(10, 8))
    plt.imshow(attn_matrix, cmap="viridis", aspect="auto")
    plt.xticks(range(len(src_tokens)), src_tokens, rotation=90, fontsize=8)
    plt.yticks(range(len(tgt_tokens)), tgt_tokens, fontsize=8)
    plt.xlabel("Source Tokens")
    plt.ylabel("Generated Tokens")
    plt.title("Figure 4: Decoder Cross-Attention Map (Last Layer, Averaged Over Heads)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("results/figures/figure4_attention_map.png", dpi=300)
    plt.close()
    print("Saved: results/figures/figure4_attention_map.png")