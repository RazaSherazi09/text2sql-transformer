import json
import os
import torch
import matplotlib.pyplot as plt
import sentencepiece as spm

from starter.data_prep import load_split, encode_source, encode_target
from starter.embeddings import TokenEmbedding, InputLayer
from starter.tokenizer import PAD_ID, BOS_ID, EOS_ID
from model.transformer import Seq2SeqTransformer
from decode import greedy_decode, beam_search_decode, parse_sql_string, format_readable_sql


def load_trained_model(checkpoint_path="best_model.pt", sp_path="sql_sp.model", device="cpu"):
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, sp


def generate_predictions(model, sp, split="dev", method="greedy", output_path=None, device="cpu"):
    """
    Generates prediction files matching Section 2.4 / 4.3:
    One JSON line per example: {"query": {...}} or {"error": "parse"}
    """
    examples, tables = load_split(split)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    total = len(examples)
    parse_failures = 0
    predictions = []

    print(f"Generating {method} predictions for {split} ({total} examples)...")
    for i, ex in enumerate(examples):
        header = tables[ex["table_id"]]["header"]
        src_text = encode_source(ex["question"], header)
        src_ids = torch.tensor([sp.encode(src_text) + [EOS_ID]], dtype=torch.long)

        if method == "beam":
            token_ids, _ = beam_search_decode(model, src_ids, beam_size=4, max_len=64, device=device)
        else:
            token_ids, _ = greedy_decode(model, src_ids, max_len=64, device=device)

        # Remove BOS and EOS
        cleaned_ids = [t for t in token_ids if t not in (BOS_ID, EOS_ID)]
        pred_text = sp.decode(cleaned_ids)
        parsed = parse_sql_string(pred_text)

        if parsed is not None:
            record = {"query": parsed}
        else:
            record = {"error": "parse"}
            parse_failures += 1

        predictions.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    failure_rate = (parse_failures / total) * 100
    print(f"Saved: {output_path} | Parse failures: {parse_failures}/{total} ({failure_rate:.2f}%)")
    return output_path, failure_rate


def evaluate_component_accuracy(gold_file="WikiSQL/data/dev.jsonl", pred_file="results/dev_greedy.jsonl"):
    """
    Calculates Table 4 Component Accuracy:
    sel column correct, agg correct, WHERE clause correct.
    """
    with open(gold_file, encoding="utf-8") as f:
        gold = [json.loads(line) for line in f]
    with open(pred_file, encoding="utf-8") as f:
        pred = [json.loads(line) for line in f]

    total = len(gold)
    sel_correct = 0
    agg_correct = 0
    where_correct = 0

    for g, p in zip(gold, pred):
        if "error" in p:
            continue
        g_sql = g["sql"]
        p_sql = p["query"]

        if g_sql["sel"] == p_sql["sel"]:
            sel_correct += 1
        if g_sql["agg"] == p_sql["agg"]:
            agg_correct += 1

        # Check conditions (order ignored)
        gold_conds = sorted([[c[0], c[1], str(c[2]).strip().lower()] for c in g_sql["conds"]])
        pred_conds = sorted([[c[0], c[1], str(c[2]).strip().lower()] for c in p_sql["conds"]])
        if gold_conds == pred_conds:
            where_correct += 1

    return {
        "sel_correct": (sel_correct / total) * 100,
        "agg_correct": (agg_correct / total) * 100,
        "where_correct": (where_correct / total) * 100,
    }


def plot_attention_map(model, sp, example_idx=0, split="dev", save_path="results/figures/attention_map.png", device="cpu"):
    """
    Task 5.3: Cross-attention weights of last decoder layer averaged over heads
    Generated tokens (rows) x Source tokens (columns)
    """
    examples, tables = load_split(split)
    ex = examples[example_idx]
    header = tables[ex["table_id"]]["header"]
    src_text = encode_source(ex["question"], header)

    src_tokens = [sp.id_to_piece(idx) for idx in sp.encode(src_text) + [EOS_ID]]
    src_ids = torch.tensor([sp.encode(src_text) + [EOS_ID]], dtype=torch.long)

    token_ids, cross_attn = greedy_decode(model, src_ids, max_len=64, device=device)
    tgt_tokens = [sp.id_to_piece(idx) for idx in token_ids]

    # cross_attn: (1, h, T, S) -> average over 4 heads -> (T, S)
    attn_avg = cross_attn.squeeze(0).mean(dim=0).cpu().numpy()

    # Match length of actual generated tokens
    attn_matrix = attn_avg[:len(tgt_tokens), :len(src_tokens)]

    plt.figure(figsize=(10, 8))
    plt.imshow(attn_matrix, cmap="viridis", aspect="auto")
    plt.xticks(range(len(src_tokens)), src_tokens, rotation=90, fontsize=8)
    plt.yticks(range(len(tgt_tokens)), tgt_tokens, fontsize=8)
    plt.xlabel("Source Tokens")
    plt.ylabel("Generated Tokens")
    plt.title("Decoder Cross-Attention Map (Last Layer, Averaged Over Heads)")
    plt.colorbar()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved attention map to {save_path}")