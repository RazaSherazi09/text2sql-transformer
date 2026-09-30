import torch
from starter.embeddings import TokenEmbedding, InputLayer
from model.transformer import Seq2SeqTransformer, make_causal_mask, make_padding_mask

def run_checks():
    print("=" * 60)
    print("Running Section 3.2 Correctness Checks")
    print("=" * 60)

    vocab_size = 8000
    d_model = 256
    pad_id = 0

    shared = TokenEmbedding(vocab_size, d_model, pad_id=pad_id)
    enc_in = InputLayer(shared, d_model, dropout=0.0)
    dec_in = InputLayer(shared, d_model, dropout=0.0)

    model = Seq2SeqTransformer(
        enc_in, dec_in, shared,
        d_model=256, h=4, d_ff=1024, num_layers=3, dropout=0.0, pad_id=pad_id
    )
    model.eval()

    # --- Check 1: Weight Sharing ---
    assert model.generator.weight is shared.emb.weight, "Weight sharing check failed!"
    print("✓ Check 1 Passed: Weight sharing (model.generator.weight is shared.emb.weight)")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,}")

    # --- Check 2: Causal Mask ---
    src = torch.randint(4, vocab_size, (2, 16))
    tgt_a = torch.randint(4, vocab_size, (2, 8))
    tgt_b = tgt_a.clone()
    tgt_b[:, -1] = torch.randint(4, vocab_size, (2,))  # Modify only the last token

    with torch.no_grad():
        out_a, _ = model(src, tgt_a)
        out_b, _ = model(src, tgt_b)

    causal_diff = (out_a[:, :-1, :] - out_b[:, :-1, :]).abs().max().item()
    assert causal_diff < 1e-5, f"Causal mask failed! Prior token diff: {causal_diff}"
    print(f"✓ Check 2 Passed: Causal mask invariance (max prior diff: {causal_diff:.2e})")

    # --- Check 3: Padding Mask Invariance ---
    # Adding extra pad tokens to source should not change the representation of the original tokens
    src_padded = torch.cat([src, torch.full((2, 6), pad_id, dtype=torch.long)], dim=1)
    with torch.no_grad():
        mask_orig = make_padding_mask(src, pad_id)
        mask_padded = make_padding_mask(src_padded, pad_id)
        enc_orig = model.encode(src, mask_orig)
        enc_padded = model.encode(src_padded, mask_padded)[:, :src.size(1), :]

    pad_diff = (enc_orig - enc_padded).abs().max().item()
    assert pad_diff < 1e-4, f"Padding mask failed! Token representation diff: {pad_diff}"
    print(f"✓ Check 3 Passed: Padding mask invariance (max token diff: {pad_diff:.2e})")

    # --- Check 4: Attention Rows Sum to 1.0 ---
    with torch.no_grad():
        _, attn_weights = model(src, tgt_a)
        # attn_weights shape: (B, h, T, S)
        row_sums = attn_weights.sum(dim=-1)
        row_diff = (row_sums - 1.0).abs().max().item()
    assert row_diff < 1e-4, f"Attention weights do not sum to 1! Max diff: {row_diff}"
    print(f"✓ Check 4 Passed: Attention rows sum to 1.0 (max diff: {row_diff:.2e})")

    print("\n" + "=" * 60)
    print("ALL ARCHITECTURE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_checks()