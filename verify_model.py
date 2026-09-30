import torch
from starter.embeddings import TokenEmbedding, InputLayer
from model.transformer import Seq2SeqTransformer, make_causal_mask, make_padding_mask

def run_checks():
    print("Running Section 3.2 Correctness Checks...")
    vocab_size = 8000
    d_model = 256
    pad_id = 0

    shared = TokenEmbedding(vocab_size, d_model, pad_id=pad_id)
    enc_in = InputLayer(shared, d_model)
    dec_in = InputLayer(shared, d_model)

    model = Seq2SeqTransformer(
        enc_in, dec_in, shared,
        d_model=256, h=4, d_ff=1024, num_layers=3, dropout=0.0, pad_id=pad_id
    )
    model.eval()

    # 1. Weight sharing check
    assert model.generator.weight is shared.emb.weight, "Weight sharing check failed!"
    print("✓ Weight sharing check passed (is check is True).")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # 2. Causal mask test
    # Altering the last token of decoder input must not change outputs at prior positions
    src = torch.randint(4, vocab_size, (2, 20))
    tgt_a = torch.randint(4, vocab_size, (2, 10))
    tgt_b = tgt_a.clone()
    tgt_b[:, -1] = torch.randint(4, vocab_size, (2,))  # Modify only the last token

    with torch.no_grad():
        out_a, _ = model(src, tgt_a)
        out_b, _ = model(src, tgt_b)

    diff = (out_a[:, :-1, :] - out_b[:, :-1, :]).abs().max().item()
    assert diff < 1e-5, f"Causal mask failed: earlier tokens changed by {diff}!"
    print(f"✓ Causal mask check passed (max prior variation: {diff:.2e}).")

    # 3. Padding mask test
    # Appending padding tokens to source must not change encoder representation of original tokens
    src_padded = torch.cat([src, torch.full((2, 5), pad_id, dtype=torch.long)], dim=1)
    with torch.no_grad():
        mask_orig = make_padding_mask(src, pad_id)
        mask_padded = make_padding_mask(src_padded, pad_id)
        enc_orig = model.encode(src, mask_orig)
        enc_padded = model.encode(src_padded, mask_padded)[:, :20, :]

    pad_diff = (enc_orig - enc_padded).abs().max().item()
    assert pad_diff < 1e-4, f"Padding mask failed: representation changed by {pad_diff}!"
    print(f"✓ Padding mask check passed (max token variation: {pad_diff:.2e}).")

    print("\nAll architecture correctness checks passed!")

if __name__ == "__main__":
    run_checks()