import torch
import torch.nn as nn
from model.layers import EncoderLayer, DecoderLayer

def make_padding_mask(seq, pad_id=0):
    """
    Creates an additive attention mask for padding positions.
    seq: (B, L)
    Returns: (B, 1, 1, L) float tensor where:
      - 0.0 for real tokens
      - -1e9 for pad tokens
    """
    is_pad = (seq == pad_id).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L) bool
    mask = torch.zeros_like(is_pad, dtype=torch.float32)
    return mask.masked_fill(is_pad, -1e9)


def make_causal_mask(size, device=None):
    """
    Creates a lower-triangular causal mask preventing future token attention.
    Returns: (1, 1, size, size) float tensor where:
      - 0.0 for allowed (past/present) tokens
      - -1e9 for future tokens
    """
    tri = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
    mask = torch.zeros(size, size, dtype=torch.float32, device=device)
    mask = mask.masked_fill(tri, -1e9)
    return mask.unsqueeze(0).unsqueeze(1)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, h=4, d_ff=1024, num_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, h=h, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x, _ = layer(x, mask=mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=256, h=4, d_ff=1024, num_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, h=h, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        last_cross_attn = None
        for layer in self.layers:
            x, _, cross_attn = layer(x, memory, self_mask=self_mask, cross_mask=cross_mask)
            last_cross_attn = cross_attn
        return x, last_cross_attn


class Seq2SeqTransformer(nn.Module):
    """
    Full Encoder-Decoder Transformer with tied embedding/output weights.
    Strictly follows Vaswani et al. and the assignment constraints.
    """
    def __init__(self, enc_input_layer, dec_input_layer, shared_embedding, 
                 d_model=256, h=4, d_ff=1024, num_layers=3, dropout=0.1, pad_id=0):
        super().__init__()
        self.enc_input_layer = enc_input_layer
        self.dec_input_layer = dec_input_layer
        self.shared_embedding = shared_embedding
        self.pad_id = pad_id

        self.encoder = TransformerEncoder(
            d_model=d_model, h=h, d_ff=d_ff, num_layers=num_layers, dropout=dropout
        )
        self.decoder = TransformerDecoder(
            d_model=d_model, h=h, d_ff=d_ff, num_layers=num_layers, dropout=dropout
        )

        # Output projection linear layer
        self.generator = nn.Linear(d_model, shared_embedding.emb.num_embeddings, bias=False)
        # Weight sharing requirement: output projection weight IS the embedding weight
        self.generator.weight = self.shared_embedding.emb.weight

    def encode(self, src, src_mask=None):
        x = self.enc_input_layer(src)
        return self.encoder(x, mask=src_mask)

    def decode(self, tgt, memory, tgt_mask=None, cross_mask=None):
        y = self.dec_input_layer(tgt)
        return self.decoder(y, memory, self_mask=tgt_mask, cross_mask=cross_mask)

    def forward(self, src, tgt):
        # src: (B, S), tgt: (B, T)
        src_mask = make_padding_mask(src, self.pad_id)
        
        # Decoder self-attention: causal mask + target padding mask
        tgt_pad_mask = make_padding_mask(tgt, self.pad_id)
        causal_mask = make_causal_mask(tgt.size(1), device=tgt.device)
        decoder_self_mask = tgt_pad_mask + causal_mask

        # Cross-attention mask hides source padding positions from decoder queries
        cross_mask = make_padding_mask(src, self.pad_id)

        memory = self.encode(src, src_mask=src_mask)
        dec_out, last_cross_attn = self.decode(
            tgt, memory, tgt_mask=decoder_self_mask, cross_mask=cross_mask
        )
        logits = self.generator(dec_out)
        return logits, last_cross_attn