import torch
import torch.nn as nn
from model.attention import MultiHeadAttention

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network:
    FFN(x) = max(0, xW1 + b1)W2 + b2
    with intermediate dimension d_ff = 1024 and dropout = 0.1.
    """
    def __init__(self, d_model=256, d_ff=1024, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer using Post-Norm:
    LayerNorm(x + Sublayer(x))
    """
    def __init__(self, d_model=256, h=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 1. Self-Attention + Post-Norm
        attn_out, attn_weights = self.self_attn(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # 2. Feed-Forward + Post-Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x, attn_weights


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer using Post-Norm:
    1. Masked Self-Attention -> Add & Norm
    2. Cross-Attention over encoder output -> Add & Norm
    3. FFN -> Add & Norm
    """
    def __init__(self, d_model=256, h=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        # 1. Masked Decoder Self-Attention + Post-Norm
        self_attn_out, self_attn_weights = self.self_attn(x, x, x, mask=self_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        # 2. Cross-Attention (Q from decoder, K, V from encoder memory) + Post-Norm
        cross_attn_out, cross_attn_weights = self.cross_attn(x, memory, memory, mask=cross_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        # 3. Feed-Forward + Post-Norm
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x, self_attn_weights, cross_attn_weights