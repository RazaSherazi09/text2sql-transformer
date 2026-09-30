import math
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention as described in Vaswani et al. (Section 3.2.1).
    Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k) + mask) V
    Returns both attended representation and attention weight distribution.
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q: (B, h, L_q, d_k)
        # k: (B, h, L_k, d_k)
        # v: (B, h, L_k, d_v)
        # mask: broadcastable to (B, h, L_q, L_k), containing large negative values for masked positions
        d_k = q.size(-1)
        
        # Calculate raw dot-product scores: (B, h, L_q, L_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights_dropped, v)  # (B, h, L_q, d_v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    Projects queries, keys, and values h times, applies ScaledDotProductAttention,
    concatenates results, and applies final linear projection.
    """
    def __init__(self, d_model=256, h=4, dropout=0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h  # 64 when d_model=256, h=4

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 1. Linear projections: (B, L, d_model) -> (B, L, h, d_k) -> (B, h, L, d_k)
        q_proj = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v_proj = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        attn_out, attn_weights = self.attention(q_proj, k_proj, v_proj, mask=mask)

        # 3. Concatenate heads: (B, h, L, d_k) -> (B, L, h, d_k) -> (B, L, d_model)
        concat = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4. Final output projection
        output = self.w_o(concat)
        return output, attn_weights