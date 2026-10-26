import math
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == False, -1e9)

        weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(weights, v)

        return output, weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads    # 256 / 4 = 64

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.last_weights = None           

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        out, weights = self.attention(q, k, v, mask)

        self.last_weights = weights

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.w_o(out)
