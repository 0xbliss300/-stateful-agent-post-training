import torch
import torch.nn as nn

class ScaleDotSelfAttention(nn.Module):
    """
    单头缩放点积注意力
    input: [B,T,d_model]
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        # q,k,v:[B,T,d_model]
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        # k^T:[B, d_model, T], q @ k^T:[B, T, T]
        scores = q @ k.transpose(-2, -1) / (self.d_model ** 0.5)
        # 对得分做mask, mask == 0 被屏蔽
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim = -1)
        # out:[B, T, d_model]
        out = attn @ v 

        return self.w_o(out)


class MultiHeadAttention(nn.Module):
    """
    input:[B, T, d_model]
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        # 每个头的维度
        self.d_model = d_model
        self.d_k = d_model // n_heads # 每个头的维度
        self.n_heads = n_heads # 头数
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask = None):
        # 这里直接在前向过程中完成了 多头的注意力计算
        B, T, _ = x.shape
        # 投影 + 切分多头：[B, T, d_model] -> [B, n_heads, T, d_k]
        q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # 每个头并行进行缩放点积注意力
        scores = q @ k.transpose(-2, -1) / (self.d_k ** 0.5) # [B, n_heads, T, T]
        # mask == 0 被屏蔽
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim = -1) # [B, n_heads, T, T]
        head = attn @ v # [B, n_heads, T, d_k]
        head = head.transpose(1, 2).reshape(B, T, -1) # [B, n_heads, T, d_k] -> [B, T, n_heads, d_k] -> [B, T, d_model]

        return self.w_o(head)
