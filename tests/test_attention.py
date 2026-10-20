import torch
from src.foundations.attention import ScaleDotSelfAttention, MultiHeadAttention


# ==================================================================
# 测试 1：单头 —— 形状、行和为 1、causal 上三角为 0、与手写参考一致
# ==================================================================
def test_single_head():
    print("=" * 60)
    print("测试 1：单头 ScaleDotSelfAttention")
    B, T, D = 2, 8, 32
    x = torch.randn(B, T, D)
    causal = torch.tril(torch.ones(T, T)).unsqueeze(0)  # [1,T,T] 可广播到 [B,T,T]

    m = ScaleDotSelfAttention(D)
    out = m(x, mask=causal)
    print("  out.shape:", tuple(out.shape))              # (2,8,32)

    # 手动复现，顺便取出 attn 做检查
    q, k, v = m.w_q(x), m.w_k(x), m.w_v(x)
    scores = q @ k.transpose(-2, -1) / (D ** 0.5)
    scores = scores.masked_fill(causal == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)

    print("  行和≈1:", torch.allclose(attn.sum(-1), torch.ones(B, T), atol=1e-6))
    upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
    print("  上三角为0:", bool((attn[:, upper] == 0).all()))

    out_ref = m.w_o(attn @ v)
    print("  与参考一致:", torch.allclose(out, out_ref, atol=1e-6))

    # 不带 mask 时行和也应为 1
    attn_nomask = torch.softmax(q @ k.transpose(-2, -1) / (D ** 0.5), dim=-1)
    print("  无mask行和≈1:", torch.allclose(attn_nomask.sum(-1), torch.ones(B, T), atol=1e-6))


# ==================================================================
# 测试 2：多头 —— 形状、d_k、mask 广播、与“逐头手写”参考一致
# ==================================================================
def test_multi_head():
    print("=" * 60)
    print("测试 2：多头 MultiHeadAttention")
    B, T, D, H = 2, 8, 32, 4
    x = torch.randn(B, T, D)
    causal = torch.tril(torch.ones(T, T)).unsqueeze(0)  # [1,T,T] -> 广播 [B,H,T,T]

    m = MultiHeadAttention(D, H)
    out = m(x, mask=causal)
    print("  d_k:", m.d_k, "| out.shape:", tuple(out.shape))   # 8 | (2,8,32)

    # 取出 attn 检查
    q = m.w_q(x).view(B, T, H, m.d_k).transpose(1, 2)
    k = m.w_k(x).view(B, T, H, m.d_k).transpose(1, 2)
    v = m.w_v(x).view(B, T, H, m.d_k).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / (m.d_k ** 0.5)
    scores = scores.masked_fill(causal == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)

    print("  attn.shape:", tuple(attn.shape))                  # (2,4,8,8)
    print("  行和≈1:", torch.allclose(attn.sum(-1), torch.ones(B, H, T), atol=1e-6))
    upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
    print("  上三角为0:", bool((attn[:, :, upper] == 0).all()))

    head_ref = (attn @ v).transpose(1, 2).reshape(B, T, -1)
    out_ref = m.w_o(head_ref)
    print("  与逐头参考一致:", torch.allclose(out, out_ref, atol=1e-6))


# ==================================================================
# 测试 3：拆分/合并无损 + 多头在 H=1 时应退化为单头（缩放因子除外）
# ==================================================================
def test_split_merge_and_h1():
    print("=" * 60)
    print("测试 3：拆分/合并无损 & H=1 退化")
    B, T, D, H = 2, 8, 32, 4
    x = torch.randn(B, T, D)
    m = MultiHeadAttention(D, H)

    # view -> transpose -> transpose -> reshape 应还原
    y = x.view(B, T, H, m.d_k).transpose(1, 2)
    y = y.transpose(1, 2).reshape(B, T, -1)
    print("  拆分/合并无损:", torch.allclose(y, x))

    # H=1 时多头等价于单头（把权重同步过去比较）
    single = ScaleDotSelfAttention(D)
    multi1 = MultiHeadAttention(D, 1)
    with torch.no_grad():
        single.w_q.weight.copy_(multi1.w_q.weight); single.w_q.bias.copy_(multi1.w_q.bias)
        single.w_k.weight.copy_(multi1.w_k.weight); single.w_k.bias.copy_(multi1.w_k.bias)
        single.w_v.weight.copy_(multi1.w_v.weight); single.w_v.bias.copy_(multi1.w_v.bias)
        single.w_o.weight.copy_(multi1.w_o.weight); single.w_o.bias.copy_(multi1.w_o.bias)

    o1 = single(x)
    o2 = multi1(x)
    print("  H=1 与单头输出一致:", torch.allclose(o1, o2, atol=1e-6))


# ==================================================================
# 测试 4：padding mask（按 key 屏蔽）—— 屏蔽列权重为 0
# ==================================================================
def test_padding_mask():
    print("=" * 60)
    print("测试 4：padding mask（屏蔽第 3、5 个 key）")
    B, T, D = 2, 8, 32
    x = torch.randn(B, T, D)

    # key_mask: [B, T]，1=保留，0=屏蔽
    key_mask = torch.ones(B, T)
    key_mask[:, [3, 5]] = 0
    # 广播到 [B, 1, 1, T]，对单头则是 [B, 1, T] -> 直接用 [B,1,T] 即可
    m = ScaleDotSelfAttention(D)
    q, k, v = m.w_q(x), m.w_k(x), m.w_v(x)
    scores = q @ k.transpose(-2, -1) / (D ** 0.5)
    scores = scores.masked_fill(key_mask.unsqueeze(1) == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)

    print("  被屏蔽列权重全为0:",
          bool((attn[:, :, [3, 5]] == 0).all()))
    print("  行和≈1:", torch.allclose(attn.sum(-1), torch.ones(B, T), atol=1e-6))

    # 多头下用 [B,1,1,T]
    mh = MultiHeadAttention(D, 4)
    q = mh.w_q(x).view(B, T, 4, mh.d_k).transpose(1, 2)
    k = mh.w_k(x).view(B, T, 4, mh.d_k).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / (mh.d_k ** 0.5)
    scores = scores.masked_fill(key_mask[:, None, None, :] == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    print("  多头屏蔽列权重全为0:",
          bool((attn[:, :, :, [3, 5]] == 0).all()))


if __name__ == "__main__":
    test_single_head()
    test_multi_head()
    test_split_merge_and_h1()
    test_padding_mask()
    print("=" * 60)
    print("全部测试完成")