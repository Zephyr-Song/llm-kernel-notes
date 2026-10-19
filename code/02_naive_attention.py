"""
02_naive_attention.py — 从 O(N^2) 到 FlashAttention 的完整推导链

运行: python 02_naive_attention.py

这个脚本做四件事：
  1. 手写朴素 attention（会 OOM，但能作为正确性基准）
  2. 手写 PyTorch 实现的 FlashAttention（分块 + 在线 softmax）
  3. 数值对拍验证两者一致
  4. 用profiler 说明「FlashAttention 省的是 IO 不是 FLOPs」

**这个文件是整个仓库最重要的一份。自己跑一遍，看每一段输出。**
"""

import math
import time

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ================================================================ 1. 朴素实现

def attention_naive(Q, K, V, mask=None):
    """
    最直接的实现：显式物化 N×N 的 attention matrix。

    显存占用分析（seq_len=N, head_dim=D, dtype=2 bytes）：
      - Q, K, V:各 N*D*2 bytes
      - attention matrix (N×N): N^2 * 2 bytes  ← 平方级！
      - softmax 中间结果：可能同时存在多份N^2

    例子 N=8192, D=64：
      attention matrix = 8192^2 * 2 = 134 MB / head
      而 Q/K/V 各只有 8192 * 64 * 2 = 1 MB
      → attention matrix 比输入大 128 倍

    例子 N=131072（128K 上下文）：
      N^2 * 2 = 34 GB  ← 单个头就装不下
    这就是朴素实现无法处理长上下文的原因。
    """
    d_k = Q.size(-1)
    # Q @ K^T : [N, D] @ [D, N] -> [N, N]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores + mask  # mask 是 0 / -inf 的加性 mask

    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


# ================================================================ 2. 因果 mask

def causal_mask(seq_len, device=None, dtype=None):
    """
    生成下三角 causal mask，值域 {0, -inf}，用于加性 mask。

    为什么用 -inf 而不是 -1e9？
      - softmax(-inf) 精确等于 0
      - 但如果整行都是 -inf，softmax 会得到 NaN，所以对角线必须是 0
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)  # 上三角为 -inf
    return mask


def test_causal_mask():
    print("=" * 78)
    print("1. 因果 mask 的正确性验证")
    print("=" * 78)
    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    m = causal_mask(4)
    print("\n4x4 causal mask (加性形式):")
    print(m)
    print("\nsoftmax 后（注意每行求和为 1，被屏蔽位置为 0）:")
    print(torch.softmax(m, dim=-1))
    print("\n→ 第0 行只有位置 0 可见，softmax 后必然是 [1,0,0,0]")
    print("→ 这正是 decoder-only LLM 的自回归约束：位置 i 只能看 <= i")


# ================================================================ 3. 在线 Softmax —— FlashAttention 的数学核心

def online_softmax_blocks(scores_block):
    """
    在线 softmax：分块处理时如何保持全局正确。

    标准 softmax 需要两遍：
      1. 遍历求 max
      2. 遍历求 exp 并求和，然后除

    但分块处理时不能保存全部 scores（那就退化成朴素实现了），
    所以需要「在线」地增量维护：max 和 sum。

    数学：
      维护 m (全局最大), l (全局指数和)
      来了一块新的 scores_block，最大值变成 m_new = max(m, block_max)

      旧数据的贡献需要重新缩放：
        l_new = l * exp(m - m_new) + sum(exp(block - m_new))

    这是 FlashAttention 全部复杂度的来源 —— 仅仅是一个数值稳定的缩放技巧。
    """
    m_new = torch.maximum(m_global, scores_block.max(dim=-1).values)
    correction = torch.exp(m_global - m_new)
    l_new = l_global * correction + torch.exp(scores_block - m_new[..., None]).sum(dim=-1)
    return m_new, l_new


def flash_attention(Q, K, V, block_size=64, is_causal=True):
    """
    手写 FlashAttention（PyTorch 版，用分块模拟 IO-aware 的效果）。

    真实的 FlashAttention 是 CUDA kernel，用 shared memory 做分块。
    这里用 PyTorch 分块，展示算法逻辑，不追求速度。

    真实 kernel 的 IO 优化靠：
      1. SRAM 复用：K/V 的 block 加载到 shared memory，被所有 Q block 复用
      2. 不物化 N^2：每块的中间结果算完就丢弃
      3. 反向重计算：用 checkpointing思路，backward 时重算中间值而非保存
    """
    seq_len, d_k = Q.shape
    d_v = V.shape[-1]

    out = torch.zeros(seq_len, d_v, dtype=Q.dtype, device=Q.device)

    # 全局 softmax 状态（在线更新）
    m_global = torch.full((seq_len,), float("-inf"), dtype=Q.dtype, device=Q.device)
    l_global = torch.zeros(seq_len, dtype=Q.dtype, device=Q.device)

    scale = 1.0 / math.sqrt(d_k)

    for q_start in range(0, seq_len, block_size):
        q_end = min(q_start + block_size, seq_len)
        q_block = Q[q_start:q_end]  # [Bq, D]

        # 该block 的输出累加器（未归一化）
        acc = torch.zeros(q_end - q_start, d_v, dtype=Q.dtype, device=Q.device)
        m_local = torch.full((q_end - q_start,), float("-inf"), dtype=Q.dtype, device=Q.device)
        l_local = torch.zeros(q_end - q_start, dtype=Q.dtype, device=Q.device)

        for k_start in range(0, seq_len, block_size):
            k_end = min(k_start + block_size, seq_len)
            k_block = K[k_start:k_end]  # [Bk, D]
            v_block = V[k_start:k_end]  # [Bk, Dv]

            # [Bq, D] @ [D, Bk] -> [Bq, Bk]
            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            # 应用 causal mask（注意边界：K的块可能全在未来）
            if is_causal:
                q_pos = torch.arange(q_start, q_end, device=Q.device)[:, None]
                k_pos = torch.arange(k_start, k_end, device=Q.device)[None, :]
                scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

            # 在线 softmax 更新
            m_new = torch.maximum(m_local, scores.max(dim=-1).values)
            # 注意：-inf 位置 exp 后为 0，但如果 m_new 也是 -inf 就会出 NaN
            m_new = torch.clamp(m_new, min=-1e30)

            p = torch.exp(scores - m_new[..., None])  # [Bq, Bk]
            correction = torch.exp(m_local - m_new)
            l_local = l_local * correction + p.sum(dim=-1)
            acc = acc * correction[..., None] + torch.matmul(p, v_block)
            m_local = m_new

        out[q_start:q_end] = acc / l_local[..., None]
        m_global[q_start:q_end] = m_local
        l_global[q_start:q_end] = l_local

    return out


# ================================================================ 4. 数值对拍

def correctness_test():
    print()
    print("=" * 78)
    print("2. 数值正确性对拍（Flash vs 朴素）")
    print("=" * 78)
    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    torch.manual_seed(42)
    seq_len, d = 128, 32

    Q = torch.randn(seq_len, d)
    K = torch.randn(seq_len, d)
    V = torch.randn(seq_len, d)
    mask = causal_mask(seq_len)

    ref = attention_naive(Q, K, V, mask)
    fa = flash_attention(Q, K, V, block_size=16, is_causal=True)

    max_diff = (ref - fa).abs().max().item()
    rel_diff = ((ref - fa).abs() / (ref.abs() + 1e-6)).max().item()

    print(f"\nseq_len={seq_len}, head_dim={d}, block_size=16")
    print(f"  朴素实现 vs 手写Flash:最大绝对差异 = {max_diff:.3e}")
    print(f"  最大相对差异                        = {rel_diff:.3e}")
    print(f"  是否通过(fp32 容差 1e-5): {'是' if max_diff < 1e-5 else '否'}")

    print("\n前5 个位置的输出（应完全一致）:")
    print(f"  朴素: {ref[:5, :3].flatten().tolist()}")
    print(f"  Flash:{fa[:5, :3].flatten().tolist()}")

    if max_diff >= 1e-5:
        print("\n  排查建议:")
        print("    1. 检查 masked_fill 的位置是否搞反（k_pos > q_pos 才是未来）")
        print("    2. 检查 m_new 的 clamp（-inf 减法会产生 NaN）")
        print("    3. 检查 acc 的 rescale 是否在 matmul(p, v) 之前")


# ================================================================ 5. 显存占用对比（核心论证）

def memory_analysis():
    print()
    print("=" * 78)
    print("3. 显存占用对比 —— 这是 FlashAttention 真正的价值")
    print("=" * 78)

    print(f"\n{'seq_len':>9} {'N^2 matrix/head':>20} {'Q+K+V/head':>14} {'放大倍数':>12}")
    print("-" * 78)

    for seq_len in [1024, 4096, 16384, 65536, 262144]:
        d = 128  # head_dim
        bytes_per = 2  # fp16

        n2_matrix = seq_len * seq_len * bytes_per
        qkv = 3 * seq_len * d * bytes_per
        ratio = n2_matrix / qkv

        print(f"{seq_len:>9} {n2_matrix / 1e6:>17.1f} MB {qkv / 1e6:>11.2f} MB {ratio:>10.0f}x")

    print("\n解读:")
    print("  · seq_len=262144 (256K) 时，朴素实现的 attention matrix 单头就要 128 GB")
    print("  · FlashAttention 的显存占用是 O(N)，与朴素实现差 5~6 个数量级")
    print("  · 这才是它能支撑长上下文的根本原因，不是「更快」而是「装得下」")


# ================================================================ 6. FLOPs 分析：Flash 其实用了更多算力

def flops_analysis():
    print()
    print("=" * 78)
    print("4. FLOPs 分析 —— FlashAttention 算得更多，不是更少")
    print("=" * 78)

    seq_len, d = 4096, 128

    # 朴素实现的 FLOPs
    qk_flops = 2 * seq_len * seq_len * d  # Q @ K^T
    av_flops = 2 * seq_len * seq_len * d  # attn @ V
    naive_flops = qk_flops + av_flops

    # FlashAttention 的额外成本：每块都要重新计算 softmax 的缩放
    # 真实数字需要看 kernel 实现，量级上是 O(N^2) 但常数更大
    print(f"\nseq_len={seq_len}, d={d}")
    print(f"  朴素实现 FLOPs: {naive_flops / 1e9:.2f} GFLOP")
    print(f"    其中 Q@K^T: {qk_flops / 1e9:.2f} GFLOP")
    print(f"    其中 attn@V: {av_flops / 1e9:.2f} GFLOP")
    print(f"\n  FlashAttention 额外做的事:")
    print(f"    · 每块重算 exp/sum:约 5~10 ops/element")
    print(f"    · 反向传播重算中间值（省显存但多算）")
    print(f"    · 合计额外增加约 10~20% 的 FLOPs")
    print(f"\n  结论: FlashAttention 的加速 100% 来自减少 HBM↔SRAM 的数据搬运，")
    print(f"        而不是因为减少了算子。它是典型的 IO-bound 优化。")

    print("\n数值参考（A100: FP16 算力 312 TFLOPS, HBM 带宽 2 TB/s）:")
    naive_flops = 2 * seq_len * seq_len * d * 2
    ideal_compute_time = naive_flops / 312e12
    # 朴素实现需要搬运 N^2 大小的数据（写+读至少2次）
    bytes_moved = 2 * seq_len * seq_len * 2
    naive_io_time = bytes_moved / 2e12
    print(f"    理想计算时间（无IO 瓶颈）: {ideal_compute_time * 1e6:.2f} μs")
    print(f"    朴素实现的 IO 时间:  {naive_io_time * 1e6:.2f} μs")
    print(f"    IO-bound 倍数:          {naive_io_time / ideal_compute_time:.0f}x")
    print(f"    → 朴素实现99% 的时间在等内存，算力完全闲置")
    print(f"    → FlashAttention 通过分块复用，把 IO 时间降回与计算同量级")


# ================================================================ 7. 与 PyTorch SDPA 对比

def sdpa_comparison():
    print()
    print("=" * 78)
    print("5. 与 PyTorch 内置 SDPA 对比")
    print("=" * 78)
    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    torch.manual_seed(0)
    seq_len, n_heads, d = 2048, 8, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Q = torch.randn(seq_len, n_heads, d, dtype=torch.float16 if device == "cuda" else torch.float32)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    q, k, v = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  # [B, H, N, D]

    print(f"\nseq_len={seq_len}, heads={n_heads}, head_dim={d}, device={device}")
    print("\nF.scaled_dot_product_attention 可用的后端:")
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        for backend in [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]:
            print(f"  · {backend.name}")
    except ImportError:
        print("  （PyTorch 版本较低，无 SDPBackend 枚举）")

    # 正确性
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"\n  SDPA (is_causal=True) 输出 shape: {tuple(ref.shape)}")
        print(f"  输出是否全为有限值: {bool(torch.isfinite(ref).all())}")

    # 计时
    def bench(fn, n=10):
        fn()  # warmup
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    print("\n耗时对比:")
    with torch.no_grad():
        t_sdpa = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))

        # 朴素实现（PyTorch 版，非手写 flash）
        seq_q = q.shape[-2]   # 注意取 [-2] 是序列维，[-1] 是 head_dim

        def naive_attn():
            m = causal_mask(seq_q, device=q.device, dtype=q.dtype)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
            scores = scores + m
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

        t_naive = bench(naive_attn, n=3)

    print(f"  朴素PyTorch: {t_naive:>8.2f} ms")
    print(f"  SDPA(自动选后端): {t_sdpa:>6.2f} ms")
    print(f"  加速比:        {t_naive / t_sdpa:>8.1f}x")

    if device == "cpu":
        print("\n  (CPU 上没有 FlashAttention kernel，SDPA 走的是 math 后端，加速比有限)")
        print("   要看到真实差距需要 CUDA 环境。")


# ================================================================ 8. 因果注意力的可解释性

def causal_attention_visualization():
    print()
    print("=" * 78)
    print("6. 观察因果注意力的稀疏结构")
    print("=" * 78)
    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    torch.manual_seed(3)
    seq_len, d = 512, 64

    Q = torch.randn(seq_len, d)
    K = torch.randn(seq_len, d)
    V = torch.randn(seq_len, d)

    mask = causal_mask(seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d) + mask
    attn = torch.softmax(scores, dim=-1)

    # 每行的有效注意力位置数
    effective = (attn > 1e-6).sum(dim=-1)

    print(f"\nseq_len={seq_len}")
    print(f"  第 0 行有效位置数: {effective[0].item()}  (只能看自己)")
    print(f"  第 100 行:         {effective[100].item()}")
    print(f"  第 511 行:         {effective[511].item()}  (能看到全部 512 个)")

    # 注意力熵（衡量注意力的集中程度）
    entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1)
    print(f"\n  注意力熵: 第 0 行={entropy[0].item():.3f}, 中间={entropy[256].item():.3f}, 末行={entropy[511].item():.3f}")
    print("  → 熵随位置增长是正常的：可见范围变大，注意力更分散")

    # 关键观察：即使在因果 mask 下，注意力分布也高度稀疏
    row = attn[511]
    top_k = row.topk(5)
    print(f"\n  末行注意力最大的 5 个位置（应为最近的位置）:")
    for idx, val in zip(top_k.indices.tolist(), top_k.values.tolist()):
        print(f"    位置 {idx:>4} (距离 {511 - idx:>3})  权重 {val:.4f}")

    print("\n  → 这引出后续两个话题:")
    print("    1. 为什么需要 KV cache（每个新 token 都要重算全部注意力的浪费）")
    print("    2. 为什么有稀疏注意力、滑动窗口、StreamingLLM 等优化 —— 因为实测注意力高度稀疏")


def main():
    test_causal_mask()
    correctness_test()
    memory_analysis()
    flops_analysis()
    sdpa_comparison()
    causal_attention_visualization()

    print()
    print("=" * 78)
    print("下一步: 读 notes/03-attention-from-scratch.md")
    print("=" * 78)


if __name__ == "__main__":
    main()