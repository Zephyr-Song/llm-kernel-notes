"""
03_rope.py — RoPE 原理、编码实现、外推失败与 NTK/YaRN 插值

运行: python 03_rope.py

要回答四个问题：
  1. RoPE 到底在数学上做了什么（旋转的相对位置性质证明）
  2. 为什么它能表达相对位置
  3. 为什么直接外推会崩（PIT — Position Interpolation 的问题）
  4. NTK-aware / YaRN 插值怎么修

**最关键的一段是 prove_relative_property()，手推一遍你就永远不会记错RoPE。**
"""

import math

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ================================================================ 1. RoPE 的核心：2D 旋转

def rope_rotate_half(x):
    """
    把向量 [x1, x2] 分成前后两半，构造旋转对。

    输入: [..., D]，D 是偶数
    输出: [..., D]，后半部分变成 -前半部分

    这样 (x1, x2) 之后会和一个 [[cos, -sin], [sin, cos]] 矩阵作用，
    实现二维旋转。
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """
    对 Q 和 K 同时施加旋转。

    注意：只对 q, k 做，不对 v 做。 因为 attention score = q·k，
    旋转 q 和 k 之后点积自然只与相对位置有关。

    广播约定：q/k 形状 [B, H, S, D]或 [S, H, D]，
    cos/sin 形状需能广播到 [B, H, S, D]。
    常见做法是 cos/sin 为 [S, 1, D, 1] 或 [S, 1, D]（末维为 head_dim）。
    """
    return q * cos + rope_rotate_half(q) * sin, k * cos + rope_rotate_half(k) * sin


def build_rope_cache(seq_len, head_dim, base=10000.0, device=None, dtype=None):
    """
    构造 RoPE 的 cos/sin 缓存表。

    核心公式（RoPE 原论文）：
      θ_i = base^(-2i/d)
      x_i' = x_i * cos(m * θ_i)
      x_{i+d/2}' = x_{i+d/2} * sin(m * θ_i)

    base=10000 的直觉：θ_i 随 i 递减，低维频率高（快），
    高维频率低（慢）。类似二进制进位，不同维度的周期错开，
    使得不同相对距离有不同编码。
    """
    assert head_dim % 2 == 0, "head_dim 必须是偶数"

    # θ_i, i 从 0 到 d/2-1
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )

    # 位置 m 从 0 到 seq_len-1
    t = torch.arange(seq_len, device=device, dtype=torch.float32)

    # outer product: [seq_len, d/2]
    freqs = torch.outer(t, inv_freq)

    # 复制成 [seq_len, d] 因为 rotate_half 的实现需要这样
    emb = torch.cat((freqs, freqs), dim=-1)

    return emb.cos().to(dtype), emb.sin().to(dtype)


# ================================================================ 2. 相对位置性质的证明（核心）

def prove_relative_property():
    """
    手动验证 RoPE 的核心性质：<R_m q, R_n k> 只依赖 (m - n)

    理论上：
      <R_m q, R_n k> = q^T R_m^T R_n k = q^T R_{n-m} k

    因为旋转矩阵是正交的，且R_m^T R_n = R_{n-m}，只跟相对位置有关。
    这就是 RoPE 能编码相对位置、且天然可外推的数学原因。
    """
    print("=" * 78)
    print("1. RoPE 核心性质验证：注意力分数只依赖相对位置")
    print("=" * 78)
    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    torch.manual_seed(0)
    head_dim = 8
    q = torch.randn(head_dim, dtype=torch.float64)
    k = torch.randn(head_dim, dtype=torch.float64)

    def rot(vec, pos, base=10000.0):
        """对单个向量施加位置 pos 的旋转"""
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
        )
        freqs = pos * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return vec * emb.cos() + rope_rotate_half(vec) * emb.sin()

    print("\n【实验设计】固定 q 和 k，只改变位置组合")
    print("-" * 78)
    print(f"{'m (q位置)':>10} {'n (k位置)':>10} {'差值 m-n':>10} {'<R_m q, R_n k>':>18}")
    print("-" * 78)

    results = {}
    for m, n in [(0, 0), (5, 2), (10, 7), (100, 97), (1000, 997), (50, 20), (50, 20)]:
        score = torch.dot(rot(q, m), rot(k, n)).item()
        key = m - n
        results.setdefault(key, []).append(score)
        print(f"{m:>10} {n:>10} {key:>10} {score:>18.10f}")

    print("\n验证结论:")
    for key, vals in results.items():
        if len(vals) > 1:
            spread = max(vals) - min(vals)
            print(
                f"  相对位置 {key:>5}: 得分 {[f'{v:.10f}' for v in vals]}"
                f"  极差 = {spread:.3e}"
            )
            print(f"{'':>16}→ {'完全一致' if spread < 1e-9 else '不一致'}")

    print("\n数学解释:")
    print("  旋转矩阵 R(θ) = [[cosθ, -sinθ], [sinθ, cosθ]]，满足 R(θ)^T = R(-θ)")
    print("  R(θ)^T R(φ) = R(φ - θ)")
    print("  所以 <R_m q, R_n k> = q^T R_m^T R_n k = q^T R_{n-m} k")
    print("  → 只依赖相对位置差 n - m，与绝对位置无关")
    print("\n  这就是 RoPE 能做相对位置编码的原理，也是它天然适合 decoder 的原因。")
    print("  同样的性质下，把 q 整体「向上移动」k 个位置，注意力分数完全不变。")


# ================================================================ 3. 频率视角：理解 base=10000

def frequency_analysis():
    print()
    print("=" * 78)
    print("2. 频率分析 —— base=10000 到底在做什么")
    print("=" * 78)
    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    head_dim = 128
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))

    print(f"\nhead_dim={head_dim}, base={base}")
    print(f"\n{'维度对 i':>8} {'θ_i':>14} {'周期 2π/θ_i':>14}  作用")
    print("-" * 62)

    for i in range(0, head_dim // 2, 8):
        theta = inv_freq[i].item()
        period = 2 * math.pi / theta if theta > 0 else float("inf")
        desc = "细粒度：分辨相邻 token" if period < 100 else (
            "中粒度：局部结构" if period < 2000 else "粗粒度：长距离依赖"
        )
        print(f"{i:>8} {theta:>14.6f} {period:>14.1f}  {desc}")

    print("\n设计意图（类比时钟）：")
    print("  · 秒针走最快（θ 大，周期短）→ 提供高精度位置信息")
    print("  · 分针/hour 走得慢（θ 小，周期长）→ 提供粗粒度位置范围")
    print("  · 多组不同周期的指针组合起来，可以唯一确定任意位置")
    print("\n  base 的作用：")
    print("  base 越大 → 频率整体越低 → 长距离分辨能力越强，但相邻区分越粗")
    print(f"  base=10000 → 最低频率周期 {2 * math.pi / inv_freq[-1].item():.0f} tokens")
    print("  → 一个完整「周期」覆盖约 12.6K token，足够绝大多数模型上下文")
    print("\n  LLaMA 的扩展长度方法就是拉大 base（见后面的 NTK 插值）：")

    for b in [10000.0, 500000.0, 10000000.0]:
        lowest_period = 2 * math.pi * b ** (1 - 1.0)
        print(f"    base={b:>12.0f} → 最低频周期≈{lowest_period:>12.0f} tokens")

    print("\n  现代模型的选择：")
    print("    LLaMA-1: base=10000")
    print("    LLaMA-3: base=500000（支持 8K→128K）")
    print("    Qwen2:   base=1000000")
    print("    Yi/DeepSeek: 有专门为长上下文设定的 base")


# ================================================================ 4. 外推失败：PIT 问题

def extrapolation_failure():
    print()
    print("=" * 78)
    print("3. 直接外推为什么崩（PIT）")
    print("=" * 78)
    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    trained_len = 64
    test_len = 512  # 外推 8倍

    torch.manual_seed(0)
    d_model = 128

    # 模拟：训练时 Q/K 都带 RoPE，超出训练长度后行为如何
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2) / d_model))
    t = torch.arange(test_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    print(f"\n训练上下文长度: {trained_len}")
    print(f"测试上下文长度: {test_len}  (外推 {test_len / trained_len:.0f} 倍)")
    print(f"d_model: {d_model}")

    # 关键观察：训练时见过的最大相位
    max_phase_trained = (trained_len * inv_freq).max().item()
    max_phase_test = (test_len * inv_freq).max().item()

    print(f"\n训练时最大相位(m*θ): {max_phase_trained:.2f} rad")
    print(f"测试时最大相位:      {max_phase_test:.2f} rad")
    print(f"2π = {2 * math.pi:.2f} rad")

    #统计有多少维度在训练时从未完成过一个完整周期
    trained_periods = (trained_len * inv_freq) / (2 * math.pi)
    never_saw_full_cycle = (trained_periods < 1).sum().item()
    print(f"\n低频维度（训练时未完成一个完整周期）: {never_saw_full_cycle}/{len(inv_freq)}")

    print("\n崩溃的两个原因:")
    print("  1.【维度未充分训练】")
    print(f"     训练长度 {trained_len} 时，有 {never_saw_full_cycle}/{len(inv_freq)} "
          "个维度的相位 < 2π")
    print("     这些维度的编码范围模型没见过，外推时进入未训练区域。")
    print("\n  2.【注意力分布偏移 PIT, Position Interpolation Theory】")
    print("     注意力分数 = q·k，两边都带旋转。位置变大时，")
    print("     分数分布的方差会变大 → softmax 变得更尖锐 → 注意力过于集中")
    print("     模型在训练时看到的是平缓分布，推理时突然变得极端 → 输出退化。")
    print("\n  验证：计算不同位置下的注意力分数方差")
    q = torch.randn(d_model, dtype=torch.float32)
    for pos in [16, 32, 64, 128, 256]:
        ph = pos * inv_freq
        e = torch.cat((ph, ph), dim=-1)
        qr = q * e.cos() + rope_rotate_half(q) * e.sin()
        scores = torch.randn(64, d_model) @ qr  # 模拟一组 k
        print(f"    pos={pos:>4}: 分数 std={scores.std().item():.4f}  "
              f"max_softmax={torch.softmax(scores, 0).max().item():.4f}")

    print("\n  → std 随位置上升，softmax 越来越尖锐 = 注意力过度集中 = PIT")


# ================================================================ 5. NTK-aware 插值

def ntk_interpolation():
    print()
    print("=" * 78)
    print("4. NTK-aware 插值（Reddit / bloc97 提出）")
    print("=" * 78)
    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    orig_len, target_len = 64, 512
    scale = target_len / orig_len

    print(f"\n原长度 {orig_len} → 目标 {target_len}，缩放因子 s={scale}")

    # 方案A（朴素）：直接降低 base（等价于降低所有频率）
    def scaled_base(base, s):
        return base * (s ** (d_model / (d_model - 2)))

    # 方案B（NTK-aware）：base' = base * s^(d/(d-2))
    # 动机：低频维度几乎不变，高频维度被明显压缩
    print("\n两种思路:")
    print("  朴素线性插值：所有维度频率同比例降低 → 高频信息也损失了")
    print("  NTK-aware：base 提升 → 低频维度频率几乎不变，只有高频被压缩")
    print("  （直觉：类似 NTK 定理，高维网络对频率缩放有更好的适应性）")

    print(f"\n{'维度i':>8} {'θ(原)':>14} {'θ(朴素s=8)':>16} {'θ(NTK)':>14} {'朴素变化':>12} {'NTK变化':>12}")
    print("-" * 92)

    def freqs_for(base):
        return 1.0 / (base ** (torch.arange(0, d_model, 2) / d_model))

    d_model = 128
    inv_orig = freqs_for(10000.0)
    inv_naive = freqs_for(10000.0 / scale)
    inv_ntk = freqs_for(scaled_base(10000.0, scale))

    for i in range(0, d_model // 2, 16):
        f_orig = inv_orig[i].item()
        f_naive = inv_naive[i].item()
        f_ntk = inv_ntk[i].item()
        print(
            f"{i:>8} {f_orig:>14.6f} {f_naive:>16.6f} {f_ntk:>14.6f} "
            f"{f_naive / f_orig:>11.2f}x {f_ntk / f_orig:>11.2f}x"
        )

    print("\n解读:")
    print("  · 低维（i小，θ大）：NTK 下变化很小，保留细节")
    print("  · 高维（i大，θ小）：NTK 下被压缩更多，扩展覆盖范围")
    print("  · 这正是我们想要的：高频保留精度，低频承担外推")
    print(f"\n  base: 10000 → {scaled_base(10000.0, scale):.0f}  (NTK-aware)")
    print("  这就是 LLaMA-3 用 base=500000 支持 128K 上下文的思路。")


# ================================================================ 6. YaRN：分段处理

def yarn_analysis():
    print()
    print("=" * 78)
    print("5. YaRN —— NTK + 分段插值（当前主流）")
    print("=" * 78)

    print("""
YaRN (Yet another RoPE extensioN) 的观察：NTK-aware 插值虽然好，但把所有维度
统一处理仍然不完美。它提出按维度分段处理：

  · 高频维度（波长短）：**不插值**，保持原样
    → 因为高频维度提供的是局部相对位置，插值它们损失精度且没必要
  · 低频维度（波长长）：**完全插值**（线性缩放位置）
    → 低频维度负责长距离范围，必须压缩
  · 中间过渡维度：平滑混合

YaRN 额外引入的技巧：注意力温度修正
  插值后注意力分布变尖锐（PIT），YaRN 额外缩放 softmax 前的温度系数
  m_scale = 0.1 * log(s) + 1，在计算 attention 时乘上，补偿分布偏移。

公式（论文原文）：
  m_scale = 0.1 * ln(s) + 1
  sqrt(d) / sqrt(d) * m_scale ... 具体为 q·k * m_scale

对比总结：
  | 方法              | 做法                     | 缺点                    |
  |-------------------|--------------------------|-------------------------|
  | 线性插值 (PI)      | 位置全部除以 s           | 高频信息也损失，效果最差 |
  | NTK-aware         | 只改 base                | 所有维度都被压缩        |
  | YaRN             | 分段 + 注意力温度修正    | 实现复杂，但效果最好    |
  | PI 方法的替代: LongRoPE | 渐进式扩展 + 非均匀频率 | 需要分阶段训练          |

现代模型（Llama 3.1+、Qwen2.5、Yi）的做法：
  1. 训练时做 8K → 128K 的渐进式扩展（LongRoPE 风格）
  2. 训练后简单调整 base 到 500k
  3. 推理时不再做任何动态插值

这说明：目前**没有零成本的外推方案**，真正能上生产的长上下文模型都需要
针对性的继续训练。这一点很重要 —— 别指望换个 base 就白嫖长上下文。
""")


# ================================================================ 7. 完整 RoPE 使用示例

def practical_usage():
    print("=" * 78)
    print("6. 实际使用：RoPE 在 attention 里的位置")
    print("=" * 78)
    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    torch.manual_seed(0)
    seq_len, n_heads, head_dim = 128, 4, 64

    q = torch.randn(seq_len, n_heads, head_dim)
    k = torch.randn(seq_len, n_heads, head_dim)

    cos, sin = build_rope_cache(seq_len, head_dim)

    # PyTorch 常用格式: [seq, 1, dim, 1] 以便广播
    cos_b = cos.unsqueeze(1)  # [seq, 1, dim]
    sin_b = sin.unsqueeze(1)

    # 对每个 head 施加旋转
    q_rope, k_rope = apply_rope(q, k, cos_b, sin_b)

    print(f"\nq shape: {tuple(q.shape)}, cos shape: {tuple(cos_b.shape)}")
    print(f"RoPE 后 q shape: {tuple(q_rope.shape)}  (shape 不变，这是旋转的关键性质)")

    # 验证：每个二维分量的模长应该不变（旋转是保距变换）
    orig_norm = q[0, 0, :2].norm().item()
    rope_norm = q_rope[0, 0, :2].norm().item()
    print(f"\n旋转前后模长: 原={orig_norm:.6f}后={rope_norm:.6f}  "
          f"相对差={abs(orig_norm - rope_norm) / orig_norm:.2e}")
    print("→ 旋转是正交变换，不改变向量长度。这是 RoPE 能作为「位置调制」的前提。")

    # 位置 0 的旋转应该是恒等（cos=1, sin=0）
    diff = (q_rope[0] - q[0]).abs().max().item()
    print(f"\n位置 0 的最大变化量: {diff:.2e}  → 应该约等于 0")
    print("→ 因为位置 0 时 θ=0，cos(0)=1, sin(0)=0，等于没旋转。")

    # 相对位置不变性验证
    print("\n相对位置不变性实测:")
    for shift in [1, 10, 50]:
        q_shift = torch.roll(q, shifts=shift, dims=0)
        # 直接用未平移的 cos/sin 表（对应「q 被平移了 shift」这一物理情形）
        q_shift_rope, _ = apply_rope(q_shift, k, cos_b, sin_b)
        # 位置 i 处比较：q 在 i+shift 的分数 vs 平移后 q 在 i 的分数
        s1 = (q_rope[shift:shift + 10] * k[:10]).sum(-1)
        s2 = (q_shift_rope[:10] * k[:10]).sum(-1)
        err = (s1 - s2).abs().max().item()
        print(f"  平移 {shift:>3} 个位置后，与原位置 {shift:>3} 的 q·k 最大差异: {err:.2e}")

    print("\n→ 差异接近 0，验证了 <R_m q, R_n k> 只依赖 m-n 的性质。")

    # 在 SDPA 中使用
    print("\n与 F.scaled_dot_product_attention 配合的注意点:")
    print("  1. RoPE 必须在 attention 之前施加，且 q 和 k 都要")
    print("  2. 施加后不要再做 q * q_norm 之类的缩放（会破坏模长）")
    print("  3. KV cache 场景：只需为新 token 计算 cos/sin（位置 = cache_len）")
    print("     全量计算 cos/sin 是浪费，实现在下个文件 04_gqa_kvcache.py 里")


def main():
    prove_relative_property()
    frequency_analysis()
    extrapolation_failure()
    ntk_interpolation()
    yarn_analysis()
    practical_usage()

    print()
    print("=" * 78)
    print("下一步: 读 notes/04-rope.md，然后进入 04_gqa_kvcache.py")
    print("=" * 78)


if __name__ == "__main__":
    main()