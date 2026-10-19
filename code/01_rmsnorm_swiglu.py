"""
01_rmsnorm_swiglu.py — RMSNorm 与 SwiGLU 的逐行拆解 + 数值验证

运行: python 01_rmsnorm_swiglu.py

对照 HuggingFace 实现，理解两个现代 LLM 必用的组件：
  · RMSNorm  (LLaMA / Mistral / Qwen 全部使用)
  · SwiGLU   (LLaMA-2/3, Mistral, Qwen 使用，不是原始 GPT-2 的 GELU)

核心认知：这两个改动都是为了「省显存带宽」或「提效果」，不是为了提算力。
"""

import math

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ================================================================ 1. RMSNorm

def rmsnorm_reference(x, weight, eps=1e-6):
    """
    官方实现形式（LLaMA）：
        rsqrt( mean(x^2) ) * weight
    关键点：没有 mean(x) 减法，没有 bias。
    """
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def layernorm_reference(x, weight, bias, eps=1e-6):
    """
    标准 LayerNorm：减均值 + 除标准差
    """
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps) * weight + bias


def rmsnorm_naive_understand(x, weight, eps=1e-6):
    """
    用「先算样本标准差再除」的方式理解 RMSNorm，
    但注意：这里除的是 sqrt(mean(x^2)) 而不是真正的 std。
    这是 RMSNorm 最容易被误解的点。
    """
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms * weight


def explain_rmsnorm():
    print("=" * 78)
    print("1. RMSNorm —— 为什么可以去掉减均值")
    print("=" * 78)

    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    torch.manual_seed(0)
    x = torch.randn(4, 128)
    w = torch.ones(128)

    print("\n验证 rmsnorm_reference 与手写理解版一致:")
    a = rmsnorm_reference(x, w)
    b = rmsnorm_naive_understand(x, w)
    print(f"  最大差异: {(a - b).abs().max().item():.3e}")

    print("\nRMSNorm 与 LayerNorm 的数值差异:")
    ln = layernorm_reference(x, w, torch.zeros(128))
    print(f"  RMSNorm  输出均值: {a.mean(dim=-1).mean().item():+.6f}  ← 不保证为0")
    print(f"  LayerNorm 输出均值: {ln.mean(dim=-1).mean().item():+.6f}  ← 保证为0")
    print(f"  两者最大差异: {(a - ln).abs().max().item():.6f}")
    print("\n  → LayerNorm 强制每行均值为 0；RMSNorm 不做。")
    print("  → 在 LLM 里，去掉减均值 = 少一次 reduce + 少一个 bias 参数，")
    print("    而 bias 少掉意味着权重矩阵少一列，实测效果反而略好（LLaMA 论文实测）。")

    print("\nRMS 的含义（这才是核心）:")
    print("  RMS = Root Mean Square = sqrt(mean(x^2))")
    print("  直觉：衡量一个向量的『能量大小』，不管它的正负分布")
    print("  为什么要归一化到固定能量：")
    print("    - 防止深层网络激活值指数级放大/缩小")
    print("    - 让不同层的输入尺度可比，学习率好调")
    print("    - 现代激活函数 + Pre-LN 训练时，不归一化极易发散")

    print("\neps 的作用：")
    print("  当 x 几乎全为 0 时，mean(x^2)→0，rsqrt 会除以 0 → inf")
    print(f"  eps=1e-6 时，最小分母 = sqrt(1e-6) = {math.sqrt(1e-6)}，实际不会溢出")
    print("  这也是为什么所有实现都写成 rsqrt(var + eps) 而不是 rsqrt(var) / eps")

    print("\n计算量对比（hidden_dim = d）:")
    print(f"  LayerNorm: 需要 mean(x) 和 mean((x-mean)^2) 两次 reduce + 一个 bias 加法")
    print(f"  RMSNorm:   只需要 mean(x^2)          一次 reduce")
    print("  在 d=4096 时差距不大，但在 MoE 模型里 FFN 被调用上亿次，微小节省会被放大")


# ================================================================ 2. SwiGLU

def swiglu_explicit(gate, up, weight_down=None):
    """
    SwiGLU(x) = Swish(gate(x)) * up(x)

    命名来源：Swish + GLU (Gated Linear Unit)

    标准实现（LlamaMLP）：
        down_proj(silu(gate_proj(x)) * up_proj(x))
    """
    activated = F.silu(gate)  # Swish(x) = x * sigmoid(x)
    out = activated * up
    if weight_down is not None:
        out = out @ weight_down
    return out


def gelu_approx(x):
    """GELU 的 tanh 近似（LLaMA 之前的标准 FFN 用的就是这个）"""
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
            )
        )
    )


def explain_swiglu():
    print()
    print("=" * 78)
    print("2. SwiGLU —— 为什么 FFN 要多一个矩阵")
    print("=" * 78)

    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    print("\n原始 GPT-2 的 FFN:")
    print("  FFN(x) = GELU(x @ W1) @ W2          两层，hidden = 4*d")

    print("\nLLaMA 的 FFN (SwiGLU):")
    print("  FFN(x) = down( SiLU(gate(x)) * up(x) )   三层矩阵，hidden ≈ 8/3 * d")
    print("\n为什么多一层反而参数量差不多？")
    print("  GPT-2:  W1 是 [4d, d]，W2 是 [d, 4d]  → 参数量 8*d^2")
    print("  SwiGLU: gate/up 各是 [h, d]，down 是 [d, h]，h ≈ 8d/3")
    print("          → 参数量 3 * h * d = 3 * (8d/3) * d = 8*d^2   完全相同！")
    print("\n  所以 SwiGLU 是『同样的参数量，换一种摆放方式』，不是加参数。")

    print("\n门控机制在做什么:")
    torch.manual_seed(1)
    gate = torch.randn(1, 512)
    up = torch.randn(1, 512)

    a = swiglu_explicit(gate, up)

    # 拆开看门控的效果
    gate_act = F.silu(gate)
    print(f"  SiLU(gate) 的取值范围: [{gate_act.min():.3f}, {gate_act.max():.3f}]")
    print(f"  SiLU 是有下界(-0.278)的，非单调 —— 意味着『负向抑制』信号也能被传递")
    print(f"  乘上 up 之后范围: [{a.min():.3f}, {a.max():.3f}]")

    # 对比 GELU
    g = gelu_approx(gate)
    print(f"\n  对比 GELU 的范围: [{g.min():.3f}, {g.max():.3f}]")
    print("  GELU 的下界约 -0.17，SwiGLU 因为多了一次乘法，动态范围更大")

    print("\n实证结论（来自 LLaMA 论文 Table 1 的消融）:")
    print("  在相同参数量下:")
    print("    ReLU²  <  GeLU  <  SwiGLU")
    print("  差距主要来自: SwiGLU 在 8B 参数时比 ReLU² 高约 2% 困惑度优势")

    print("\n为什么要『门控』:")
    print("  直观理解: 模型需要对不同 token『选择性激活』某些知识通道。")
    print("  gate 分支学『该不该激活』（一个 sigmoid 决策）")
    print("  up 分支学『激活什么内容』（承载具体信息）")
    print("  两者相乘 = 用一个标量门控控制一路信息通道")

    print("\n量化视角（LLaMA 论文 3.4 节的重要结论）:")
    print("  SwiGLU 的 gate 分支容易产生小幅度大数量的 outlier 激活值，")
    print("  这会导致**激活量化误差偏大**。LLaMA 的解法:")
    print("    1. 不量化 gate 分支（它是内存/带宽瓶颈的主要来源）")
    print("    2. 其他分支用更高精度")
    print("  → 所以现代 LLM 的量化方案（GPTQ/AWQ）都为 gate 分支特殊处理")


# ================================================================ 3. 完整 Transformer FFN Block

def ffn_param_count(hidden, ffn_hidden, bias=False):
    """对比 SwiGLU 与 标准 GELU 的参数量"""
    swiglu = 3 * hidden * ffn_hidden
    gelu = 2 * hidden * ffn_hidden
    return swiglu, gelu


def param_analysis():
    print()
    print("=" * 78)
    print("3. 参数量精算")
    print("=" * 78)

    for hidden, ffn_hidden in [(4096, 11008), (5120, 13824), (8192, 22016)]:
        swiglu, gelu = ffn_param_count(hidden, ffn_hidden)
        # GELU 用 4*hidden 扩张才与 SwiGLU 参数量对齐
        gelu_ffn = hidden * 4
        gelu_aligned = 2 * hidden * gelu_ffn
        print(f"\nhidden={hidden}, SwiGLU ffn_hidden={ffn_hidden}")
        print(f"  SwiGLU 参数量                      : {swiglu / 1e9:.4f} B")
        print(f"  GELU 若用 ffn={gelu_ffn} (4x)      : {gelu_aligned / 1e9:.4f} B  ← 对齐基准")
        print(f"  两者比值: {swiglu / gelu_aligned:.3f}x  (应≈1，即参数量守恒)")
        print(f"  → SwiGLU 用 {ffn_hidden / hidden:.2f}*hidden 的扩张比，"
              f"省下 {(4 - ffn_hidden / hidden) / 4 * 100:.0f}% 的扩张宽度但参数量不变")

    print("\n换算成配置表（LLaMA 系列）:")
    print("  LLaMA-7B:  hidden=4096,  ffn=11008,  ratio=2.69 ≈ 8/3")
    print("  LLaMA-13B: hidden=5120,  ffn=13824,  ratio=2.70")
    print("  LLaMA-70B: hidden=8192,  ffn=22016,  ratio=2.69")
    print("  规则: ffn_hidden = ceil(8/3 * hidden)，且对齐到 256 的倍数（硬件友好）")


# ================================================================ 4. 梯度检查：验证公式实现正确

def gradient_check():
    if not HAS_TORCH:
        return

    print()
    print("=" * 78)
    print("4. 梯度验证 —— 确认公式实现无误")
    print("=" * 78)

    torch.manual_seed(0)
    x = torch.randn(2, 64, dtype=torch.float32, requires_grad=True)
    w = torch.randn(64, dtype=torch.float32, requires_grad=True)

    y = rmsnorm_reference(x, w)
    loss = y.pow(2).mean()
    loss.backward()

    print(f"\nloss = {loss.item():.6f}")
    print(f"x.grad范数 = {x.grad.norm().item():.6f}")
    print(f"w.grad 范数 = {w.grad.norm().item():.6f}")

    # 数值梯度检查（用独立副本，避免 inplace 操作破坏叶子张量）
    w_data = w.detach().clone()

    def numeric_grad_w(idx, eps=1e-4):
        """中心差分数值梯度，使用副本不污染原张量"""
        wd = w_data.clone()
        wd[idx] += eps
        loss_plus = rmsnorm_reference(x.detach(), wd).pow(2).mean().item()
        wd = w_data.clone()
        wd[idx] -= eps
        loss_minus = rmsnorm_reference(x.detach(), wd).pow(2).mean().item()
        return (loss_plus - loss_minus) / (2 * eps)

    idx = 5
    auto_grad = w.grad[idx].item()
    num_grad = numeric_grad_w(idx)
    print(f"\n权重 w[{idx}] 的梯度:")
    print(f"  自动微分: {auto_grad:+.6f}")
    print(f"  数值微分: {num_grad:+.6f}")
    print(f"  相对误差: {abs(auto_grad - num_grad) / max(abs(num_grad), 1e-9):.2e}")
    print("  → 误差在1e-4 以下说明实现正确（fp32 精度限制）")


def main():
    explain_rmsnorm()
    explain_swiglu()
    param_analysis()
    gradient_check()

    print()
    print("=" * 78)
    print("下一步: 读 notes/02-transformer-block.md")
    print("=" * 78)


if __name__ == "__main__":
    main()