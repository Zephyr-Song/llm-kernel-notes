"""
00_dtype_bench.py — dtype 精度、带宽与量化的真实代价

运行: python 00_dtype_bench.py

这个脚本想让你亲眼看到三件事：
1. 降低 dtype 不只是省显存，算力（FLOPs）不变，但**带宽需求直接减半**。
2. fp32 -> fp16 的舍入误差在数值范围极端处会被放大（catastrophic cancellation）。
3. 量化省的不是精度而是带宽，这才是推理加速的本质来源。
"""

import time

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------- 1. dtype 基本属性表

DTYPE_INFO = {
    "float32": {"bytes": 4, "exp_bits": 8, "mantissa_bits": 23, "dynamic_range": "~3.4e38"},
    "float16": {"bytes": 2, "exp_bits": 5, "mantissa_bits": 10, "dynamic_range": "~65504"},
    "bfloat16": {"bytes": 2, "exp_bits": 8, "mantissa_bits": 7, "dynamic_range": "~3.4e38"},
    "int8": {"bytes": 1, "exp_bits": 0, "mantissa_bits": 7, "dynamic_range": "[-128, 127]"},
}


def print_dtype_table():
    print("=" * 78)
    print("dtype 属性对比")
    print("=" * 78)
    print(f"{'dtype':<10} {'bytes':>6} {'exp':>4} {'mantissa':>9}  动态范围")
    print("-" * 78)
    for name, info in DTYPE_INFO.items():
        print(
            f"{name:<10} {info['bytes']:>6} {info['exp_bits']:>4} "
            f"{info['mantissa_bits']:>9}  {info['dynamic_range']}"
        )
    print()

    print("关键结论：")
    print("  · fp16 vs bf16 都是 2 字节，差别在于 fp16 指数位少(5 vs 8)但尾数多(10 vs 7)")
    print("    → fp16 精度高但范围窄(>65504 直接 inf)，所以 fp16 适合训练、bf16 适合大模型推理")
    print("  · int8 只有 1 字节，需要额外的 scale/zero_point 才能还原数值")
    print()


# ---------------------------------------------------------------- 2. 带宽需求计算

def memory_traffic(n_params: int, dtype: str, batch: int = 1) -> float:
    """
    Decode 阶段是 memory-bound：每生成一个 token，
    所有模型权重都要从 HBM 读一遍（每个 token 读全部权重）。

    这是为什么「参数多但激活少」的模型在 decode 时速度 ≈ 显存带宽 / 模型大小。
    """
    return n_params * DTYPE_INFO[dtype]["bytes"] * batch / 1e9  # GB


def print_bandwidth_analysis():
    print("=" * 78)
    print("Decode 阶段显存带宽需求（memory-bound 分析）")
    print("=" * 78)

    model_sizes = [("7B", 7e9), ("13B", 13e9), ("70B", 70e9)]

    print(f"{'模型':<8}", end="")
    for dtype in ["float32", "float16", "int8"]:
        print(f"{dtype:>14}", end="")
    print()
    print("-" * 78)

    for name, n in model_sizes:
        print(f"{name:<8}", end="")
        for dtype in ["float32", "float16", "int8"]:
            gb = memory_traffic(n, dtype)
            print(f"{gb:>11.2f} GB", end="")
        print()

    print()
    print("解读：")
    print("  · fp32 -> fp16 带宽需求直接砍半，这是最安全的加速手段")
    print("  · fp16 -> int8 再砍一半（4-bit 就是 1/4），但需要 group-wise scale 补偿精度损失")
    print("  · 关键：decode 阶段算力用不满瓶颈在带宽。所以优化推理 = 优化带宽 = 降精度 + 压缩")
    print("  · 对比 prefill 阶段：那里是 compute-bound，算力用得满，量化收益没那么大")
    print()


# ---------------------------------------------------------------- 3. 数值精度实测

def numerical_precision_demo():
    """演示不同 dtype 在数值范围极端处的行为差异。"""
    print("=" * 78)
    print("数值精度实测：大数 + 小数 的舍入行为")
    print("=" * 78)

    # 经典例子：1.0 + 1e-8 在不同 dtype 下是否还看得出来
    print("\n[测试1] 大数吃掉小数（catastrophic cancellation）")
    print("-" * 78)
    if HAS_TORCH:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            try:
                big = torch.tensor(1e5, dtype=dtype)
                small = torch.tensor(1.0, dtype=dtype)
                result = big + small
                print(
                    f"{str(dtype):<18} 1e5 + 1.0 = {result.item():<14.6f} "
                    f"改变了吗: {'是' if result.item() != 1e5 else '否（精度丢失）'}"
                )
            except RuntimeError as e:
                print(f"{str(dtype):<18} 溢出错误: {str(e)[:40]}")
    else:
        print("(需要 torch，跳过)")

    print("\n[测试2] bf16 的尾数只有 7 位，多位有效数字会丢")
    print("-" * 78)
    if HAS_TORCH:
        vals = [3.14159265, 2.71828182, 1.41421356]
        for v in vals:
            f32 = torch.tensor(v, dtype=torch.float32)
            bf16 = v and torch.tensor(v, dtype=torch.bfloat16)
            rel_err = abs(f32.item() - bf16.item()) / abs(f32.item())
            print(
                f"  {v:<14} fp32={f32.item():<12.8f} bf16={bf16.item():<12.8f} "
                f"相对误差={rel_err * 100:.3f}%"
            )
        print("\n  → bf16 相对误差在 0.1% 量级，训练时用 GradScaler/更高 LR 来补偿")
    else:
        print("(需要 torch，跳过)")

    print("\n[测试3] NumPy 模拟 fp16 上限溢出")
    print("-" * 78)
    fp16_max = float(np.finfo(np.float16).max)
    fp32_max = float(np.finfo(np.float32).max)
    print(f"  fp16 最大值: {fp16_max:.1f}")
    print(f"  fp32 最大值: {fp32_max:.3e}")
    print(f"  大约相差 {fp32_max / fp16_max:.3e}倍")
    print("  → 推理时激活值偶尔超65504 就会 inf，这是不用 fp16 而用 bf16 的主因")


# ---------------------------------------------------------------- 4. int8 量化实测

def quantization_demo():
    """per-tensor vs per-channel 量化的精度差异。"""
    print()
    print("=" * 78)
    print("int8 量化：per-tensor vs per-channel")
    print("=" * 78)

    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    torch.manual_seed(0)
    # 模拟真实的权重分布：不同通道的幅度差异很大
    weights = torch.randn(256, 512)
    weights[0, :] *= 10.0  # 第 0 个通道幅度大10 倍
    weights[1, :] *= 0.01  # 第 1 个通道幅度很小

    def quantize_per_tensor(w):
        scale = w.abs().max() / 127.0
        q = torch.clamp(torch.round(w / scale), -128, 127)
        return q * scale

    def quantize_per_channel(w):
        """按最后一维（output channel）分别算 scale，这是现代量化的标准做法"""
        scale = w.abs().amax(dim=1, keepdim=True) / 127.0
        q = torch.clamp(torch.round(w / scale), -128, 127)
        return q * scale

    qt = quantize_per_tensor(weights)
    qc = quantize_per_channel(weights)

    err_t = (qt - weights).abs().mean().item()
    err_c = (qc - weights).abs().mean().item()

    print(f"\n原始权重: shape={tuple(weights.shape)}, mean={weights.mean():.4f}, std={weights.std():.4f}")
    print(f"per-tensor  量化误差 MAE = {err_t:.6f}")
    print(f"per-channel 量化误差 MAE = {err_c:.6f}")
    print(f"per-channel 误差降低 {(1 - err_c / err_t) * 100:.1f}%")

    print("\n逐通道对比（第 0 和第 1 通道是幅度异常的）：")
    for ch in [0, 1, 2]:
        orig = weights[ch]
        err_t_ch = (quantize_per_tensor(orig.unsqueeze(0))[0] - orig).abs().mean().item()
        err_c_ch = (quantize_per_channel(orig.unsqueeze(0))[0] - orig).abs().mean().item()
        print(
            f"  channel {ch}: |w|均值={orig.abs().mean():.4f}  "
            f"per-tensor误差={err_t_ch:.6f}  per-channel误差={err_c_ch:.6f}"
        )

    print("\n结论：")
    print("  · per-tensor 用一个 scale 覆盖全部权重，遇到幅度差异大的通道会精度爆炸")
    print("  · per-channel / per-group 是必要代价：每个通道一个 scale，误差大幅下降")
    print("  · 现代方案（GPTQ/AWQ）继续往 group_size=128 甚至更小走，权衡是额外存储 scale")
    print("  · AWQ 的思路更进一步：不均匀量化，按重要性分配 bit 数")


# ---------------------------------------------------------------- 5. PyTorch dtype 换算耗时

def dtype_cast_benchmark():
    print()
    print("=" * 78)
    print("dtype 转换耗时对比")
    print("=" * 78)

    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    n = 4096 * 4096  # ~16M 元素，约 64MB fp32
    src = torch.randn(4096, 4096)

    for target in [torch.float16, torch.bfloat16]:
        # warmup
        _ = src.to(target)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            _ = src.to(target)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        print(
            f"  fp32 -> {str(target):<16} 10次转换耗时: {(t1 - t0) * 1000:>8.2f} ms "
            f"(单次 {(t1 - t0) * 100:.2f} ms)"
        )

    print()
    print("说明：上面是 CPU 上的结果，GPU 上会快但仍是额外的 kernel launch 开销。")
    print("推理引擎的做法是：量化在模型加载时一次做完，推理时不再转换。")


def main():
    print_dtype_table()
    print_bandwidth_analysis()
    numerical_precision_demo()
    quantization_demo()
    dtype_cast_benchmark()

    print()
    print("=" * 78)
    print("下一步: 读 notes/01-tensor-and-dtype.md，把上面的实验结论固化下来")
    print("=" * 78)


if __name__ == "__main__":
    main()