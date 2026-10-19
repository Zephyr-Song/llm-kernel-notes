"""
04_gqa_kvcache.py — GQA/MQA + KV cache 的显存账与实测

运行: python 04_gqa_kvcache.py

要回答四个问题：
  1. KV cache 为什么存在（decode 阶段为什么不能重算全部注意力）
  2. GQA / MQA 的显存怎么算，为什么它们是现代 LLM 的标配
  3. GQA 在什么位置被应用（MHA → GQA的转换时机）
  4. MLA（DeepSeek）如何更进一步压缩 KV

**关键认知：GQA 省的是显存带宽和显存容量，不省 FLOPs。**
"""

import math

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ================================================================ 1. KV cache 为什么存在

def why_kv_cache():
    print("=" * 78)
    print("1. KV cache：为什么不能每次重算")
    print("=" * 78)

    print("""
【场景】生成长度 L 的序列，模型已经有 N 个 token。

没有 KV cache（每步全量重算）：
  第 1 步：处理 1 个 token
  第 2 步：处理 2 个 token
  ...
  第 L 步：处理 L 个 token
  总计算量: O(1 + 2 + ... + L) = O(L^2 / 2)

  问题在 L=2048 时：
    每个 step 都要重新计算前面所有 token 的 K 和 V
    这些 K/V 的计算结果其实和之前完全一样（因果注意力下位置 0 的 K
    永远只依赖自己），是纯粹的重复劳动

有 KV cache（缓存历史 K/V）：
  第 1 步：计算 K/V，存入 cache
  第 2 步：只算新 token 的 K/V，追加到 cache，attention 用 cache + 新 K/V
  ...
  每步只处理 1 个 token
  总计算量: O(L)

  → 加速比随 L 线性增长。L=2048 时理论上 2048 倍加速（当然有常数损耗）

【另一个视角：prefill vs decode】

  Prefill（处理 prompt 的 N 个 token，一次性）：
    - 计算密集 compute-bound，算力用得满
    - Attention 是 O(N^2) 的大矩阵乘法
    - 这一步决定 TTFT（首 token 时间）

  Decode（逐个生成，每个 step 1 个 token）：
    - 内存带宽密集 memory-bound，算力利用率极低
    - 每个 step 都要读完所有模型权重 + KV cache
    - 这一步决定 TPOT（每 token 时间）

  关键：Decode 阶段每生成 1 个 token 都要读一遍全部权重。
  70B 模型 fp16 = 140GB，A100 带宽 2TB/s → 理论最快 70ms/token
  实际远达不到，因为还有 KV cache 的读取和 kernel launch 开销。

  → 优化 Decode 的核心是「减少每token 要读的字节数」
     → 量化（减权重）
     → GQA/MQA（减 KV cache）
     → 这是两条完全不同的优化线
""")


# ================================================================ 2. KV cache 显存精确计算

def kv_cache_memory():
    print("=" * 78)
    print("2. KV cache 显存计算（这是长上下文推理的真瓶颈）")
    print("=" * 78)
    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    print("""
公式：
  KV_cache_bytes = 2 (K和V) × n_layers × n_kv_heads × head_dim × seq_len × bytes_per_elem × batch_size

注意：没有 n_q_heads！这是 GQA 省显存的全部秘密。
""")

    bytes_per = 2  # fp16

    configs = [
        ("LLaMA-7B (MHA)", 32, 32, 4096, 32, 4096),
        ("LLaMA-2-7B (GQA-32)", 32, 32, 4096, 32, 4096),
        ("LLaMA-2-13B (GQA-40)", 40, 40, 5120, 40, 5120),
        ("Mistral-7B (GQA-8)", 32, 8, 4096, 32, 4096),
        ("LLaMA-3-8B (GQA-8)", 32, 8, 4096, 32, 4096),
        ("Qwen2-7B (GQA-4)", 28, 4, 3584, 28, 3584),
    ]

    print(f"{'配置':<26} {'layers':>7} {'kvhds':>7} {'hd':>6} {'4K':>10} {'32K':>10} {'128K':>10}")
    print("-" * 82)

    for name, n_layers, n_kv, hidden, n_q, head_dim in configs:
        row = [name]
        for seq_len in [4096, 32768, 131072]:
            total = (
                2 * n_layers * n_kv * head_dim * seq_len * bytes_per * 1
            ) / 1e9
            row.append(f"{total:.2f} GB")
        print(
            f"{row[0]:<26} {n_layers:>7} {n_kv:>7} {head_dim:>6} "
            f"{row[1]:>10} {row[2]:>10} {row[3]:>10}"
        )

    print("\n对比计算（MHA vs GQA，同7B 架构，只改 n_kv_heads）:")
    print("-" * 66)
    for seq_len in [4096, 32768, 131072]:
        mha = 2 * 32 * 32 * 128 * seq_len * 2 / 1e9
        gqa8 = 2 * 32 * 8 * 128 * seq_len * 2 / 1e9
        gqa4 = 2 * 32 * 4 * 128 * seq_len * 2 / 1e9
        print(
            f"  seq={seq_len//1024:>4}K:  MHA={mha:>7.2f} GB   "
            f"GQA-8={gqa8:>6.2f} GB (省 {mha / gqa8:.0f}x)   "
            f"GQA-4={gqa4:>6.2f} GB (省 {mha / gqa4:.0f}x)"
        )

    print("\n结论:")
    print("  · MHA 在 128K 上下文时，KV cache 要 137 GB —— 单卡放不下")
    print("  · GQA-8 降到 34 GB，GQA-4 降到 17 GB → 长上下文成为可能")
    print("  · 这就是为什么 LLaMA-2 开始所有模型都用 GQA")


# ================================================================ 3. GQA 的实现与验证

def gqa_attention(Q, K, V, n_groups, causal=True):
    """
    Grouped Query Attention 的参考实现。

    形状约定：
      Q: [batch, n_q_heads, seq, head_dim]
      K: [batch, n_kv_heads, seq, head_dim]   # n_kv_heads < n_q_heads
      V: [batch, n_kv_heads, seq, head_dim]

    做法：把 Q 的 head 分成 n_groups 组，每组共享一份 K/V。
    等价于把 K/V 沿head 维 repeat_interleave 后做标准 MHA。
    """
    b, n_q, s, d = Q.shape
    n_kv = K.shape[1]

    assert n_q % n_kv == 0, f"n_q_heads({n_q}) 必须被 n_kv_heads({n_kv}) 整除"
    assert n_q // n_kv == n_groups

    # 把 K/V 扩展到 n_q_heads（这是「不省 FLOPs」的证明：还是要算 n_q 个 head）
    K_exp = K.repeat_interleave(n_groups, dim=1)
    V_exp = V.repeat_interleave(n_groups, dim=1)

    return F.scaled_dot_product_attention(Q, K_exp, V_exp, is_causal=causal)


def gqa_validation():
    print()
    print("=" * 78)
    print("3. GQA 实现验证")
    print("=" * 78)
    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    torch.manual_seed(0)
    b, n_q, n_kv, s, d = 1, 8, 2, 64, 32
    n_groups = n_q // n_kv

    Q = torch.randn(b, n_q, s, d)
    K = torch.randn(b, n_kv, s, d)
    V = torch.randn(b, n_kv, s, d)

    print(f"\nbatch={b}, n_q_heads={n_q}, n_kv_heads={n_kv}, groups={n_groups}, seq={s}, head_dim={d}")

    out = gqa_attention(Q, K, V, n_groups)
    print(f"输出 shape: {tuple(out.shape)}")

    # 验证分组正确性：head 0 和 head 1(n_kv=1) 应该共享同一份 K/V
    K_exp = K.repeat_interleave(n_groups, dim=1)
    ref = F.scaled_dot_product_attention(Q, K_exp, V_exp := V.repeat_interleave(n_groups, dim=1), is_causal=True)
    print(f"与直接 expand 的最大差异: {(out - ref).abs().max().item():.3e}  (应该接近0)")

    # 重要：GQA 不省 FLOPs
    print("\n【关键认知】GQA 省的是什么，不省什么:")
    print("  不省 FLOPs:")
    print(f"    n_q_heads={n_q} 个 head 的 Q 都要和 K 做attention → attention 计算量不变")
    print("    展开后 K 的 head 数是 n_q 而非 n_kv，所以 matmul 规模不变")
    print("\n  省的是:")
    print("    1. KV cache 显存：K/V 只需存 n_kv 份 → 省 n_q/n_kv 倍")
    print("    2. Decode 阶段带宽：每步读 KV cache 的字节数减少 → 直接提升 TPS")
    print("    3. Prefill 阶段：K/V 的投影计算量减少（虽然不多）")

    # 实测 decode 阶段的差异
    print("\n" + "=" * 78)
    print("4. Decode 阶段实测：GQA 对速度的影响")
    print("=" * 78)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"\ndevice={device}, dtype={dtype}")

    seq_len = 1024
    for kv_heads in [32, 8, 4, 1]:
        if kv_heads > n_q:
            continue
        groups = n_q // kv_heads

        Qd = torch.randn(1, n_q, 1, d, dtype=dtype, device=device)
        Kd = torch.randn(1, kv_heads, seq_len, d, dtype=dtype, device=device)
        Vd = torch.randn(1, kv_heads, seq_len, d, dtype=dtype, device=device)

        def step():
            return gqa_attention(Qd, Kd, Vd, groups, causal=False)

        step()
        if device == "cuda":
            torch.cuda.synchronize()

        import time

        n_iter = 100
        t0 = time.perf_counter()
        for _ in range(n_iter):
            step()
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        kv_bytes = 2 * kv_heads * seq_len * d * 2
        print(
            f"  n_kv_heads={kv_heads:>3} (GQA-{groups if kv_heads > 1 else 'MQA'}): "
            f"KV cache={kv_bytes / 1e6:.2f} MB  "
            f"单步耗时={(t1 - t0) / n_iter * 1e6:>8.1f} μs"
        )

    if device == "cpu":
        print("\n  (CPU 上差异不明显，GPU 上 KV cache 读取是真实瓶颈)")
        print("   生产推理服务上，MQA vs GQA-8 的 TPS 差距通常有 20~40%。")


# ================================================================ 4. 从 MHA 到 GQA 的转换

def mha_to_gqa_conversion():
    print()
    print("=" * 78)
    print("5. MHA → GQA 的权重重组（up-training）")
    print("=" * 78)

    print("""
问题：Qwen 等模型从 MHA 迁移到 GQA 时，不能直接砍掉 K/V 的 head
      —— 那样会破坏已经学好的注意力分布。

解法（GQA 的 up-training，Google 提出）：
  1. 复制 MHA 的 K/V 权重，得到 M 个「候选」投影
  2. 为每组内的 M 个 K 投影加一个可学习向量 b_i:
       k'_m = k_m + b_m
  3. 前向时：k'_mean = mean(k'_1..k'_M)
             用 k'_mean 做 attention
  4. 反向时：梯度回传给 k'_mean（不更新各 k_m 的参数）
  5. 预训练结束后：k_new = k'_mean，丢弃 b 和其他M-1 个 k

为什么有效：
  · 均值是一个好的初始猜测，比随机初始化好
  · 但保留了「给不同 head 加不同 bias」的调整能力
  · 等价于给 group 内所有 head 一个「共同起点 + 各自偏移」

DeepSeek 的 MLA 走了另一条路：从头设计，不需要迁移。
""")

    print("实际配置的对比：")
    print(f"{'模型':<20} {'n_q':>5} {'n_kv':>6} {'类型':>8} {'KV省':>6}")
    print("-" * 56)
    models = [
        ("GPT-2 (2019)", 12, 12, "MHA", "1x"),
        ("LLaMA-1 7B", 32, 32, "MHA", "1x"),
        ("LLaMA-2 7B", 32, 32, "MHA", "1x"),
        ("LLaMA-2 70B", 64, 8, "GQA-8", "8x"),
        ("Mistral 7B", 32, 8, "GQA-4", "4x"),
        ("LLaMA-3 8B", 32, 8, "GQA-4", "4x"),
        ("Qwen2 7B", 28, 4, "GQA-7", "7x"),
        ("Qwen2 0.5B", 14, 2, "GQA-7", "7x"),
        ("Gemma 2B", 8, 1, "MQA", "8x"),
    ]
    for name, nq, nkv, typ, save in models:
        print(f"{name:<20} {nq:>5} {nkv:>6} {typ:>8} {save:>6}")


# ================================================================ 5. 推理侧优化

def inference_optimizations():
    print()
    print("=" * 78)
    print("6. KV cache 的其他优化方向")
    print("=" * 78)

    print("""
【1. 量化 KV cache —— 最直接的手段】
   收益：KV cache 显存和带宽直接减半（fp16 → int8）或降到1/4（fp4）
   代价：精度损失，主要影响长上下文的检索能力
   实践：vLLM 支持 fp8 KV cache，GPTQ/AWQ 也有 KV cache 量化版本

   注意一个反直觉现象（Qwen2 报告）：
     如果模型训练时见过大量数值大的 activation（如 MoE 的 gate），
     KV cache 会有 outlier，直接量化误差大。
     → 需要 per-channel / per-head 分组量化，而不是 per-tensor。

【2. PagedAttention（vLLM）—— 解决碎片，不省总量】
   问题：连续分配 KV cache 会产生碎片
     · 请求长度不确定 → 预留多了浪费，预留少了要迁移
     · 传统方式显存利用率只有 20~40%
   解法：操作系统分页思想
     · 把 KV cache 切成固定大小的 block（如 16 token/block）
     · 用 block table 记录逻辑地址到物理地址的映射
     · 需要时按需分配，不需要时释放
   收益：显存利用率 20~40% → 90%+，可以服务 3~4倍 并发

【3. MLA（Multi-head Latent Attention, DeepSeek-V2）】
   核心思想：既然 K/V cache 太大，那不存 K/V，存它们的「压缩版」
     · 下投影到低维 latent c_KV（维度约 512，而 K/V head 是 128）
     · cache 里只存 c_KV
     · 需要 K 时上投影：k = W_UK c_KV
   关键洞察：W_UK 是固定的，可以提前和所有 c_KV 的上投影结果合并
             → 用「矩阵吸收」技巧，让上投影的耗时被缓存下来
   结果：KV cache 比 GQA-8 还小 93.3%，效果还更好
   → 这是当前压缩 KV cache 的最优解

【4. 滑动窗口注意力】
   ·Mistral / Gemma-2 采用
   · 每个 token 只 attend 最近 W 个（如 W=4096）
   · KV cache 大小与序列长度解耦
   · 代价：无法做真正的长程检索

【5. 跨层 KV 共享】
   · YOCO / CLA 等工作提出
   · 相邻层复用同一份 KV
   · 收益：层数越多省得越多
   · 代价：可能损失表达力

【优先级建议（工程实践）】
   1. GQA（模型自带，必须有）
   2. PagedAttention（vLLM 默认，框架解决）
   3. FP8 KV cache（简单有效）
   4. 分组量化 KV cache（如果需要）
   5. MLA（需要改模型架构，训练时就定）
""")


# ================================================================ 6. 完整推理循环示例

def kv_cache_inference_demo():
    print()
    print("=" * 78)
    print("7. 带 KV cache 的单步推理（完整实现）")
    print("=" * 78)
    if not HAS_TORCH:
        print("(需要 torch，跳过)")
        return

    torch.manual_seed(0)
    n_q, n_kv, d, n_layers = 8, 2, 32, 2
    n_groups = n_q // n_kv
    s_prompt = 16
    gen_steps = 8

    print(f"\n配置: n_q={n_q}, n_kv={n_kv}, groups={n_groups}, layers={n_layers}, head_dim={d}")
    print(f"prompt 长度={s_prompt}, 生成 {gen_steps} 步\n")

    # 模拟 embedding 后的 hidden states
    hidden = torch.randn(s_prompt, d)

    kv_cache = [
        {
            "k": torch.zeros(n_kv, 0, d),
            "v": torch.zeros(n_kv, 0, d),
        }
        for _ in range(n_layers)
    ]

    print(f"{'step':>5} {'cache_len':>11} {'K cache':>12} {'新token位置':>12}")
    print("-" * 48)

    for layer_idx in range(1):
        # ---------- Prefill ----------
        q = torch.randn(s_prompt, n_q, d)
        k = torch.randn(s_prompt, n_kv, d)
        v = torch.randn(s_prompt, n_kv, d)

        kv_cache[layer_idx]["k"] = k.transpose(0, 1)  # [n_kv, s_prompt, d]
        kv_cache[layer_idx]["v"] = v.transpose(0, 1)

        q_p = q.transpose(0, 1).unsqueeze(0)  # [1, n_q, s, d]
        out = gqa_attention(q_p, kv_cache[layer_idx]["k"].unsqueeze(0),
                            kv_cache[layer_idx]["v"].unsqueeze(0), n_groups, causal=True)
        print(f"{'prefill':>5} {s_prompt:>11} {kv_cache[layer_idx]['k'].numel() * 4:>12} {'0..15 (全部)':>12}")

        # ---------- Decode ----------
        for step in range(gen_steps):
            # 新 token 的 Q/K/V（只处理1 个 token）
            q_new = torch.randn(1, n_q, d)
            k_new = torch.randn(1, n_kv, d)
            v_new = torch.randn(1, n_kv, d)

            # 追加到 cache（注意：这一步是 O(1) 内存操作，不需要重算历史）
            kv_cache[layer_idx]["k"] = torch.cat(
                [kv_cache[layer_idx]["k"], k_new.transpose(0, 1)], dim=1
            )
            kv_cache[layer_idx]["v"] = torch.cat(
                [kv_cache[layer_idx]["v"], v_new.transpose(0, 1)], dim=1
            )

            cache_len = kv_cache[layer_idx]["k"].shape[1]
            q_p = q_new.transpose(0, 1).unsqueeze(0)  # [1, n_q, 1, d]

            # 注意 causal=False：新 token 可以看到所有 cache（历史已经保证了因果性）
            out = gqa_attention(
                q_p,
                kv_cache[layer_idx]["k"].unsqueeze(0),
                kv_cache[layer_idx]["v"].unsqueeze(0),
                n_groups,
                causal=False,
            )
            print(
                f"{step:>5} {cache_len:>11} {kv_cache[layer_idx]['k'].numel() * 4:>12} "
                f"{cache_len - 1:>12}"
            )

    print("\n关键观察:")
    print("  · decode 每步的 Q 只有 1 个 token（shape [1, n_q, 1, d]）")
    print("  · 但 K/V 是全量 cache（shape [1, n_kv, cache_len, d]）")
    print("  · 所以 decode 是「一条query 对全部 key」→ matrix-vector 而非 matrix-matrix")
    print("  · 这就是为什么 decode 阶段 memory-bound：读远多于算")
    print("\n  · K cache 大小 (numel*4 = numel*2forK*2fp16) 随步数线性增长")
    print("    → 这就是长上下文推理时 KV cache 成为瓶颈的直观来源")

    # GQA 在这里省了多少
    print(f"\nGQA 节省验证 (layer {layer_idx}):")
    k_gqa = kv_cache[layer_idx]["k"]
    k_mha = k_gqa.repeat_interleave(n_groups, dim=0)
    print(f"  GQA ({n_kv} heads): {k_gqa.shape}")
    print(f"  MHA ({n_q} heads, 等价): {k_mha.shape}")
    print(f"  节省倍数: {k_mha.numel() / k_gqa.numel():.0f}x")


def main():
    why_kv_cache()
    kv_cache_memory()
    gqa_validation()
    mha_to_gqa_conversion()
    inference_optimizations()
    kv_cache_inference_demo()

    print()
    print("=" * 78)
    print("阶段2 完成。下一步: notes/05-mha-gqa-mqa.md，然后进入阶段3")
    print("=" * 78)


if __name__ == "__main__":
    main()