# 03 · Attention：从 O(N²) 到 FlashAttention

> 对应代码：`code/02_naive_attention.py`（**这个仓库最重要的文件**）

## 朴素实现与它的死因

```python
scores = Q @ K.transpose(-2, -1) / sqrt(d)    # [N, N]  ← 元凶
scores = scores + causal_mask
attn = torch.softmax(scores, dim=-1)
return attn @ V
```

### 显存账（fp16, head_dim=128）

| seq_len | N² matrix / head | Q+K+V / head | 放大倍数 |
|---|---|---|---|
| 1,024 | 2.1 MB | 0.79 MB | 2.7x |
| 4,096 | 33.6 MB | 3.1 MB | 11x |
| 16,384 | 537 MB | 12.6 MB | 43x |
| 65,536 | 8.6 GB | 50.3 MB | 170x |
| 262,144 | 137 GB | 201 MB | 682x |

**结论**：seq_len 上万后，`N²` 的 attention matrix 就成了显存黑洞。这就是长上下文难做的**根本原因**——不是算力不够，是**装不下**。

---

## FlashAttention：算法核心

### 在线 Softmax（唯一需要理解的数学）

标准 softmax 要两遍：
```
1. 遍历求 max
2. 遍历算 exp、求和、归一化
```

FlashAttention 不能保存全部 scores（那就退化了），所以要**在线增量维护** max 和 sum：

```
来了一块新的 scores_block：
  m_new = max(m_old, block_max)              # 新的全局最大
  l_new = l_old * exp(m_old - m_new)          # 旧数据重新缩放
         + sum(exp(block - m_new))            # 新的累加
  acc   = acc * exp(m_old - m_new) + P @ V    # 输出累加器同样重标
```

**全部复杂度只来自这一个缩放因子 `exp(m_old - m_new)`。** 这是数值稳定技巧（保证 exp 的参数 ≤ 0，不会溢出）。

> **易错点**：当 `m_old = -inf`（第一块）时，`exp(-inf - m_new) = 0`，正确。但若整块 scores 都是 `-inf`（causal mask 下可能发生），`m_new = -inf` → `-inf - (-inf) = NaN`。所以必须 clamp。
>
> 我在代码里用 `torch.clamp(m_new, min=-1e30)` 解决。踩过这个坑。

### 为什么快：IO 而非 FLOPs

**FlashAttention 的 FLOPs 反而更多**（每块要重算 exp/sum，反向要重算中间值）：

```
朴素:  QK^T (2N²d ops) + attn@V (2N²d ops)  = 4N²d
Flash: 同上 + 块内 softmax 约 5~10 ops/element + 反向重算
     → 总计多 10~20% FLOPs
```

**加速 100% 来自减少 HBM↔SRAM 的数据搬运。**

用roofline 模型算一下（A100: FP16 312 TFLOPS, HBM 2TB/s, seq=4096, d=128）：

```
理想计算时间 = 4.3 GFLOP / 312 TFLOPS ≈ 14 μs
朴素 IO 时间  = 33.6 MB × 2 (写+读) / 2 TB/s ≈ 34 μs
→ IO-bound 约 2.4x（seq 越大越极端）
```

**朴素实现 99% 的时间在等内存，算力完全闲置。**

FlashAttention 的三个 IO 优化：
1. **SRAM 复用** — K/V 的 block 载入 shared memory，被所有 Q block 复用，避免反复读 HBM
2. **不物化 N²** — 每块的中间结果算完即弃，显存从 O(N²) 降到 O(N)
3. **反向重计算** — backward 时重算中间值而非保存，换显存（compute 换 memory）

---

## 必须记住的结论

> **FlashAttention 更快，是因为它更省 IO。它并没有变快（甚至更慢），它变的是「等内存」的时间。**
>
> 所以 FlashAttention 的收益和 seq_len 正相关。短上下文（512 以内）几乎没收益；长上下文收益巨大。
>
> 更重要的：**它把 O(N²) 显存降到 O(N)**，这才让长上下文成为可能。

---

## 手写验证时的易错点

|坑 | 症状 | 排查 |
|---|---|---|
| causal mask 方向反 | 未来token 能看到过去 | `mask[k_pos > q_pos]` 不能写成 `>=` |
| m_new 是 -inf | 输出 NaN | `clamp(m_new, min=-1e30)` |
| acc 重缩放顺序错 | 数值不对但不NaN | 缩放必须在 `matmul(p, v)` **之前** |
| block_size 不整除 seq_len | 最后一块形状错误 | 循环用 `min(start+size, seq_len)` |

对拍基准：`attention_naive` 的输出。只要 `max_diff < 1e-5`（fp32）就说明实现正确。

---

## PyTorch SDPA 与后端选择

`F.scaled_dot_product_attention` 是 PyTorch 的统一入口，会自动选后端：

```
FLASH_ATTENTION      → FlashAttention-2 CUDA kernel（最优）
EFFICIENT_ATTENTION  → memory-efficient 实现（xformers 风格，支持更多 shape）
MATH                 → 朴素实现（唯一能处理任意 mask）
CUDNN_ATTENTION      → cuDNN 后端
```

**强制指定后端**（用于调试，确认到底走了哪条路径）：
```python
from torch.nn.attention import SDPBackend, sdpa_kernel
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**选后端的坑**：
- `MATH` 是唯一支持任意 `attn_mask` 的。如果你传了一个自定义 float mask，可能**静默降级到 MATH**，性能暴跌几十倍且无警告。
- 想用 Flash 后端，必须满足：`is_causal=True` 或 mask 可转成 causal、head_dim 是 2 的幂且 ≤ 256、dtype 是 fp16/bf16。
- **排查性能问题时，第一件事是打印实际用的后端**，而不是怀疑模型。可以用 `torch.profiler` 看 kernel 名。

---

## 因果注意力的稀疏性（引出后续优化）

跑 `causal_attention_visualization()` 会看到：

```
第 0 行有效位置数: 1      (只能看自己)
第 100 行:         101
第 511 行:         512
```

这是定义决定的，不算发现。**真正的发现是：即使在 causal mask 内，注意力的数值分布也是高度稀疏的**——少数几个 token 拿走了绝大部分权重。

这直接支撑了三类后续优化：
1. **KV cache** — 每个新 token 都要重算对全部历史的注意力，浪费严重
2. **稀疏注意力**（Longformer/BigBird）— 只 attend 少数重要位置
3. **StreamingLLM / 滑窗** — 假设注意力集中在局部

---

## 验收清单

- [ ] 手写 FlashAttention，输出与朴素实现误差 < 1e-5
- [ ] 能解释「在线 softmax」中 `exp(m_old - m_new)` 的作用
- [ ] 能证明FlashAttention 的 FLOPs **更多**而不是更少
- [ ] 算出 seq_len=128K 时朴素 attention matrix 有多大
- [ ] 知道如何强制指定 SDPA 后端，以及何时会静默降级到 MATH
- [ ] 能解释为什么 FlashAttention 对短上下文收益很小

## 参考

- Attention Is All You Need — Vaswani et al. 2017
- FlashAttention — Dao et al. 2022
- FlashAttention-2 — Dao 2023（更好的工作划分与并行策略）
- Online Softmax — Milakov & Gimelshein 2018（理论来源）