# 02 · Transformer Block：现代 LLM 的标准零件

> 对应代码：`code/01_rmsnorm_swiglu.py`

## 一个 Block 的完整形态

现代 decoder-only LLM 的一个 Transformer Block（LLaMA 风格）：

```
              ┌──────────────────────────┐
  x  ────────▶│  RMSNorm                  │
              └──────────────────────────┘
                        │
                        ▼
              ┌──────────────────────────┐
              │  RMSNorm → Q,K,V 投影    │
              │  RoPE(Q), RoPE(K)        │
              │  Attention               │
              └──────────────────────────┘
                        │
                        ▼  x = x + attention_out      (残差)
              ┌──────────────────────────┐
              │  RMSNorm                  │
              └──────────────────────────┘
                        │
                        ▼
              ┌──────────────────────────┐
              │  SwiGLU FFN               │
              │  down(silu(gate)*up)     │
              └──────────────────────────┘
                        │
                        ▼  x = x + ffn_out            (残差)
             ┌──────────────────────────┐
             ▼                          │
          （下一层）                     │
```

**和原始 Transformer（2017）的差异：**

| 组件 | 原始 | 现代 LLM |
|---|---|---|
| Norm 位置 | Post-LN（在子层之后） | **Pre-LN**（在子层之前） |
| Norm 类型 | LayerNorm | RMSNorm |
| 激活 | ReLU / GELU | SwiGLU |
| 位置编码 | 绝对（正弦） | RoPE（相对） |
| FFN 扩张比 | 4x | ~2.67x |

---

## Pre-LN vs Post-LN：为什么必须换

这是**训练稳定性**问题，不是效果问题。

### Post-LN（原始）
```
x_{l+1} = LayerNorm(x_l + F(x_l))
```
子层输出要**穿过 LayerNorm** 才能传给下一层。当层数很深时，反向传播的梯度要连续通过几十个 LayerNorm，路径上任何一处不稳定都会被放大。**深层 Post-LN 训练极易发散**，GPT-3 当年就得靠 careful 的 lr warmup 才能训起来。

### Pre-LN（现代）
```
x_{l+1} = x_l + F(Norm(x_l))
```
残差路径变成一条**干净的恒等通路**——梯度可以直接从最后一层"抄近路"回到第一层，中间没有任何非线性。这让深层训练变得几乎无条件稳定。

### Pre-LN 的代价

**这里有一个必须知道的 trade-off：**

Pre-LN 残差流中累积的是**各子层输出的未归一化之和**。深层的 `x_l` 的**数值范数会随深度线性增长**（因为残差是直接相加的）。

后果：**模型需要更强的 post-training 对齐**（如 RLHF）才能在推理时压住这种累积。

- 这正是为什么 LLaMA-1 表现不如 LLaMA-2：LLaMA-1 没做RLHF，只做了 supervised fine-tuning，压不住 Pre-LN 的激活范数增长 → 容易产生"胡说八道"且语气生硬。
- LLaMA-2 系统性做了 RLHF → 明显改善。

**记住这个因果链：Pre-LN → 深层训练更稳 → 但需要 post-training 压范数 → 所以现代模型都强化对齐流程。这不是巧合。**

（补充：为了兼顾两者，DeepSeek-V2 引入了 **DeepSeek-V2 RMSNorm**——在每个 Block 内也加 Norm，把 Pre-LN 和 Post-LN 的优点结合。Qwen3 等新模型也用了类似的 in-block norm。）

---

## RMSNorm

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * γ
```

vs LayerNorm：

```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * γ + β
```

**区别：去掉了减均值，去掉了 bias。**

### 为什么去掉还能更好

直觉：Transformer 里的残差流已经提供了"以某个中心为中心"的能力，**归一化层自己的减均值是冗余的**。去掉它 → 少一个 reduce 操作 + 少一列权重 → 更快，且实测困惑度略好。

### 必须澄清的常见误解

RMS 除的是 `sqrt(mean(x²))`，**不是真正的标准差**（标准差要先减均值）。

```
RMS    = sqrt( (1/d) * Σ x_i² )       ← 二次均方根，衡量「能量」
Std    = sqrt( (1/d) * Σ (x_i - x̄)² )  ← 需要先减均值
```

如果用 Std 来归一化，就退化成 LayerNorm 了。所以「RMSNorm 是简化的 LayerNorm」这句话在数学上不准确——它是一种**不同的归一化准则**，恰好在 LLM 上效果更好。

### eps 的位置

正确写法：`x * rsqrt(var + eps)`，而不是 `x * rsqrt(var) * eps`。后者在 var→0 时会先除零得到 `inf`，再乘 eps 还是 `inf`。

---

## SwiGLU

```
SwiGLU(x) = SiLU(gate(x)) * up(x)
SwiGLU的 FFN:  down(SiLU(gate(x)) * up(x))
```

### 参数量守恒（重要）

很多人以为 SwiGLU 是"加了一个矩阵所以更大"，其实它**参数量完全相同**：

```
标准 FFN (4x 扩张):
  W1: [4d, d],  W2: [d, 4d]     → 参数量 = 8 d²

SwiGLU FFN (8/3 扩张):
  gate: [h, d], up: [h, d], down: [d, h]  where h = ceil(8d/3)
  → 参数量 = 3 h d = 3 * (8d/3) * d = 8 d²     完全相同！
```

**LLaMA 配置表验证：**

| 模型 | hidden | ffn_hidden | ratio |
|---|---|---|---|
| LLaMA-7B | 4096 | 11008 | 2.69 ≈ 8/3 |
| LLaMA-13B | 5120 | 13824 | 2.70 |
| LLaMA-70B | 8192 | 22016 | 2.69 |

`ffn_hidden = ceil(8/3 × hidden)`，并对齐到 256 的倍数（对硬件对齐友好）。

### 门控在做什么

```
gate 分支  →  学「该不该激活」（一个 sigmoid 决策信号）
up 分支    →  学「激活什么内容」（承载具体信息）
相乘       →  用一个标量门控控制一路信息通道
```

直觉：模型需要对不同 token **选择性激活**某些知识通道。门控提供了这种"选择性"。

**LLaMA 论文的消融（相同参数量）：**
```
ReLU² < GELU < SwiGLU
```
8B 参数时 SwiGLU 比 ReLU² 高约 2% 困惑度优势。

### 和量化的关系（很多论文里最容易被忽略的一段）

LLaMA 论文 3.4 节明确指出：

> **SwiGLU 的 gate 分支容易产生「幅度小但数量多」的 outlier 激活值**，这导致激活量化误差偏大。

LLaMA 的解法是：
1. **不量化 gate 分支**（它恰是带宽瓶颈的主要来源，不量化会损失太多收益）
2. 其他分支用更高精度

→ 所以现代量化方案（GPTQ/AWQ/KIVI）都**为 gate 分支特殊处理**。你在读量化论文时看到"我们 special-case the gate projection"，指的就是这件事。

---

## 我踩的坑

### 坑 1：以为 RMSNorm = LayerNorm 简化版

**症状**：写 RMSNorm 时除以了标准差。
**根因**：混淆了 RMS 和 Std。
**排查**：打印归一化后的均值——RMSNorm 的输出均值**不为 0**，LayerNorm 一定为 0。这个测试能立刻区分。

### 坑 2：忘了 FFN 也有 norm

**症状**：实现 Block 时只在 attention 前加了 Norm。
**修正**：现代 LLM **每个子层前都有独立 Norm**（RMSNorm 权重不共享）。这是 ResNet 那种共享-norm 的做法和 Transformer 的区别。

### 坑 3：Pre-LN 模型的输出范数随深度增长

**症状**：深层模型的 hidden state norm 逐层上升，某些层的输出数值明显偏大，导致下游 (RMSNorm/量化) 出问题。
**根因**：这是 Pre-LN 的固有特性，不是 bug。
**应对**：
- 检查 norm_r 和 hidden 在每层的实际范数（打印出来看趋势）
- 这类模型必须做 post-training（RLHF/DPO），否则推理时容易过度自信地胡说

### 坑 4：SwiGLU 的 ffn_hidden 没对齐到 256 倍数

**症状**：训练速度莫名下降几个百分点。
**根因**：维度不对齐导致 tile/矩阵乘法的 tail 效应浪费。
**对策**：`ffn_hidden = round_up(8/3 × hidden, 256)`

---

## 验收清单

- [ ] 能默写 Pre-LN 和 Post-LN 的公式，说出各自优缺点
- [ ] 能解释「Pre-LN 导致需要 post-training」这个因果链
- [ ] 能算出任意 (hidden, ffn_hidden) 下 SwiGLU 的参数量，并说明它为何守恒
- [ ] 能说明 gate 分支和量化冲突的根源
- [ ] 能解释 RMS ≠ Std

## 参考

- LLaMA — Meta 2023（§2.1 RMSNorm, §2.3 SwiGLU, §3.4 量化）
- RMSNorm — Zhang & Sennrich 2019
- DeepSeek-V2 — 2024（DeepSeek-V2 RMSNorm, in-block norm）