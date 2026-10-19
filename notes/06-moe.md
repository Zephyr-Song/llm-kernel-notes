# 06 · MoE：参数量与计算量的解耦

> 对应代码：暂无（待补 `code/05_moe_router.py`）

## 核心动机

稠密模型有硬约束：**每个 token 都激活全部参数**。

| 模型 | 参数 | 激活参数 | 显存需求 | 推理成本 |
|---|---|---|---|---|
| LLaMA-2 7B | 7B | 7B | 14GB | 基准 |
| Mixtral 8x7B | 47B | 13B | 94GB | ~1.8x |
| DeepSeek-V3 671B | 671B | 37B | 1.3TB | ~5x |

**MoE 的核心：把「参数总量」和「单token 计算量」解耦。**

代价从「算力」转移到了「显存 + 通信」。

---

## 稀疏 FFN 的结构

稠密 FFN：
```
FFN(x) = down( act(x @ W_gate_up) )      # 所有通道都算
```

MoE FFN：
```
FFN(x) = Σ_{i in TopK(Router(x))} E_i(x)  # 只算 K 个专家（K << N）
Router(x) = softmax(x @ W_router)         # [N] 个专家的打分
TopK 取分数最高的 K 个
```

**关键点**：MoE 替换的是 FFN，**Attention 保持稠密**。这是几乎所有 MoE 模型的共同选择（DeepSeek 的 MLA + MoE 组合也是）。

---

## 负载均衡：一个必须理解的核心问题

### 问题

Router 是学出来的。如果不干预，会出现**专家坍塌（expert collapse）**：
- 少数几个专家被所有 token 选中 → 它们过载训练
- 其余专家永远收不到梯度 → 完全不学习、浪费显存

### 解决方案：辅助损失

```
L_aux = N · Σ_i f_i · P_i
```

- `f_i` = 训练 batch 中被路由到专家 i 的 token **比例**
- `P_i` = 专家 i 在 batch 中获得的**平均路由概率**
- 系数 `N` = 专家总数（让 loss 量级与主 loss 可比）

**直觉**：`f_i` 是「实际使用率」，`P_i` 是「意图使用率」。最小化这个乘积会同时压低两者的方差，逼迫 router 分配均匀。

**具体做法**：对每个 batch，计算这两个向量后求元素积之和。**这个损失会加到主训练 loss 上，通常权重 0.001~0.01。**

### 读论文时的警示

**看到声称「MoE 训练更稳定」的论文，先确认它是不是偷偷用了更激进的 aux loss 或者调大了 batch size。** 因为 MoE 的很多「优势」其实是负载均衡做好的结果，而不是稀疏架构本身的。

---

## 专家并行的通信代价

这是 MoE 在实践中最大的坑。

```
Batch 分成 G 份，分发到 G 个 GPU
每层 FFN 前后各要 All-to-All 通信一次
```

**通信量**：每个 token 要把自己的 hidden state 发送到它选中的 K 个专家所在的 GPU。

**对比稠密模型**：稠密 FFN 只需 All-Reduce（或 TP 下的 All-Reduce），通信模式规整。

**MoE 的现实困难**：
1. **通信量大**：MoE 通常比同等参数的稠密模型慢，因为通信成为瓶颈
2. **负载不均**：token 路由不均导致某些 GPU 等其他 GPU（straggler）
3. **实现复杂**：需要和 DP/TP/PP 三维并行组合，配置组合爆炸

**缓解手段**：
- **EPLB（Expert Parallel Load Balancer）** — 动态重分配专家到 GPU
- **DeepSeek 的 Device-DP + Expert-EP 双维并行** — 把计算密度高的节点专门用于专家计算
- **Token drop / capacity factor** — 超容量的 token 直接丢弃（早期 MoE 的做法，有精度损失）

---

## 现代 MoE 的细粒度趋势

```
Mixtral 8x7B      : 8 个专家，选 2    (粗粒度，d_ff=14336)
DeepSeek-V3       : 256 个专家，选 8   (细粒度，d_ff=2048)
                    + 1 个共享专家(永远激活)
```

**细粒度 + 共享专家** 是现在的标准配置：

- **细粒度**：更多专家，每个更小 → 参数分配更灵活，总激活量不变
- **共享专家**：一个永远激活的 FFN，负责捕获所有 token 都需要的共性知识，让路由专家专注于差异化知识

DeepSeek-V3 还有一个设计：**MoE 层和 Attention 层用不同的 expert 数量**（第一层 dense，之后每两层一个 MoE 层），进一步降低通信频率。

---

## 显存账：MoE 的隐藏成本

MoE 的「省算力」常被误读为「省显存」。实际相反：

```
Mixtral 8x7B:
  参数量 47B (比 13B 大 3.6 倍)
  → fp16 显存 = 94GB，而 13B 只要 26GB
```

**MoE 用显存换算力。** 如果你的目标是「单卡跑大模型」，MoE 通常帮不上忙；如果是「多卡跑更大模型且追求吞吐」，MoE 很有价值。

---

## 路由的三种变体

| 方式 | 做法 | 代表 |
|---|---|---|
| Token Choice | 每个 token 选 TopK 个专家 | Switch, Mixtral（主流） |
| Expert Choice | 每个专家从所有 token 里选 TopN 个 | MoE, DeepSeek V2/V3（负载天然均衡） |
| Shared Expert | 额外一个永远激活的专家 | DeepSeek V3 |

**Expert Choice** 的优势：**负载天然均衡**（每个专家的处理量由设计固定），不需要 aux loss。但可能丢弃 token，且需要和 DP 通信配合。

---

## 我踩的坑（预告）

写完`code/05_moe_router.py` 后补充。已有的相关坑：

### 坑：MoE 训练时 aux loss 权重敏感
- aux loss 权重太小 → 专家坍塌
- 太大 → router 学不动，主任务性能下降
- 常见做法是 aux loss 只在训练后期逐渐升权，前期让模型先学会主任务

### 坑：batch size 必须大
- MoE 的负载均衡是**统计意义上的**，batch 太小则每个 batch 的负载方差极大，无法学到均衡
- 实践中 MoE 的 global batch size 往往要比同规模稠密模型大 4~8 倍

---

## 验收清单

- [ ] 解释 MoE 的「参数量与计算量解耦」，并算出 Mixtral 8x7B 的激活参数量
- [ ] 写出负载均衡辅助损失的公式并解释每一项
- [ ] 说明 Expert Parallel 的通信模式和它的成本
- [ ] 解释为什么「共享专家」能提升效果
- [ ] 说清 MoE 是省算力还是省显存（**是省显存！**）

## 参考

- Switch Transformer — Fedus et al. 2021
- Mixtral of Experts — Mistral 2023
- ST-MoE — StableMoE 2022（负载均衡的系统性研究）
- DeepSeek-V3 — 2024（细粒度 + 共享专家 + 无辅助损失的负载均衡）
- MegaBlocks — 2022（MoE 的高效 kernel）