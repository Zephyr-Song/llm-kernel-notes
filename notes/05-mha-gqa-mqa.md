# 05 · MHA / GQA / MQA：KV cache 的攻防战

> 对应代码：`code/04_gqa_kvcache.py`

## KV cache 为什么存在

**没有 KV cache（每步全量重算）：**
```
第 1 步处理 1 个 token
第 2 步处理 2 个
...
第 L 步处理 L 个
总计算量 = O(L²/2)
```

而且这些重算的 K/V **和之前完全一样**（因果注意力下，位置 0 的 K 永远只依赖自己）。纯重复劳动。

**有 KV cache：**
```
每步只处理 1 个新 token，历史 K/V 直接复用
总计算量 = O(L)
```

L=2048 时理论上 2048 倍加速（当然有常数损耗）。

---

## Prefill vs Decode：两种截然不同的瓶颈

| | Prefill | Decode |
|---|---|---|
| 输入 | prompt 的 N 个 token，一次性 | 每步 1 个 token |
| 瓶颈 | **compute-bound**（算力用满） | **memory-bound**（带宽用满，算力闲置 90%+） |
| attention | O(N²) 大矩阵乘 | matrix-vector，读全量 KV cache |
| 影响指标 | TTFT（首 token 延迟） | TPOT（每 token 延迟） |
| 优化手段 | 算力优化（更好的 kernel、更低精度计算） | **带宽优化（减少要读的字节数）** |

**核心洞察**：Decode 阶段每生成 1 个 token 都要读完整个模型。
70B fp16 = 140GB，A100 带宽 2TB/s → 理论最快 70ms/token，实际远达不到。

**所以优化 Decode 的唯一主线是「减少每 token 要搬的字节数」：**
- 量化权重（减权重字节）→ AWQ / GPTQ
- GQA / MQA / MLA（减 KV 字节）→ 本文重点

---

## KV cache 显存公式

```
KV_bytes = 2 × n_layers × n_kv_heads × head_dim × seq_len × bytes_per_elem × batch_size
```

**注意：没有 `n_q_heads` 项。这就是 GQA 省显存的全部秘密。**

实测数据（fp16, batch=1）：

| 配置 | n_kv | 4K | 32K | 128K |
|---|---|---|---|---|
| MHA-7B | 32 | 4.19 GB | 33.6 GB | 137 GB |
| GQA-8 (Mistral) | 8 | 1.05 GB | 8.4 GB | 34 GB |
| GQA-4 (Qwen2) | 4 | 0.52 GB | 4.2 GB | 17 GB |

**结论**：MHA 在 128K 上下文时 KV cache 要 137GB，单卡放不下。GQA-4 降到 17GB，长上下文从「不可能」变成「可行」。

---

## GQA / MQA 的实现与关键认知

```python
# GQA：n_q=8, n_kv=2, groups=4
K_exp = K.repeat_interleave(n_groups, dim=1)   # [B, 8, S, D]
out = F.scaled_dot_product_attention(Q, K_exp, V_exp, is_causal=True)
```

### 必须纠正的常见误解

> **GQA 不省 FLOPs。**

展开后 K 的 head 数是 `n_q` 而非 `n_kv`，所以 attention 的 matmul 规模不变。**Q 的每个 head 都还是要和 K 做完整 attention。**

GQA 省的是：
1. **KV cache 显存** — 只存 `n_kv` 份，省 `n_q/n_kv` 倍
2. **Decode 带宽** — 每步读 KV 的字节数减少 → 直接提升 TPS
3. **Prefill 的 K/V 投影计算量** — 从 `n_q` 个 head 降到 `n_kv` 个（占比不大）

**这就是为什么「GQA 提升速度」这个说法只在 decode 阶段成立，且是带宽角度而非算力角度。**

### 三种变体

| 变体 | n_kv | KV 省 | 代表模型 |
|---|---|---|---|
| MHA | = n_q | 1x | GPT-2, LLaMA-1 |
| MQA | 1 | n_q x | Gemma, PaLM |
| **GQA** | 1 < n_kv < n_q | 可调 | **LLaMA-2/3, Mistral, Qwen** |

**GQA 是默认选择**，因为它在「省多少」和「效果掉多少」之间提供了可调的平衡点。

### 现代模型的实际配置

| 模型 | n_q | n_kv | 类型 |
|---|---|---|---|
| LLaMA-2 70B | 64 | 8 | GQA-8 |
| Mistral 7B | 32 | 8 | GQA-4 |
| LLaMA-3 8B | 32 | 8 | GQA-4 |
| Qwen2 7B | 28 | 4 | GQA-7 |
| Gemma 2B | 8 | 1 | MQA |

**小模型用更激进的 MQA**（因为 n_q 本来就小，MQA 损失可控），大模型用保守的 GQA-8。

---

## 从 MHA 迁移到 GQA：up-training

**问题**：Qwen 从 MHA 迁移到 GQA 时，不能直接砍掉 K/V 的 head —— 那样会破坏已学好的注意力分布。

**解法（Google 提出的 GQA up-training）**：
```
1. 复制 MHA 的 K/V 权重，得到 M 个候选投影
2. 为每组内 M 个 K 投影各加一个可学习向量 b_i:  k'_m = k_m + b_m
3. 前向：k'_mean = mean(k'_1..k'_M)，用 k'_mean 做 attention
4. 反向：梯度回传给 k'_mean（不更新各 k_m 的参数）
5. 预训练结束：k_new = k'_mean，丢弃 b 和其他 M-1 个 k
```

**为什么有效**：均值是比随机初始化好得多的起点，同时保留了「组内各 head 加不同偏移」的调整能力。

DeepSeek 的 MLA 走了另一条路：**从头设计，不需迁移。**

---

## 更激进的 KV 压缩

### 1. KV cache 量化 —— 最直接

```
fp16 → fp8 : 显存和带宽减半，精度损失小
fp16 → int4: 降到 1/4
```

**关键坑**（Qwen2 技术报告特别讨论过）：
> 如果模型训练时见过大量数值大的激活（如 MoE 的 gate），它的 KV cache 会有 outlier，**直接 per-tensor 量化误差很大**。必须用 per-channel / per-head 分组量化。

**这是我在 `00_dtype_bench.py` 里演示的那个坑在真实场景的重演** —— 权重和激活的 outlier 处理策略是一样的。

### 2. PagedAttention（vLLM）—— 解决碎片

**问题**：连续分配 KV cache 会产生碎片
- 请求长度不确定 → 预留多了浪费，少了要迁移
- 传统方式显存利用率只有 **20~40%**

**解法**：操作系统分页思想
- KV cache 切成固定大小 block（如 16 token/block）
- 用 block table 记录逻辑地址 → 物理地址映射
- 按需分配，用完即释放

**收益**：显存利用率 20~40% → **90%+**，可服务 3~4倍并发。

> **注意**：PagedAttention 不减少总量，只消除碎片。它和量化/GQA 是**正交**的优化，可以叠加。

### 3. MLA（Multi-head Latent Attention，DeepSeek-V2）

**核心思想**：既然 KV cache 太大，那就只存它的「压缩版」。

```
标准:   cache K (128维) 和 V (128维)  → 每个 head 存 256 维
MLA:    下投影到低维 latent c_KV（约 512 维，整个层共享）
        cache 里只存 c_KV
        需要 K 时上投影: k = W_UK · c_KV
```

**关键洞察（矩阵吸收技巧）**：
```
注意力分数 = q^T k = q^T W_UK c_KV = (W_UK^T q)^T c_KV
                                    ↑
                    W_UK^T q 只需算一次，然后和 c_KV 做内积
```
也就是说，上投影的耗时被「吸收」进了 Q 的投影，可以提前算好。**不需要真的把 K 重建出来。**

**结果**：KV cache 比 GQA-8 还小 **93.3%**，效果还更好。这是目前压缩 KV cache 的最优解。

### 4. 滑动窗口注意力
- Mistral / Gemma-2 采用
- 每 token 只 attend 最近 W 个（如 4096）
- **KV cache 大小与序列长度解耦**
- 代价：无法做真正的长程检索

### 5. 跨层 KV 共享
- YOCO / CLA 提出
- 相邻层复用同一份 KV
- 收益随层数增长

---

## 工程优先级建议

```
1. GQA（模型自带，必须有）
2. PagedAttention（vLLM 默认，框架层解决）
3. FP8 KV cache（简单有效）
4. 分组量化 KV cache（如果精度不够）
5. MLA（要改架构，训练时就定死）
```

---

## 我踩的坑

### 坑 1：以为 GQA 省算力
**症状**：预估 GQA-8 应该有 8x 加速，实际只快 1.2x。
**根因**：decode 阶段 attention 的计算量本来就小（1 个 query vs N 个 key），瓶颈在**读 K/V 的带宽**和**读权重**，不是算力。省掉的是 K/V 读取量，不是 FLOPs。

### 坑 2：causal mask 在 decode 阶段用错了
**症状**：生成结果混乱，重复 token。
**根因**：decode 时新 token 应该看到**所有 cache 内容**，所以 `causal=False`。如果还传 `causal=True`，PyTorch 会把 cache 里的历史位置也屏蔽掉。
**正解**：历史位置在 prefill 阶段已经保证了因果性，decode 阶段的新 token 可以看到全部历史。

### 坑 3：拿 `q.shape[-1]` 当序列长度
**症状**：`causal_mask` 创建出错误 shape，`RuntimeError: size of tensor a (64) must match ... at dimension 2`。
**根因**：attention 的 shape 是 `[B, H, S, D]`，序列维是 `[-2]`（S），`[-1]` 是 head_dim。
**正解**：写代码时明确区分 `seq_len` 和 `head_dim` 两个变量名，别都用 `d`。

---

## 验收清单

- [ ] 算出任意配置 (n_layers, n_kv, head_dim, seq_len) 的 KV cache 大小
- [ ] 解释为什么 GQA 不省 FLOPs，只省带宽和显存
- [ ] 说明 MQA / GQA / MHA 的取舍
- [ ] 复述 MHA → GQA 的 up-training 流程
- [ ] 解释 MLA 的矩阵吸收技巧为什么能消除上投影开销
- [ ] 区分 PagedAttention（治碎片）和量化（治总量）

## 参考

- GQA — Ainslie et al. 2023
- MQA — Shazeer 2019
- MLA / DeepSeek-V2 — DeepSeek 2024
- PagedAttention / vLLM — Kwon et al. 2023
- KIVI — 2023（KV cache 2-bit 量化）
- YOCO — 2024（跨层 KV 共享）