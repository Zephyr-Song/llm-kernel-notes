# 01 · Tensor 与 dtype：一切优化的地基

> 对应代码：`code/00_dtype_bench.py`

## 直觉先行

如果只记住一件事：**推理性能问题，90% 是带宽问题，不是算力问题。**

- **Prefill**（处理 prompt）：算力瓶颈。GPU 算力跑满，算得快。
- **Decode**（逐 token 生成）：带宽瓶颈。每生成 1 个 token 都要把整个模型读一遍，算力闲置 90%+。

所以任何优化手段，如果没减少「要搬运的字节数」，收益就很有限。这条判断标准能帮你砍掉大部分无效尝试。

## dtype 的属性

| dtype | 字节 | 指数位 | 尾数位 | 动态范围 | 相对精度 |
|---|---|---|---|---|---|
| float32 | 4 | 8 | 23 | ~3.4e38 | ~1e-7 |
| float16 | 2 | 5 | 10 | ~65504 | ~1e-3 |
| bfloat16 | 2 | 8 | 7 | ~3.4e38 | ~1e-2 |
| int8 | 1 | 0 | 7 | [-128,127] | 需 scale |

### fp16 vs bf16：为什么大模型选 bf16

两者都是 2 字节，区别在指数位/尾数位的分配：

```
fp16:  1 + 5 exp + 10 mantissa  → 范围窄，精度高
bf16:  1 + 8 exp +  7 mantissa  → 范围宽，精度低
```

**选择逻辑：**
- 训练时用 fp16，因为需要精度，且激活值范围可控（小模型）。
- 大模型推理用 bf16，因为 70B 模型的激活值经常超过 65504（fp16 上限 → `inf`）。

LLaMA 全系用 bf16就是这个原因。早期用 fp16 训练时经常遇到 loss 突然变 `NaN`，根因就是激活溢出。

> 注：很多实现会在 fp16 上加 loss scaling 来扩大有效范围，这是另一种解法。但 bf16 更省事，代价是精度低一些。

## 量化：真正省带宽的手段

### per-tensor vs per-channel

per-tensor 用一个 scale 覆盖所有权重：
```python
scale = w.abs().max() / 127
q = clamp(round(w / scale), -128, 127)
dequant = q * scale
```

问题是**如果不同通道的幅度差异大，小幅度的通道会被彻底压成 0**。

per-channel（按 output channel 分组算 scale）：
```python
scale = w.abs().amax(dim=1, keepdim=True) / 127   # 每行一个 scale
q = clamp(round(w / scale), -128, 127)
dequant = q * scale
```

实测（见代码输出）：per-channel 的 MAE 误差比 per-tensor 低一个数量级，因为每个通道的 scale 精确匹配了它的幅度。

**现代方案的演进路径：**
```
per-tensor (LLM.int8, 粒度最粗)
  → per-channel (SmoothQuant)
    → per-group, group_size=128 (GPTQ / AWQ)
      → 非均匀位宽分配 (AWQ)
```

代价是**要额外存scale**。group_size=128、int8 时，scale 占比约1/129，可忽略。

### AWQ 的核心洞察

AWQ（Activation-aware Weight Quantization）的观察很有价值：

> **不是所有权重的重要性相同。有些权重通道对输出影响特别大，即使幅度很小。**

它按「激活值大小」而不是「权重幅度」来分配重要性，然后保护那些**幅度小但激活频繁**的权重通道。效果是：**4-bit 量化的困惑度损失从不可接受降到接近 fp16**。

这个思路的迁移价值：你做任何量化/剪枝/压缩，都应该问「我的重要性度量选对了吗」。

## 我踩的坑

### 坑1：以为量化只影响精度

**症状**：量化后单看指标（困惑度、BLEU）只掉一点，但实际推理慢得离谱或反而更慢。
**根因**：小 batch 下量化 kernel 的 dequantize 开销可能超过省下的带宽。量化要批量化处理（per-token/per-group 融合算子）才划算。
**结论**：量化是**大batch 场景**的优化。单请求小 batch 场景收益有限。

### 坑2：per-tensor 量化激活值时遇到 outlier

**症状**：某些 token 输出直接乱码。
**根因**：LLM 激活值有极端 outlier（某些通道能到其他通道的 100 倍）。per-tensor 的 scale 被 outlier 撑大，其他通道全部压成 0。
**解法**：per-channel / per-token 量化，或者先做 SmoothQuant 把激活的 outlier 迁移到权重。
**延伸**：Qwen2 的技术报告专门讨论过这个问题——**模型如果训练时见过大量大激活（MoE 的 gate），它的 KV cache outlier 会更严重**，所以量化方案要区别对待。

### 坑3：混淆「精度损失」和「不可逆退化」

fp32→fp16 是可接受的精度损失（相对误差 1e-3）。int8 per-tensor 是**信息丢失**（小幅度通道直接变0，不可恢复）。前者是平滑退化，后者是结构性破坏。

## 验收清单

学完这一节，你应该能回答：

- [ ] 为什么 decode 阶段是 memory-bound？给出量化后的算术强度计算
- [ ] fp16 和 bf16 差在哪？为什么大模型都用 bf16
- [ ] per-tensor 和 per-channel 量化的误差差多少？为什么
- [ ] AWQ 和 GPTQ 的核心差异是什么
- [ ] 量化在什么 batch size 下才划算

## 参考

- LLM.int8() — Dettmers et al. 2022
- SmoothQuant — Xiao et al. 2022
- GPTQ — Frantar et al. 2022
- AWQ — Lin et al. 2023
- KIVI — 2023（KV cache 的 2-bit 量化，per-channel）