# 08 · 推理引擎栈：从单次前向到服务

> 整理中。核心是「一个请求从进来到一个 token 出来，中间发生了什么」。

## Prefill vs Decode 的系统视角

| | Prefill | Decode |
|---|---|---|
| 阶段 | 处理 prompt 的 N 个 token | 逐个生成，每步 1 个 |
| 瓶颈 | **compute-bound** | **memory-bound** |
| 关键指标 | TTFT | TPOT |
| 优化方向 | 算力（更好的 kernel、TP） | 带宽（量化、GQA、批处理） |

**为什么这个区分如此重要**：
- 算力优化（更快的 matmul）对 decode 几乎无效，因为 decode 算力本来就用不满
- 带宽优化（减精度）对 prefill 收益小，因为 prefill 算力已经用满

**这个判断能砍掉大部分无效优化尝试。**

---

## 朴素服务的三个致命问题

### 问题 1：静态 batching
```
一批请求进来 → 全部 pad 到最长长度 → 推理 → 全部等最慢的那个
```
长度差异大时，短请求的计算被浪费在 padding 上。

### 问题 2：无 KV cache 复用
相同前缀的请求（few-shot prompt、多轮对话历史）各自算一遍 KV，浪费。

### 问题 3：显存碎片
连续分配 KV cache，请求长度不定 → 预留多则浪费、少则迁移 → 利用率仅 20~40%。

---

## vLLM 的三个核心设计

### 1. PagedAttention —— 治碎片

借鉴操作系统的分页机制：
```
KV cache 切成固定大小 block（如 16 token）
block table: 逻辑块号 → 物理块号 的映射
按需分配，用完即释放
```

**收益**：显存利用率 20~40% → **90%+**

**注意**：它**不减少总量**，只消除碎片。和量化/GQA 是正交的，可以叠加。

### 2. Continuous Batching —— 治静态 batching

传统：等一批全部生成完，再处理下一批。
Continuous：**每个 decode step 动态检查，完成的请求移出、新请求插入。**

```
Step 1: [A, B, C]
Step 2: [A, B, C, D]        ← D 插入
Step 3: [A, C, D, E, F]     ← B 完成移出，E/F 插入
```

**收益**：吞吐提升可达 **10~20 倍**（相比静态 batching）。

**代价**：需要更复杂的状态管理；单个请求的延迟变得不可预测。

### 3. Prefix Caching —— 治重复计算

用 Radix Tree 组织 KV cache，识别请求间的前缀重叠：
```
请求1: "系统提示 + 问题A"        → 计算 KV
请求2: "系统提示 + 问题B"        → 「系统提示」的 KV 直接复用
```

**收益**：多轮对话、few-shot 场景下省掉大量重复 prefill。

---

## 推理侧的调度取舍

真实服务里你在做多目标优化：

| 指标 | 优化手段 | 代价 |
|---|---|---|
| **TTFT**（首token 延迟） | chunked prefill、大 batch | prefill 抢占 decode，TPOT 上升 |
| **TPOT**（每 token 延迟） | decode 优先、KV 压缩 | TTFT 变差 |
| **吞吐量**（tokens/s 总） | continuous batching、大 batch | **两个延迟指标都变差** |

**这三个目标本质冲突。** 具体怎么取舍取决于场景：
- **聊天产品**：TPOT 优先（用户能接受首字慢一点，但不能接受打字慢）
- **批量处理**：吞吐量优先，延迟无所谓

**ORca 的关键发现**：单纯增大 batch 会在 batch 变大后收益递减，因为长请求会拖住整批。更好的策略是**迭代级调度（iteration-level scheduling）**——把 prefill 切碎，和 decode 交错执行，让短请求不被长 prefill 饿死。

---

## 量化方案在推理侧的位置

| 量化对象 | 收益 | 风险 |
|---|---|---|
| **权重 (W8/W4)** | decode 带宽减半/减四 | 需 per-group scale；精度略降 |
| **KV cache (FP8)** | 长上下文显存大减 | **长上下文检索能力下降** |
| 激活 | 通常不量化 | 实现复杂，收益小 |

**关键认知**：量化 KV cache 在短上下文上几乎无损，在长上下文上代价明显。**这是「宣称长度≠有效长度」差距的重要来源之一。**

---

## 工程上必须先测量的三件事

在优化任何东西之前，先拿到基线：

### 1. 实际的 compute / memory 占比
```python
# PyTorch profiler 能直接告诉你
with torch.profiler.profile() as prof:
    model(*inputs)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```
**如果你的 "compute-bound" 场景里 memory 占比超过 30%，说明瓶颈判断错了。**

### 2. 实际的 kernel（不是假设的 kernel）
```python
# SDPA 可能静默降级到 MATH 后端
from torch.nn.attention import SDPBackend, sdpa_kernel
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    ...  # 如果不报错，说明 Flash 路径可用
```
**这是我真实踩过的坑**：传了自定义 float mask → 静默降级到 MATH → 性能暴跌几十倍且无任何警告。

### 3. 真实的长度分布
不是平均长度，而是 **P50 / P90 / P99**。静态 batch 的效率完全取决于长度方差。**P99 很长时，静态 batching 的浪费是灾难性的。**

---

## 常见性能问题排查顺序

遇到「我的模型很慢」，按这个顺序查：

```
1. 是不是 memory-bound？看 profiler 的 memory 占比
2. 有没有静默降级？打印实际的 kernel 名（不是猜的）
3. batch 里的长度方差多大？P99 是不是远超 P50
4. 量化生效了吗？打印实际的 dtype
5. 有没有多余的数据同步？（.item() / .cpu() 会强制同步）
6. KV cache 显存占多少？碎片率多少？
```

**第 5 条特别常见**：训练代码里一句 `loss.item()` 就够把吞吐砍掉一半（每个 step 强制 GPU→CPU 同步）。

---

## 待补充

- [ ] `code/07_inference_bench.py`：用 FakeTensor 搭一个推理服务的性能剖析
- [ ] chunked prefill 的调度逻辑
- [ ] PD 分离（Prefill/Decode 分离部署）的收益分析

## 参考

- vLLM — Kwon et al. 2023
- Orca — Yu et al. 2022（迭代级调度）
- SGLang / RadixAttention — Zheng et al. 2023
- DistServe / Splitwise — 2023（PD 分离）
- FlashInfer — 2024
- Roofline Model — Williams et al. 2009（性能分析的基本工具）