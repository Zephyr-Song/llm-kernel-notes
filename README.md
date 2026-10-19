# LLM Architecture & Inference Kernels · 学习笔记

从「读得懂论文」到「写得动 kernel」的路线图。

这不是又一份 awesome-list。这是我自己按顺序踩过一遍、留下来的笔记 + 可跑的最小实现。
每篇笔记都遵循同一个结构：**直觉 → 公式 → 真实实现里长什么样 → 我踩的坑**。

---

## 为什么开这个仓库

学 LLM 有两个极端的坑：

- **只读论文** —— 看完 Attention 论文觉得自己懂了，但一打开 llama.cpp 或 vLLM 的源码不知道从哪看起。
- **只抄代码** —— 会用 `F.scaled_dot_product_attention`，但说不出它和朴素实现在数值上差在哪、为什么要 FlashAttention。

这个仓库的目标是走中间路线：**每个概念都有一个自己手写的最小实现**，能跑、能对拍、能踩坑。

---

## 学习路线

按依赖顺序排列，不要跳。每个阶段结束前把对应代码跑通并理解输出。

### 阶段 0 · 地基（1 周）

| 笔记 | 内容 | 对应代码 |
|---|---|---|
| [01-tensor-and-dtype.md](notes/01-tensor-and-dtype.md) | dtype 精度、bf16/fp16 差异、量化到 cuboid 的代价 | `code/00_dtype_bench.py` |

**为什么从dtype 开始**：所有 kernel 优化的本质都是「在可接受的精度损失下省内存带宽」。不懂 dtype，后面所有量化技巧都无从谈起。

### 阶段 1 · Transformer 骨架（2 周）

| 笔记 | 内容 | 对应代码 |
|---|---|---|
| [02-transformer-block.md](notes/02-transformer-block.md) | Pre-LN vs Post-LN、RMSNorm 为什么替代 LayerNorm、SwiGLU 的门控结构 | `code/01_rmsnorm_swiglu.py` |

### 阶段 2 · Attention（3 周，核心）

| 笔记 | 内容 | 对应代码 |
|---|---|---|
| [03-attention-from-scratch.md](notes/03-attention-from-scratch.md) | 从 O(N²) 到 FlashAttention 的完整推导链 | `code/02_naive_attention.py` |
| [04-rope.md](notes/04-rope.md) | RoPE 原理、NTK / YaRN 插值、外推为什么会崩 | `code/03_rope.py` |
| [05-mha-gqa-mqa.md](notes/05-mha-gqa-mqa.md) | 多查询变体的显存账怎么算、KV cache 为什么是瓶颈 | `code/04_gqa_kvcache.py` |

### 阶段 3 · 现代架构变体（2 周）

| 笔记 | 内容 | 对应代码 |
|---|---|---|
| [06-moe.md](notes/06-moe.md) | MoE 的路由、负载均衡损失、专家并行的通信代价 | — |
| [07-long-context.md](notes/07-long-context.md) | 从 MHA 到 MLA、稀疏注意力、位置外推的现状 | — |

### 阶段 4 · 推理引擎（3 周）

| 笔记 | 内容 |
|---|---|
| [08-inference-stack.md](notes/08-inference-stack.md) | vLLM PagedAttention、continuous batching、prefill/decode 两阶段 |
| [09-kernel-writing.md](notes/09-kernel-writing.md) | 从 PyTorch 到 Triton 的路怎么走、常见 IO bound 怎么判断 |

### 阶段 5 · 持续跟进（长期）

- [papers/README.md](papers/README.md) — 分层的论文清单，标注「必读 /选读 / 有空看」
- [notes/experiment-log.md](notes/experiment-log.md) — 我的实验记录，包括失败的

---

## 核心概念速查

按「踩坑优先级」排序，不是按定义。

| 概念 | 一句话本质 | 常见误解 |
|---|---|---|
| **RMSNorm** | LayerNorm 去掉均值中心化，省一次 reduce | 不是「简化版」，是**更好**的归一化 |
| **RoPE** | 用旋转把位置编码成相对位置 | 不是绝对位置编码，**外推时才会崩** |
| **GQA** | K/V 的 head 数少于 Q，省 KV cache | 不是「减少计算」，decode 阶段计算量不变，省的是**显存带宽** |
| **FlashAttention** | 分块 + 在线 softmax，避免物化 N² 矩阵 | 提升来自** IO 而非 FLOPs**，算力其实更多了 |
| **MoE** | 稀疏 FFN，参数量与计算量解耦 | 不是「免费变大」，代价是**通信**和负载不均 |
| **MLA** | 把 KV 压缩到低维 latent，再上投影 | 不是简单降维，是**换一种缓存格式** |
| **PagedAttention** | 操作系统分页思想解决 KV cache 碎片 | 不是为了省显存，是为了**消除碎片**导致的浪费 |
| **continuous batching** | 每个 decode step 动态进出请求 | 不是「批处理」，是**调度问题** |

---

## 手写实现清单

全部是**纯 NumPy / 纯 PyTorch 实现**，不依赖任何推理框架。目的是看清每一步。

```python
# 每个文件都能直接 python xxx.py 跑起来
code/00_dtype_bench.py       # dtype 精度与带宽对比
code/01_rmsnorm_swiglu.py    # RMSNorm + SwiGLU 逐行拆解 + 数值验证
code/02_naive_attention.py   # O(N²) 朴素实现 → 手写 FlashAttention → 对拍
code/03_rope.py             # RoPE + NTK 插值外推实验
code/04_gqa_kvcache.py       # GQA + KV cache 显存账精确计算与实测
```

**验收标准不是「跑通」，是「你能解释每一行为什么这样写」。**

---

## 方法论：怎么读 kernel 源码

以 vLLM / llama.cpp 为例，推荐顺序：

1. **先读 Python 层** —— 看清楚调用链和tensor shape 的变化。kernel 对应的是哪个 op，一目了然。
2. **再看 shape 检查和 assert** —— 这些 assert 往往是最宝贵的信息，告诉你这个 kernel 支持哪些非标准 shape。
3. **最后才看 CUDA/Triton 源码** —— 到这一步你只需要看三件事：grid/block 配置、内存访问模式（coalesced 有没有）、有没有用到 shared memory。

不要一上来就打开 `.cu` 文件从第一行读到最后一行，那是浪费时间。

---

## 相关资源

- **《How to Scale Your Model》(Jared Kaplan et al.)** —— 架构演进的逻辑讲得最清楚，2 小时读完
- **《FlashAttention-2》** —— 讲清楚了 IO-aware 调度为什么能进一步提速
- **Karpathy 的 nanoGPT / llm.c** —— 代码量最小的完整 LLM 实现，适合先跑通再看
- **Hugging Face transformers 源码** —— 工业级实现，但抽象层太多，看的时候要能穿透

> 详细的论文分层清单见 [papers/README.md](papers/README.md)。

---

## 约定

- 所有代码必须能跑，不能有伪代码
- 每个坑都要写清楚**症状 → 排查 → 根因**，这比结论本身更有价值
- 引用别人的东西必须标明出处
- 每周更新一次`notes/experiment-log.md`

---

*这个仓库会持续更新。如果你在某个概念上有不同理解，欢迎开issue 讨论 —— 讨论是我更新的主要动力。*