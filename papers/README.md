# 论文清单

按「必读 / 选读 / 有空看」分层。标注了每篇「解决什么问题」和「读完你应该能回答什么」。

---

## 必读（这些不读会跟不上现代 LLM 的基本语言）

### 架构与训练

| 论文 | 解决什么 | 读完能回答 |
|---|---|---|
| **Attention Is All You Need** (2017) | Transformer 原始设计 | Post-LN 的问题在哪 |
| **LLaMA** (2023) | 高效大模型架构 | RMSNorm/SwiGLU/RoPE 三个选择的动机；量化章节的 gate 问题 |
| **LLaMA 2** (2023) | 规模化+ 对齐 | 70B 为何用 GQA-8；post-training 流程 |
| **LLaMA 3** (2024) | 长上下文 + 全栈训练 | base=500000 的依据；数据配比；安全对齐 |
| **FlashAttention** (2022) | O(N²) 显存 | 在线 softmax 的推导；为什么省IO |
| **FlashAttention-2** (2023) | FA 的进一步优化 | 工作划分策略；为什么算得更好 |
| **RoFormer** (2021) | 相对位置编码 | RoPE 的相对位置性质证明 |
| **GQA** (2023) | KV cache 过大 | up-training 流程 |
| **DeepSeek-V2** (2024) | MLA + DeepSeek-V2 RMSNorm | 矩阵吸收技巧；in-block norm 的动机 |

### 推理与部署

| 论文 | 解决什么 |
|---|---|
| **vLLM / PagedAttention** (2023) | KV cache 碎片，显存利用率 20%→90% |
| **SmoothQuant** (2022) | 激活 outlier 导致量化难 |
| **GPTQ** (2022) | 二阶信息量化 |
| **AWQ** (2023) | 激活感知的重要性度量 |

---

## 选读（深入某个方向时读）

### 长上下文

- **NTK-aware Scaled RoPE** (bloc47) — 博客文，很短，讲清了插值思路
- **YaRN** (2023) — 分段插值 + 温度修正
- **LongRoPE** (2024) — 渐进式扩展
- **Longformer** (2020) — 稀疏注意力的经典
- **StreamingLLM** (2023) — 流式场景下的注意力 sink

> **读法建议**：先读 NTK-aware（20分钟），再读 YaRN（补充分段逻辑）。LongRoPE 只在你真的要训长上下文时才需要。

### MoE

- **Switch Transformer** (2021) — 简洁的 MoE baseline
- **Mixtral** (2023) — 8x7B 的实用 MoE
- **DeepSeek-V3** (2024) — 细粒度专家 + 负载均衡
- **MegaBlocks** (2022) — MoE 的高效 kernel

### 量化

- **LLM.int8()** (2022) — outlier 的第一次系统性处理
- **KIVI** (2023) — KV cache 的 2-bit 量化，per-channel 的必要性
- **GPTQ** (2022) — Hessian 视角
- **AWQ** (2023) — 激活感知
- **QLoRA** (2023) — 4bit 微调（如果做LoRA 相关）

### 架构变体

- **Mamba** (2023) — SSM，替代 attention
- **RetNet** (2023) — 显式衰减 + 可并行训练
- **DeepSeek-V2 RMSNorm** — 见 DeepSeek-V2
- **RingAttention** (2023) — 跨设备长序列

### 推理引擎与调度

- **Orca** (2022) — 迭代级调度，continuous batching 的理论基础
- **vLLM** (2023) — 见上
- **SGLang** (2023) — RadixAttention，前缀缓存
- **FlashInfer** (2024) — 高性能 attention 推理库
- **DistServe / Splitwise** (2023) — prefill/decode 分离

---

## 有空看（拓宽视野，不必精读）

- **nanoGPT / llm.c** (Karpathy) — 代码量最小的完整实现
- **minimind** — 中文社区的 64M 最小 LLM 教程
- **The Chinchilla** (2022) — 数据 vs 参数的 scaling law，**这篇其实值得精读**
- **Scaling Laws** (Kaplan) — 经典 scaling law
- **CL4R1T4S** — 各家 system prompt 泄露合集（研究 prompt 工程时有参考价值）

---

## 我推荐的学习顺序（含理由）

```
第1 周  LLaMA 论文（§2 架构部分）→ 建立现代 LLM 的零件清单
       配code/01_rmsnorm_swiglu.py 验证每个零件

第 2 周  FlashAttention + FlashAttention-2
        配 code/02_naive_attention.py 手写实现
       重点理解：为什么省的是 IO 不是 FLOPs

第 3 周  RoFormer + NTK-aware（博客）+ YaRN
        配 code/03_rope.py 验证相对位置性质
       重点理解：外推失败的主因是 PIT 而非位置编码未定义

第 4 周  GQA + vLLM/PagedAttention
        配 code/04_gqa_kvcache.py 算显存账
       重点理解：GQA 省带宽不省算力；PagedAttention 治碎片不治总量

第 5 周  Chinchilla（数据与参数的 scaling）+ AWQ
        重点理解：为什么现在的大模型都「过度训练」
        以及量化的重要性度量为什么选激活而非权重

第 6 周  DeepSeek-V2（MLA + in-block norm）
        这是目前 KV 压缩和 Norm 设计的最新答案
```

**为什么是这个顺序**：架构 → 注意力 → 位置编码 → KV 优化，这四步构成「理解一个现代 decoder-only LLM」的最小完备集。后面的 MoE、量化、推理引擎都是在这套骨架上的替换。

---

## 如何读论文（针对这个领域）

**别从 Introduction 开始读。** 推荐顺序：

```
1. Abstract（30秒）→ 判断是否相关
2. 看图表（5分钟）→ 架构图和性能图承载了80% 的信息
3. Method 里带公式的那几段 → 核心创新通常只有 2~3 个公式
4. Experiments 的 ablation → 作者自己认为什么最重要
5. 只有上面都懂了，才去读 Appendix 和逐行推导
```

**特别提醒**：LLM 领域的论文有大量「消融实验驱动」的写作习惯——一个消融表格比三千字论证更有说服力。**读 ablation 表比读正文更快理解作者的设计意图。**

---

## 补充资源（非论文）

- **《How to Scale Your Model》** (Jared Kaplan et al.) — 架构演进逻辑，2 小时读完，**强烈推荐作为第一篇**
- **Hugging Face transformers 源码** — 工业级实现，但抽象层多，要能穿透
- **llama.cpp** — C++ 实现，量化方案最全，适合读推理侧
- **vLLM 源码** — Python + CUDA，调度逻辑清晰，适合读系统侧
- **nanoGPT / llm.c** — 教学向，代码量小，适合先跑通建立全貌