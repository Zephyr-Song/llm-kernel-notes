# 09 · 写 Kernel：从 PyTorch 到 Triton

> 整理中。这是把「理解」变成「产出」的一步。

## 先问：真的需要写 kernel 吗？

**大多数时候不需要。** 先排除这些：

| 替代方案 | 什么时候够用 |
|---|---|
| `torch.compile` | 能自动融合的算子（80% 的 elementwise / reduction） |
| `F.scaled_dot_product_attention` | 所有标准 attention |
| Triton 已有算子 | 常见 pattern 都有人写了 |
| cuBLAS / cuDNN | 标准 matmul / conv |
| torch.compile 的 max-autotune | 会搜最好的 kernel 组合 |

**只有在这 4 种情况才值得自己写：**
1. `torch.compile` 融合不了（数据依赖的动态 shape）
2. 需要硬件特定指令（Tensor Core 的 `mma`、TMA）
3. 新型计算 pattern，现有库没有
4. 需要极致优化（生产环境的推理服务）

---

## 性能分析：先判断是 compute-bound 还是 memory-bound

**Roofline Model**：
```
算术强度 = FLOPs / 访问的字节数
         ≥ 峰值算力 / 峰值带宽  → compute-bound
         <                        → memory-bound
```

实用捷径：
- **大 matmul**（batch × 大矩阵）→ compute-bound
- **小 matmul / elementwise / reduction** → memory-bound
- **norm / softmax /激活函数** → memory-bound（几乎总是）

**如果是 memory-bound，优化方向是减少访问的字节数（融合算子），不是减少计算。**

### 判断方法（PyTorch）

```python
with torch.profiler.profile() as prof:
    model(*inputs)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

看两个列：
- `self CUDA time total` — 这个kernel 花了多少时间
- `Memory` 列 — 访问了多少字节

**如果 kernel 耗时长但 FLOPs 很少 → memory-bound，该融合。**

---

## PyTorch 优化的三个层次（按收益排序）

### 层次 1：算法层面（收益最大，改动最小）

| 手段 | 收益 |
|---|---|
| 混合精度（bf16/fp16） | 2x（带宽减半） |
| 算子融合 | 减少中间结果的读写 |
| 消除冗余计算 | 视情况 |

**实例**：LayerNorm + 之后的 scale/shift 可以融合；GELU 和残差加法可以融合。

### 层次 2：`torch.compile`

```python
model = torch.compile(model, mode="max-autotune")
```

自动做：
- **算子融合**（最常见的优化）
- **CUDA graph 捕获**（消除 launch 开销）
- **autotune**（搜索最优 tile 配置）

**适用面比想象的大。** 很多手写 kernel 前应该先试 `torch.compile`。

**注意坑**：动态 shape 会触发 recompile。用 `dynamic=True` 或给 `mark_dynamic` 提示。

### 层次 3：手写 kernel

到这里才需要考虑 CUDA 或 Triton。

---

## CUDA 优化的核心概念

### 层次结构（从慢到快）

```
寄存器 (register)      最快，一个线程私有
  ↓
共享内存 (shared mem)   同一 block 内共享，需显式管理
  ↓
全局内存 (global mem)   所有 block 共享，约 400-900 GB/s (HBM)
```

**优化的本质：把数据从慢的层次搬到快的层次。**

### 必须理解的三个概念

#### 1. 合并内存访问（Coalescing）

线程是按 warp（32 个线程）为单位执行的。**如果一个 warp 的 32 个线程访问连续的地址，只需一次事务；如果访问跳跃地址，可能要 32 次事务。**

```cuda
// 差：线程 i 访问 data[i * 32]   —— 跳跃访问
// 好：线程 i 访问 data[base + i]  —— 连续访问
```

**这是最简单也最容易忽略的优化。** 写 kernel 时第一件事应该是检查内存访问模式。

#### 2. Bank Conflict（共享内存）

共享内存被分成 32 个 bank，同一个 bank 的并发访问会串行化。

```cuda
// 无冲突：线程 i 访问 smem[i]         （32 个线程 → 32 个不同 bank）
// 有冲突：线程 i 访问 smem[i * 32]    （全部落在同一个 bank，串行 32 倍）
```

**通常需要 padding 或改变数据布局来避免。**

#### 3. 计算强度（Arithmetic Intensity）

```
计算强度 = FLOPs / 字节数
```
这个值决定了 kernel 能不能跑满峰值算力。

---

## Triton：PyTorch 的实际选择

### 为什么优先 Triton 而不是 CUDA

| 维度 | Triton | CUDA |
|---|---|---|
| 编写语言 | Python | C++/CUDA |
| 编译 | JIT，通常几秒 | nvcc，提前编译 |
| 调试 | Python 调试器 | 难 |
| 性能（简单 kernel） | 通常够用 | 最优 |
| 性能（极致优化） | 接近 | 更优 |

**Triton 让你用 Python 写 kernel，性能通常能达到 CUDA 的 90%+。** 对于研究性质的优化，这个 trade-off 几乎总是划算的。

### 第一个 Triton kernel

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)          # 拿到程序 ID
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def triton_add(x, y):
    out = torch.empty_like(x)
    n = out.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out
```

### 关键概念

| 概念 | 说明 |
|---|---|
| `program_id(0)` | 当前程序（等价于 CUDA 的 blockIdx.x） |
| `tl.arange(0, BLOCK)` | 创建块内的线程索引向量（等价于 threadIdx） |
| `tl.load(ptr + offset, mask)` | 带mask 的加载，**必须写 mask** |
| `tl.store(...)` | 存储 |
| `BLOCK_SIZE: tl.constexpr` | 编译期常量，触发特化 |
| `tl.cdiv(a, b)` | 向上取整除法 |

**最重要的一条**：Triton 是 SIMT 编程模型 —— **没有显式的 thread**，你操作的是「块内的数据向量」，编译器负责分配到线程。写 kernel 时的心智模型和 CUDA 完全不同。

---

## 优化流程（可复用的方法论）

```
1. 测量基线
   ↓
2. 判断瓶颈（roofline）
   ├── memory-bound → 融合算子，减少访问字节数
   └── compute-bound → 更好的数学（更少的 FLOPs、更高的算术强度）
   ↓
3. 先试 PyTorch 层面的优化（dtype / 融合 / torch.compile）
   ↓
4. 还有提升空间 → profile 找出最慢的 kernel
   ↓
5. 写 Triton kernel 替代它
   ↓
6. 验证正确性（对拍 PyTorch 输出）
   ↓
7. benchmark 并autotune BLOCK_SIZE 等参数
   ↓
8. 边缘 case 测试（非对齐 shape、空输入、极端值）
```

**第6 步和第 8 步是最容易跳过的，也是最容易出生产 bug 的。**

---

## 性能优化的「陷阱」

### 陷阱 1：优化错了地方

**症状**：kernel 优化后总时间没变化。
**排查**：profile 整个 pipeline，看这个 kernel 到底占多少时间。**如果只占 5%，优化到最快也没用。**

### 陷阱 2：正确但没验证

**症状**：kernel 结果和 PyTorch 一致，但慢 2 倍（因为少了一个维度）。
**对策**：**永远先对拍正确性，再 benchmark。** 错误的结果比慢更可怕。

### 陷阱 3：autotune 参数过拟合

**症状**：在测试 shape 上飞快，换个 shape 就不行。
**根因**：BLOCK_SIZE 在特定 shape 上最优，但对其他 shape 未必。
**对策**：至少测3 个不同 shape（小/中/大），确认退化可接受。

### 陷阱 4：忽略 launch 开销

**症状**：单个 kernel 极小（如 elementwise）时，优化 kernel 本身没用，因为 launch 开销主导。
**对策**：这种情况应该用 CUDA Graph 或 `torch.compile` 的 graph 模式消除 launch 开销。

---

## 待补充

- [ ] `code/08_triton_fused_kernel.py`：融合 RMSNorm + 残差的 Triton 实现
- [ ] 写 kernel 的正确性验证模板（对拍 + 边界 case）
- [ ] torch.compile 的实际效果测量

## 参考

- Triton 官方教程 — OpenAI
- CUDA C++ Programming Guide — NVIDIA
- Roofline Model — Williams et al. 2009
- PyTorch Profiler 文档
- `flash-attn` 源码 — CUDA kernel 的教科书级实现