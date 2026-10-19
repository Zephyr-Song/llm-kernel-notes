# 实验与踩坑记录

> 这个文件是本仓库最有长期价值的部分。**结论会过时，但踩坑不会。**
> 格式：`日期 | 主题 | 症状 → 排查 → 根因 → 修正`

---

## 2026-10-19 · 仓库初始化

记录手写实现过程中真实踩到的 bug（都是跑起来才发现的，不是事后编的）。

### 坑1：causal_mask 用错shape 维度

**症状**
```
RuntimeError: The size of tensor a (64) must match the size of tensor b (2048)
at non-singleton dimension 2
```

**排查**
attention 的 shape 是 `[B, H, S, D]`：
- `shape[-1]` = head_dim = 64
- `shape[-2]` = seq_len = 2048

我写 `causal_mask(q.shape[-1])` → mask 变成 [64, 64]，和 [2048, 2048] 的 scores 对不上。

**根因**：变量名含混，用 `d` 同时指代序列和维度。

**修正**：`seq_q = q.shape[-2]`，并且在代码里**分开命名 `seq_len` 和 `head_dim`**。

> **教训**：张量操作里，**末两维的含义必须显式命名**。这类 bug 编译器不会报错，shape 恰好能广播时才更难发现。

---

### 坑 2：数值梯度检查里 inplace 修改叶子张量

**症状**
```
RuntimeError: expand(torch.FloatTensor{[64]}, size=[]):
the number of sizes provided (0) must be >= the number of dimensions (1)
```

**排查**：`w` 是叶子张量且 `requires_grad=True`。数值梯度检查里我做`w[idx] += eps` 这种 inplace 操作，破坏了 autograd 的版本计数。

**根因**：inplace 操作叶子张量 + 需要 grad → PyTorch 直接拒绝。

**修正**：用 `w_data = w.detach().clone()`，所有加减在副本上做：
```python
wd = w_data.clone(); wd[idx] += eps
loss_plus = f(x.detach(), wd)
```

> **教训**：任何数值梯度检查都要在**独立副本**上做。这个坑在写优化器测试时也常遇到。

---

### 坑 3：全量分配超大 tensor

**症状**
```
RuntimeError: [enforce fail at alloc_cpu.cpp:117]
DefaultCPUAllocator: not enough memory:
you tried to allocate 274877906944 bytes.
```

**排查**：`torch.randn(n, n if n <= 4096 else 4096)`，n=4096² 时第二个参数还是 4096，第一个是 16777216 → 分配了 256GB。

**根因**：写这行时脑子里的逻辑是「大就截断」，但实际是两个独立参数相乘。

**修正**：删掉那行，直接 `torch.randn(4096, 4096)`。

> **教训**：`randn` 的第一个参数和第二个参数相乘。想控制大小就只传一个 tuple。
>
> 顺带一句：这个错误的报错信息（"you tried to allocate N bytes"）其实非常有用的调试提示——看到具体字节数就能立刻反推哪里算错了。

---

### 坑 4：RoPE 相对位置验证里 cos/sin 表不匹配

**症状**：`apply_rope` 里 shape 不匹配。

**排查**：验证相对位置不变性时，我对**平移后的 q** 用了**未平移的 cos/sin 表**——这个场景本身是合法的（物理含义就是「q 被移动了」），但我重新建了表，掩盖了本意。

**修正**：直接复用原始 `cos_b / sin_b`，并加注释说明这里的物理含义。

> **教训**：做等价性验证时，**验证对象的定义要写清楚**。我这次的困惑来自「本来正确的语义」被写成了「看起来奇怪」的形式。

---

### 坑 5：fp32 → fp16 比值输出精度错误（次要）

**症状**：打印 `fp32_max / fp16_max` 得到一个 40 位的大整数。

**根因**：`np.float16.max` 是 numpy scalar，`.1e` 格式化对它无效。

**修正**：先 `.item()` 转成 Python float，再用 `.3e`。

---

## 验证通过的记录（这些是可信的基线）

跑通后记录关键数值，作为后续修改的回归基准：

```
00_dtype_bench.py       OK
01_rmsnorm_swiglu.py    OK
02_naive_attention.py   OK   Flash vs 朴素: max_diff = 2.98e-07  ✓ (< 1e-5)
03_rope.py              OK   相对位置验证: 极差 = 1.665e-15       ✓ (float64 精确)
04_gqa_kvcache.py       OK
```

**关键数字备查：**
- FlashAttention 对拍误差：2.98e-07（fp32 理论精度 ~1e-7，符合预期）
- RoPE 相对位置一致性：1.665e-15（float64 的机器精度极限，说明数学实现精确正确）
- KV cache MHA-7B @ 128K = 137 GB，GQA-8 = 34 GB，GQA-4 = 17 GB

---

## 待验证 / 存疑

- [ ] FlashAttention-2 的工作划分策略到底怎么分的，和 FA-1 差在哪（还没读源码）
- [ ] MLA 的矩阵吸收在 DeepSeek 实现里是否真的避免了上投影（需读 DeepSeek-V2 代码）
- [ ] PyTorch SDPA 在传自定义 float mask 时是否真的静默降级到 MATH（未实测）
- [ ] GPTQ 的 Hessian 具体怎么算（二阶信息指的是什么，需读原论文）