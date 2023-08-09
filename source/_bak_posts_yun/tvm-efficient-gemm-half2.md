---
title: tvm efficient gemm half2
categories:
  - Technical
tags:
  - CUDA Programming
  - MLSys
date: 2022-09-09 21:23:22
---

在之前的两篇文章中，我们分别用TVM的Tensor Expression与TIR Script完成了在Nvidia Cuda Core上的高效的FP32 矩阵乘法，3090-24GB的各种精度在Cuda Core和Tensor Core上的Peak TFLOPS如下表所示：

| **3090-24GB** | **FP32** | **FP16**   | **BF16**   | **INT32** | **INT8**   | **INT4**    |
| ------------- | -------- | ---------- | ---------- | --------- | ---------- | ----------- |
| Cuda Core     | 35.6     | 35.6       | 35.6       | 17.8      | 71.2       | \           |
| Tensor Core   | \        | 142 / 284* | 142 / 284* | \         | 284 / 568* | 568 / 1136* |

有意思的是，3090上，FP16的Peak Peformance和FP32是一样的，这一点比较特殊，是因为架构上的改动，一般而言fp16的性能都会是fp32的两倍或者四倍，这个主要是因为20系的gpu把fp32和int32的Cuda Core分开了，从而能同时进行fp32和int32的计算，30系把int32的core又就加上了fp32的计算单元，所以fp32的计算能力翻倍，而cutlass下的16384的gemm。

按照3090上的硬件单元分类，我们还可以探索一些有意思的加速，比如在CUDA Core上使用SIMD指令（DP4A，HFMA2来优化int8、half的性能，

<!-- more -->

### native replace

numpy在做dot的时候不会主动使用gpu，导致在这一步过程中会卡住（等了几十分钟他都没有算完，所以这一步就注释掉了。

```python
tvm.testing.assert_allclose(c.numpy(), np.dot(b_np.T, a_np), rtol=1e1)
```

直接把之前程序中的float32替换成305.729ms，

