---
title: 论文阅读｜Dual-side Sparse Tensor Core
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2022-02-17 17:08:12
---

需要关注的：

1. Sparse Tensor Core [45] [42] resolves the irregularity of weight spar- sity by applying a structural pruning scheme, which enforces a constant 50% weight sparsity to balance the workload and to exploit parallelism in the dot-product unit

   Sparse Tensor Core的处理方法如下：

   ![image-20220218141644980](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20220218141644980.png)

2. 矩阵乘法以A的列，和B的行计算得到一个部分矩阵的办法是在哪儿提出的？

   答: 本质上就是把最里面那个k循环提到了最外面：https://patterns.eecs.berkeley.edu/?page_id=158
   
3. 为啥要扩展成8x8的呢？他们设计的 Multipy-value 应该不需要稀疏矩阵乘法

4. 不是很懂这个Outer-product friendly im2col的具体实现方法

<!-- more -->
