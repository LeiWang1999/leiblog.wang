---
title: TVM中的Shared Memory Reuse Pass 分析
categories:
  - Technical
tags:
  - TVM
  - MLSys
date: 2024-09-14 15:13:08
---

近期在基于TVM(其实是bitblas.tl) 复现PPoPP 2023的一篇论文[Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU](http://arxiv.org/abs/2301.03598) . 简单来说，这个方法可以把k轴均匀地切分到每个SM上，从而缓解小shape下的SM浪费问题（BitBLAS在Contiguous Batching等场景上确实碰到了这样的问题，为了优化这部分性能不得已去复现这个论文的方法。然而这篇Blog不讲Stream-K的算法与实现细节，也不讲BitBLAS, 而是来分析一下TVM的MergeSharedMemoryAllocations这一个Pass，原因是高效的Stream-K实现需要引入大量的shared memory，而TVM中负责进行Liveness分析来合并shared memory访存的这个Pass，在复杂场景下存在BUG，导致shared memory的复用达不到预期，阻止了我们探索更大的tile size. 为此不得不对这个Pass进行一下改进，本文记录一下对这个Pass的分析和修改，以及我相信大部分TVM的用户在Hack TVM的代码的时候都会头秃，穿插一些TVM的设计和调试经验）



<div align="center" ><img src="https://github.com/LeiWang1999/Stream-k.tvm/raw/master/figures/image.png" alt="example" style="zoom:33%;" /></div>



<!-- more -->

### 为什么需要 `MergeSharedMemoryAllocations` 这个 Pass

在高性能的GPU Kernel中，**共享内存（shared memory）** 的使用对于性能优化至关重要，普通的Tile划分需要在Shared Memory上做Cache，软件流水还会成倍得增加Shared Memory的使用，Block内跨线程Reduce等操作也需要通过Shared Memory作为媒介。以CUTLASS为例，不难发现高性能的Kernel都有着不低的Stage(及软件流水的层数，一般为3，或者4)，这同样代表着高性能的Kernel需要使用不小的Shared Memory空间。试想一下，用户仿照CUTLASS的Tile Size手写一个高性能的Kernel，但是因为没有做Shared Memory的Reuse，导致使用的Shared Memory比CUTLASS多出一半，往往就会导致编译失败，丧失了一些优化机会。

而显然，我们可以使用一些静态分析的方法，例如Liveness Analysis，来分析出每个Buffer的生命周期，从而求解出一个Shared Memory Reuse的方案，而在TVM中，实现这一方案的Pass就是**MergeSharedMemoryAllocations**, 其主要功能是合并多次使用但生命周期不重叠的共享内存块。通过这样的合并操作，MLC(Machine Learning Compiler)可以减小存储器的碎片化，提升计算的性能和资源利用率。

考虑一个简单的矩阵乘法（Matrix-Matrix Multiplication, GEMM）场景，在这种场景下我们需要把输入矩阵和部分结果临时存储在共享内存中以加快计算速度。假设我们要计算矩阵 `C = A * B`，其中矩阵 `A` 的维度为 `MxK`，矩阵 `B` 的维度为 `KxN`。

在传统的Tile分块矩阵乘法（Block Matrix Multiplication）算法中，我们通常会将矩阵 `A` 和 `B` 切分成多个小块（Tile），并将这些块加载到共享内存中进行计算。这样做的好处是可以充分利用共享内存的高带宽和低延迟，减少对全局内存的访问次数，例如，在如下的代码片段中：

```c
// Allocate shared memory for matrix tiles
__shared__ float Asub[32][32];
__shared__ float Bsub[32][32];
__shared__ float Csub[32][32];

// Load sub-matrix into shared memory
Asub[threadIdx.y][threadIdx.x] = A[row + threadIdx.y][k + threadIdx.x];
Bsub[threadIdx.y][threadIdx.x] = B[k + threadIdx.y][col + threadIdx.x];

// Perform computation
for (int t = 0; t < 32; ++t) {
    Cvalue += Asub[threadIdx.y][t] * Bsub[t][threadIdx.x];
}

// Store into Csub
Csub[threadIdx.y][threadIdx.x] = Cvalue;

// Store into C
C[row + threadIdx.y][col + threadIdx.x] = Csub[threadIdx.y][threadIdx.x];
```

这里的 `Asub` ,`Bsub` 和`Csub`是三个大小为 `32x32` 的共享内存块，一共会使用3072个float大小的shared memory，不难发现，当程序执行到`Csub[threadIdx.y][threadIdx.x] = Cvalue;`的时候，Asub和Bsub其实已经不会被使用到了，此时我们应该复用这部分存储，倘若如此，我们可以省下1024个float大小的shared memroy，相应的，我们可以探索更大的Tile Size或者Pipeline。而在常用的Tile Shape中，往往是BM~=BN >> BK的，这就导致C_shared往往很大，不复用存储会为硬件带来非常大的压力。

### MergeSharedMemoryAllocations 的分析和改进

首先，我们需要简要回顾一下这个Pass的修改历史，社区的大佬**[masahi](https://github.com/masahi)**在2021年的时候写了最原始的Pass，[CUDA\] Support multiple TIR-level dynamic shared memory allocations by masahi · Pull Request #8571 · apache/tvm (github.com)](https://github.com/apache/tvm/pull/8571) ，当时还没有活跃变量分析的内容，猜想只是因为dynamic shared memory只能声明一次，所以必须要把原本的多个alloc给整合成一个，年底的时候**[jinhongyii](https://github.com/jinhongyii)** 在这个Pass上增加了对各个Buffer的活跃变量分析，使得Buffer可以被复用，再这之后的一些更改大部分是打打补丁（例如针对一些TVM的buildin intrin，例如异步拷贝和TensorCore相关的指令)，去年的时候，我对这个Pass做了一个简单的改进，提高了一些场景下的复用率，并且将这个内容扩展到静态Shared Memory中去[CUDA\] Simple extend to optimize reuse for static shared memory. by LeiWang1999 · Pull Request #16342 · apache/tvm (github.com)](https://github.com/apache/tvm/pull/16342)，与此同时，这个Pass的名字也从`MergeDynamicSharedMemoryAllocations`变成了`MergeSharedMemoryAllocations `.（至于为什么不all in dynamic shared memory呢？其实作者当时是被ThreadSync这个Pass给坑了，在dynamic的时候莫名其妙多插了很多sync，导致笔者认为static在某些case下要更快，如今看来，这两者别无二致）。

讲过历史，

[ ] 一个Example

[ ] 这个Pass修改历史

MergeSharedMemoryAllocations的分析

[ ] 活跃变量分析

[ ] 打印出中间变量的结果

​	[ ] 穿插一些TIR的设计

[ ] 定位问题

[ ] 解决办法

