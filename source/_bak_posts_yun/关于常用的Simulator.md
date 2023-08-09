---
title: 关于常用的Simulator
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2022-03-11 22:45:19
---

<!-- more -->

主要分为三个部分：

Architecture：加速器的体系根据其结构可分为三组，固定数据流的加速器(例如TPU、NVDLA、Eyeriss)、灵活数据流的加速器(例如Eyeriss v2、Maeri、Sigma)和chiplet多核加速器(例如SIMBA)。与包括CPU和GPU在内的传统体系结构不同，空间体系结构使用ScratchPad作为片上buffer。和传统的CPU体系结构里的Cache不一样，ScratchPad是可编程的，数据从DDR往ScracthPad上load的过程是由用户生成指令来操作，因为要取的数据地址相对来讲固定，所以理论上存在一个最有的数据流动序列，详见HPCA19上的《 Communication Lower Bound in Convolution Accelerators 》，而Cache是根据局部性原理由硬件完成这个操作，用户不可控。

Cost Models：也是因为Cache的原因，对加速器运行效率的建模不一定需要使用cycle级别的模拟器就可以。现在开源的就存在着不同的cost模型，用于以不同程度的保真度，为不同类型的加速器建模。例如，SCALE-sim来仿真脉动阵列(TPU)， MAESTRO仿真的阵列具有可配置的宽高比，Timeloop在模拟的时候可以考虑到复杂的内存层次结构建模，Tetris可以建模3D阵列。

Mappers：使用Cost Models，可以用目标硬件上的特定的map来估计程序的性能。然而，要为给定的工作负载和体系结构找到最佳映射并不简单，原因有两个。首先，映射的空间可能非常大，这使得穷举搜索变得不可行。这导致了几种映射器的开发，它们通过修剪搜索空间或用有效的方法搜索来减少搜索时间。Marvel提出了一种分离芯片外map空间和片上map空间的方法，timeloop利用了基于采样的搜索方法，Interstella使用了基于启发式的搜索方法，Mind Mapping开发了一个子模型来基于梯度搜索，而Gamma使用了基于遗传算法的方法通过利用先前的结果来有效地推进。其次，定义Map的搜索空间本身通常很复杂，因为不同的操作和不同的硬件加速器可能会对可行的映射施加约束。这就是为什么现在的Mapper在今天高度依赖于特定的成本模型，从而限制了可扩展性。
