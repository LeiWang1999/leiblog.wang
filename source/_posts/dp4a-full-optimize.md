---
title: dp4a full optimize
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2022-09-24 15:08:49
---

w关于IGEMM在CUDA Core上的优化，网上基本没有看到开源的实现(甚至连实现不好的版本都没有，可能大家都是直接用的cublas/cutlass来实现吧，这里是我通过手写CUDA代码的方式使用DP4A指令达到sota的性能，不得不说这个东西没有人实现也是有原因的，坑是真的多。

我们以M=K=N 16384的矩阵乘举例。

<!-- more -->
