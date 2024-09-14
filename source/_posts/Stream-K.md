---
title: 浅分析一下Marlin与Stream-K
categories:
  - Technical
tags:
  - CUDA Programing
date: 2024-08-26 15:22:56
---

在优化BitBLAS的性能的时候，作者发现在小批次的一些矩阵乘法（主要是30B以下的大语言模型的线性层尺寸）上，FP16xINT4的性能比Marlin要差，但是和当时测试的TensorRT-LLM里使用的CUTLASS WeightOnlyDequantize的性能是类似的。为了满足一下好奇心，这里对比了一下BitBLAS与Marlin的一些实现，最后的结论是相比于Marlin，缺陷在于没有使用Stream-K。

<!-- more -->

## 1. 浅谈一下BitBLAS

## 2. Marlin的实现分析

## 3. 充分消除Wavefronts浪费的Stream-K

