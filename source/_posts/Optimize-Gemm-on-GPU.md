---
title: Optimize Gemm on GPU
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2022-08-24 11:09:31
---

<!-- more -->

### 合适地选择Block Size和Grid Size

Block Size是指一个Block中的thread个数，增大block的数量一方面有利于提高程序的并行性，但是如果同一个block的thread之间存在线程的同步，则过大的block size会带来同步的overhead，导致SM利用率降低，而Grid Size是指Block的数量，如何好的
