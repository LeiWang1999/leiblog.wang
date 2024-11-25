---
title: High Performance AMD Matrix Core Codegen
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2024-11-25 15:40:06
---

不久之前的一篇分享里，我介绍了AMD CDNA架构(MI210, MI250, MI300)上的异步拷贝相关指令，在[BitBLAS](https://github.com/microsoft/BitBLAS)可以找到相关的实现，然而实际过程中发现AMD的异步拷贝指令的要求实际上要比那篇分享所写的更加苛刻，每个warp里的线程必须要求访问连续的数据，或者通过M0寄存器来控制每个线程的偏置。

一般来说，我们习惯这个指令就是明确的要load给定指针的一小块数据就行了，但是这个指令因为上述提到的两个限制就很难做到。经过笔者非常繁琐的Micro bencmark之后，笔者终于调教出了可以让每个线程Load给定数据块的写法，如下:

```cpp
template <bool pre_nop = false>
CK_TILE_DEVICE void async_buffer_load_dword_v(void* smem, int32x4_t rsrc, index_t voffset) {
  auto const lds_ptr_sgpr = __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(smem)));
  asm volatile(
      "s_mov_b32 m0, %0; \n\t"
      "buffer_load_dword %1, %2, 0 offen lds;\n\t" ::"s"(lds_ptr_sgpr),
      "v"(voffset), "s"(rsrc)
      : "memory");
}
if constexpr(N == 4) {
  async_buffer_load_dword_v(lds_base_ptr, make_wave_buffer_resource(((int32_t *)global_base_ptr) - threadIdx.x), threadIdx.x * N /*assume 4 bytes*/);
}
```

在这篇文章里，笔者填一下AMD Matrix Core的坑，介绍一下过去一个月里BitBLAS针对AMD的的高性能Matrix Core支持，在这篇文章里笔者将介绍一下MFMA（AMD版的MMA）。如何进行AMD Kernel的性能分析，及Profile一个AMD Kernel，最后我们介绍若干种绞尽了笔者脑汁的优化方法，完全利用好硬件的带宽(全都是128bits的内存访问指令，并且没有Memory bank conflict)。

这篇文章涉及到的算子有矩阵乘法和Flash Attention。本篇文章的实现在BitBLAS里, Codegen以及Swizzle等Layout变换依托于TVM, TVM可以帮助我们显式地操作一个数据的Layout，相比Triton更加灵活和可观。虽然AMD提供的文档十分有限，但是在这一个月里笔者参考了很多AMD开发人员提供的实现，例如[Composable Kernel](https://github.com/ROCm/composable_kernel)和[Triton for ROCm](https://github.com/ROCm/triton)，笔者从这些项目中收获良多。

<!-- more -->

## AMD Matrix Core

some sample reference:

1. [amd-lab-notes](https://github1s.com/amd/amd-lab-notes/blob/540726835443513845cd16e5c5517466e6fbef6c/matrix-cores/src/mfma_fp32_32x32x8fp16.cpp)


from llvm:

```cpp
amdgcn_mfma_f32_16x16x16bf16_1k,           // llvm.amdgcn.mfma.f32.16x16x16bf16.1k
amdgcn_mfma_f32_16x16x16f16,               // llvm.amdgcn.mfma.f32.16x16x16f16
amdgcn_mfma_f32_16x16x1f32,                // llvm.amdgcn.mfma.f32.16x16x1f32
amdgcn_mfma_f32_16x16x2bf16,               // llvm.amdgcn.mfma.f32.16x16x2bf16
amdgcn_mfma_f32_16x16x4bf16_1k,            // llvm.amdgcn.mfma.f32.16x16x4bf16.1k
amdgcn_mfma_f32_16x16x4f16,                // llvm.amdgcn.mfma.f32.16x16x4f16
amdgcn_mfma_f32_16x16x4f32,                // llvm.amdgcn.mfma.f32.16x16x4f32
amdgcn_mfma_f32_16x16x8bf16,               // llvm.amdgcn.mfma.f32.16x16x8bf16
amdgcn_mfma_f32_32x32x1f32,                // llvm.amdgcn.mfma.f32.32x32x1f32
amdgcn_mfma_f32_32x32x2bf16,               // llvm.amdgcn.mfma.f32.32x32x2bf16
amdgcn_mfma_f32_32x32x2f32,                // llvm.amdgcn.mfma.f32.32x32x2f32
amdgcn_mfma_f32_32x32x4bf16,               // llvm.amdgcn.mfma.f32.32x32x4bf16
amdgcn_mfma_f32_32x32x4bf16_1k,            // llvm.amdgcn.mfma.f32.32x32x4bf16.1k
amdgcn_mfma_f32_32x32x4f16,                // llvm.amdgcn.mfma.f32.32x32x4f16
amdgcn_mfma_f32_32x32x8bf16_1k,            // llvm.amdgcn.mfma.f32.32x32x8bf16.1k
amdgcn_mfma_f32_32x32x8f16,                // llvm.amdgcn.mfma.f32.32x32x8f16
amdgcn_mfma_f32_4x4x1f32,                  // llvm.amdgcn.mfma.f32.4x4x1f32
amdgcn_mfma_f32_4x4x2bf16,                 // llvm.amdgcn.mfma.f32.4x4x2bf16
amdgcn_mfma_f32_4x4x4bf16_1k,              // llvm.amdgcn.mfma.f32.4x4x4bf16.1k
amdgcn_mfma_f32_4x4x4f16,                  // llvm.amdgcn.mfma.f32.4x4x4f16
amdgcn_mfma_f64_16x16x4f64,                // llvm.amdgcn.mfma.f64.16x16x4f64
amdgcn_mfma_f64_4x4x4f64,                  // llvm.amdgcn.mfma.f64.4x4x4f64
amdgcn_mfma_i32_16x16x16i8,                // llvm.amdgcn.mfma.i32.16x16x16i8
amdgcn_mfma_i32_16x16x4i8,                 // llvm.amdgcn.mfma.i32.16x16x4i8
amdgcn_mfma_i32_32x32x4i8,                 // llvm.amdgcn.mfma.i32.32x32x4i8
amdgcn_mfma_i32_32x32x8i8,                 // llvm.amdgcn.mfma.i32.32x32x8i8
amdgcn_mfma_i32_4x4x4i8,                   // llvm.amdgcn.mfma.i32.4x4x4i8
```

或者AMD开源出了一个小工具[amd_matrix_instruction_calculator](https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator)可以帮助我们计算获得给定指令的相关信息，包括每个线程所获取数据的索引计算，以及每个位置的对应的线程索引计算等等，非常之方便。

```bash
git clone https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator
cd amd_matrix_instruction_calculator
./matrix_calculator.py --architecture cdna1 --instruction v_mfma_f32_32x32x8f16 --detail-instruction

# Output:
# Architecture: CDNA1
# Instruction: V_MFMA_F32_32X32X8F16
#     Encoding: VOP3P-MAI
#     VOP3P Opcode: 0x4c
#     VOP3P-MAI Opcode: 0xc
#     Matrix Dimensions:
#         M: 32
#         N: 32
#         K: 8
#         blocks: 1
#     Execution statistics:
#         FLOPs: 16384
#         Execution cycles: 64
#         FLOPs/CU/cycle: 1024
#         Can co-execute with VALU: True
#         VALU co-execution cycles possible: 56
#     Register usage:
#         GPRs required for A: 2
#         GPRs required for B: 2
#         GPRs required for C: 16
#         GPRs required for D: 16
#         GPR alignment requirement: 4 bytes
#     VOP3P-MAI register encoding:
#         A matrix source field: Src0
#         B matrix source field: Src1
#         C matrix source field: Src2
#         D matrix source field: Vdst
#     Register capabilities:
#         A matrix can use ArchVGPRs: True
#         A matrix can use AccVGPRs: True
#         B matrix can use ArchVGPRs: True
#         B matrix can use AccVGPRs: True
#         C and D matrix can use ArchVGPRs: False
#         C and D matrix can use AccVGPRs: True
#     Register modifiers:
#         CBSZ and ABID bits supported: False
#         BLGP bits supported: True
#     Matrix element to register mapping with no modifiers:
#         A[i][k].block GPR: (floor(k / 2) % 2).[16*(k % 2)+15 : 16*(k % 2)]
#         A[i][k].block Lane: 32 * floor(k / 4) + i
#         B[k][j].block GPR: (floor(k / 2) % 2).[16*(k % 2)+15 : 16*(k % 2)]
#         B[k][j].block Lane: 32 * floor(k / 4) + j
#         C or D[i][j].block GPR: 4 * floor(i / 8) + (i % 4)
#         C or D[i][j].block Lane: (32 * floor(i / 4)) % 64 + j
#     Register to matrix element mapping with no modifiers:
#         A i: (lane % 32)
#         A k: 4 * floor(lane / 32) + 2 * GPR_num + floor(GPR_bits / 16)
#         A block: 0
#         B j: (lane % 32)
#         B k: 4 * floor(lane / 32) + 2 * GPR_num + floor(GPR_bits / 16)
#         B block: 0
#         C or D i: (8 * floor(GPR_num / 4) % 32) + 4 * floor(lane / 32) + (GPR_num % 4)
#         C or D j: (lane % 32)
#         C or D block: 0
```

## AMD Kernel性能分析

AMD没有像Nsight System和Nsight Compute那样强大的profile工具。他提供了`rocprof`，在一年之前我还对这个工具比较熟悉，现在已经基本忘了咋用了，基于`rocprof`, AMD还有一个图形化(以webui的形式提供)工具，`omniperf`，目前amd上的性能分析主要依靠该工具。

## Memory Layout 优化

## Flash Attention


```

```