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

本文假设读者对Nvidia GPU的编程有一定的了解，熟悉最基本的Tile优化程序的方法，以及Tensor Core的基本概念。

<!-- more -->

## AMD Matrix Core

AMD Matrix Core 是AMD GPU上对标NVIDIA Tensor Core的一类指令，相关的介绍可以在[AMD的官方文档](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html)找到一些编程的例子，另外仓库[amd-lab-notes](https://github1s.com/amd/amd-lab-notes/blob/540726835443513845cd16e5c5517466e6fbef6c/matrix-cores/src/mfma_fp32_32x32x8fp16.cpp)提供了一系列常用的Matrix Core指令的使用案例，对于Layout相关的部分还是比较清楚的。

这里笔者只总结AMD Matrix Core区别于Nvidia Tensor Core的一些主要特点:

1. 从暴露给用户的接口上来看，Matrix Core提供了最底层的MFMA指令，其可以类比于Nvidia GPU的MMA指令。在HIP(AMD版的CUDA)里，官方提供了rocwmma这一库，一般而言用户将代码中的命名空间`nvwmma::`替换成`rocwmma::`即可无缝使用，虽然它支持的shape不如Nvidia丰富。
2. AMD Matrix Core并没有提供给用户一个类似于Nvidia Tensor Core使用的`ldmatrix`和`stmatrix`指令，这让一些访存优化变得更加困难（当然本文会给出我们的解决办法。
3. AMD CDNA架构的Warp Size是64，及64个现成一起来协作完成一块矩阵的计算，包括warp shuffle这类指令都是以64个线程为单位，这和Nvidia的32个线程的Warp Size是不一样的。这里顺带吐槽一下AMD的端侧卡，例如RTX 7900, 属于另一类RDNA架构，其Warp Size是32，虽然和Nvidia的Warp Size一样，但是其Matrix Core指令也对用户很不友好，反而比CDNA更加难以优化。

如何知道AMD Matrix Core有哪些指令可以使用呢？这里官方貌似么有提供一个可用的指令集合，笔者是通过LLVM的源代码找到可用的组合，例如:

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

另外插一嘴，从这些指令中可以看到，AMD的Matrix Core并不支持FP16的Accum，这也是和Nvidia Tensor Core的区别之一。

接下来要涉及到另一个问题，如何确定每个线程应该Hold的数据位置？在Nvidia里，Tensor Core的Layout一般通过Nvidia的PTX文档意会。而AMD开源出了一个小工具[amd_matrix_instruction_calculator](https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator)帮助用户获取计算获得给定指令的相关信息，包括每个线程所获取数据的索引计算，以及每个位置的对应的线程索引计算等等，非常之方便。

```bash
git clone https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator
cd amd_matrix_instruction_calculator
./matrix_calculator.py --architecture cdna1 --instruction v_mfma_f32_16x16x16f16 --detail-instruction

# Output:
# Architecture: CDNA1
# Instruction: V_MFMA_F32_16X16X16F16
#     Encoding: VOP3P-MAI
#     VOP3P Opcode: 0x4d
#     VOP3P-MAI Opcode: 0xd
#     Matrix Dimensions:
#         M: 16
#         N: 16
#         K: 16
#         blocks: 1
#     Execution statistics:
#         FLOPs: 8192
#         Execution cycles: 32
#         FLOPs/CU/cycle: 1024
#         Can co-execute with VALU: True
#         VALU co-execution cycles possible: 24
#     Register usage:
#         GPRs required for A: 2
#         GPRs required for B: 2
#         GPRs required for C: 4
#         GPRs required for D: 4
#         GPR alignment requirement: 4 bytes
#     VOP3P-MAI register encoding:
#         A matrix source field: Src0
#         B matrix source field: Src1
#         C matrix source field: Src2
#         D matrix source field: Vdst
#     Register data types:
#         Src0: FP16 (IEEE binary16 floating point)
#         Src1: FP16 (IEEE binary16 floating point)
#         Src2: FP32 (IEEE binary32 floating point)
#         Vdst: FP32 (IEEE binary32 floating point)
#     Register capabilities:
#         A matrix can use ArchVGPRs: True
#         A matrix can use AccVGPRs: True
#         B matrix can use ArchVGPRs: True
#         B matrix can use AccVGPRs: True
#         C and D matrix can use ArchVGPRs: False
#         C and D matrix can use AccVGPRs: True
#     Register modifiers:
#         Sparse A matrix: False
#         CBSZ and ABID bits supported: False
#         BLGP bits supported: True
#     Matrix element to register mapping with no modifiers:
#         A[i][k].block GPR: (floor(k / 2) % 2).[16*(k % 2)+15 : 16*(k % 2)]
#         A[i][k].block Lane: 16 * floor(k / 4) + i
#         B[k][j].block GPR: (floor(k / 2) % 2).[16*(k % 2)+15 : 16*(k % 2)]
#         B[k][j].block Lane: 16 * floor(k / 4) + j
#         C or D[i][j].block GPR: (i % 4)
#         C or D[i][j].block Lane: 16 * floor(i / 4) + j
#     Register to matrix element mapping with no modifiers:
#         A i: (lane % 16)
#         A k: 4 * floor(lane / 16) + 2 * GPR_num + floor(GPR_bits / 16)
#         A block: 0
#         B j: (lane % 16)
#         B k: 4 * floor(lane / 16) + 2 * GPR_num + floor(GPR_bits / 16)
#         B block: 0
#         C or D i: 4 * floor(lane / 16) + (GPR_num % 4)
#         C or D j: (lane % 16)
#         C or D block: 0
```

根据其给出的`Matrix element to register mapping`, 以及`Register to matrix element mapping`信息，我们可以得到每个线程应该Hold的数据位置，以及每个位置对应的线程索引，例如针对`v_mfma_f32_16x16x16f16`指令:

![](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/202411252342916.png)

根据这些信息，我们就可以得到一个简单正确的利用MFMA的Hip Kernel了，Kernel长下面这个样子:

```cpp
extern "C" __global__ void __launch_bounds__(256) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, float* __restrict__ C) {
  float C_local[16];
  __shared__ half_t A_shared[2048];
  __shared__ half_t B_shared[2048];
  half_t A_local[8];
  half_t B_local[8];
  const dim3 blockIdx = tl::rasterization2DRow<10>();
  for (int i = 0; i < 4; ++i) {
    *(float4*)(C_local + (i * 4)) = make_float4(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f);
  }
  for (int ko = 0; ko < 8; ++ko) {
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      *(uint2*)(A_shared + ((i_1 * 1024) + (((int)threadIdx.x) * 4))) = *(uint2*)(A + (((((((int)blockIdx.y) * 16384) + (i_1 * 8192)) + ((((int)threadIdx.x) >> 3) * 256)) + (ko * 32)) + ((((int)threadIdx.x) & 7) * 4)));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 2; ++i_2) {
      *(uint2*)(B_shared + ((i_2 * 1024) + (((int)threadIdx.x) * 4))) = *(uint2*)(B + (((((((int)blockIdx.x) * 16384) + (i_2 * 8192)) + ((((int)threadIdx.x) >> 3) * 256)) + (ko * 32)) + ((((int)threadIdx.x) & 7) * 4)));
    }
    __syncthreads();
    for (int ki = 0; ki < 2; ++ki) {
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        *(uint2*)(A_local + (i_3 * 4)) = *(uint2*)(A_shared + (((((((((int)threadIdx.x) & 127) >> 6) * 1024) + (i_3 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (ki * 16)) + (((((int)threadIdx.x) & 63) >> 4) * 4)));
      }
      for (int j = 0; j < 2; ++j) {
        *(uint2*)(B_local + (j * 4)) = *(uint2*)(B_shared + ((((((((int)threadIdx.x) >> 7) * 1024) + (j * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (ki * 16)) + (((((int)threadIdx.x) & 63) >> 4) * 4)));
      }
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        for (int j_1 = 0; j_1 < 2; ++j_1) {
          {
    *(((float32x4*)C_local) + ((i_4 * 2) + j_1)) = __builtin_amdgcn_mfma_f32_16x16x16f16(*(((float16x4*)A_local) + i_4),
                  *(((float16x4*)B_local) + j_1),
                  *(((float32x4*)C_local) + ((i_4 * 2) + j_1)), 0, 0, 0);
  };
        }
      }
    }
  }
  for (int i_5 = 0; i_5 < 2; ++i_5) {
    for (int j_2 = 0; j_2 < 2; ++j_2) {
      for (int local_id = 0; local_id < 4; ++local_id) {
        C[(((((((((((int)blockIdx.y) * 16384) + (((((int)threadIdx.x) & 127) >> 6) * 8192)) + (i_5 * 4096)) + ((((int)threadIdx.x) & 15) * 256)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 7) * 32)) + (j_2 * 16)) + (((((int)threadIdx.x) & 63) >> 4) * 4)) + local_id)] = C_local[(((i_5 * 8) + (j_2 * 4)) + local_id)];
      }
    }
  }
}
```

但是从图中不难发现，默认的layout非常不友好，主要体现在两个方面：
  - 其中，B矩阵和C矩阵的每个线程期望访问的数据之间存在着很大Stride，没有连续的数据访问。
  - 即使A矩阵的访问是连续的4个float16, 这也不是最优的，因为访问数据的指令最大支持128bits(8xfp16),这和Nvidia是一样的。

为了验证有挂性能的猜想，最直观的方法就是利用厂商提供的Profiler工具，为此我们引入下一章，及AMD的Kernel性能分析。

## AMD Kernel性能分析

AMD没有像Nsight System和Nsight Compute那样强大的profile工具。他提供了`rocprof`，在一年之前我还对这个工具比较熟悉，现在已经基本忘了咋用了，基于`rocprof`, AMD还有一个图形化(以webui的形式提供)工具，`omniperf`，目前amd上的性能分析主要依靠该工具。

## Memory Layout 优化

## Flash Attention


```

```