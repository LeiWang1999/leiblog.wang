---
title: gptq-tvm
categories:
  - Technical
tags:
  - MLSys
date: 2023-05-02 13:59:08
---

最近有一些针对大模型的量化工作可以在保证一定精度的情况下把模型的weight量化到4bit, 3bit甚至是2bit，从而将模型的权重压缩数倍使得直接on-device地去做大语言模型的推理有了可能性，然而，在一般的推理任务中我们习惯于调用的一些library，比如cublas，cudnn等这些都没有提供这种低比特运算的kernel(不过不久前cutlass刚pr了fp16xint4, fp16xint8的kernel，但这仍然不足以面对丰富的需求)，导致实际去做这件事情不是这么容易。

于是前段时间就想着能不能通过tvm auto codegen出这些kernel，当然最近tvm community推的MLC-LLM其实是一个好的解决方案，不过在MLC-LLM发布之前的一段时间我也进行过一些简单的探索，现在可能也没什么用处了，这篇文件简单分享出来一个流程，包括怎么做模型的量化，怎么general出in-kernel compress, decompress的代码生成，怎么adhoc的解决dynamic shape，教大家怎么在单卡3090上完成一个OPT-66B的模型经过量化，用tvm生成kernel，最后单卡部署3bit的模型，如果能给大家带来一些insight就再好不过了。

这个项目放在：

[LeiWang1999/AutoGPTQ.tvm](https://github.com/LeiWang1999/GPTQ-tvm)

完全只依赖tvm的上游分支，不需要hack code去做额外的添加对动态shape和量化的支持。

<!-- more -->

## GPTQ - 量化曙光

> GPTQ并不是凭空出现的， 它的原理来自于另一个量化方法OBQ，GPTQ可以说是它的加速版。而OBQ实际上是对OBS(一种比较经典的剪枝方法）的魔改， 而OBS则来自于OBD(一种由LeCun在1990年提出的剪枝方法）。可以说是历史悠久，历久弥新了（笑） 
>
> [GPTQ: 模型量化，穷鬼救星 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/616969812)

GPTQ的项目地址是[GPTQ]([Check failed: type_code_ == kTVMObjectHandle (0 vs. 8) : expected Object but got int - Apache TVM Discuss (github.com)](https://github.com/IST-DASLab/gptq)), 从系统的design上，我们并不需要关心他的算法是如何实现的，只需要知道他的使用方式简单且有效，相比于其他的量化框架，比如商汤的PPQ，GPTQ的优点是基于pytorch，可以offload每一层进行quant，意味着你可以在一张很廉价的GPU卡上去实现模型的量化。而PPQ这种为了做到更具通用性（大概，通用在了接入tensorrt和onnxruntime）导致他们选取onnx作为一个输入和输出的格式，限制了在大模型(大概大到单张卡放不下的程度)的应用，因为onnx在模型的参数量达到很大的时候无论是导入还是导出都是一个巨大的问题，下图是GPTQ贴出来在LLAMA上的精度情况:

![image-20230505235740216](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20230505235740216.png)

基本可以说他是在3bit/4bit上的sota，虽然可能在4bit上不比直接取min max再rescale好不了多少。

这里我们解释一下经过GPTQ量化的权重是以一种什么样的形式存在，以什么样的形式被压缩，以3bit为例，需要明确的是，在GPTQ的量化过程中，只有Decode layer里的q,k,v,out,fc1,fc2这六个层是被quant的，他们也是网络主要的参数组成，所以实际上我们只需要提供高效的矩阵乘kernel就行了。

对于原先的float16的权重，量化完成之后会得到scale和zero，用来计算出int3的值，我们以group_size=-1为例，及一列数据共用一个scale和zero的值:

```python
self.zeros = zeros * scales
self.scales = scales.clone()
intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
```

通过上述表达式，我们可以还原出int3的值，在这里先会使用一个int32的矩阵(大小是NxK)进行存储，当然每个int32的数据只有最低3bit是有值的，为了完成内存的节约，我们要按照bit存储，及把这个矩阵的大小变成(NxK/8x3)个int32大小的矩阵，具体来讲，每32个连续的int3的数据需要存储在3个int32的数据中，在计算的时候我们在kernel内部对其进行解码，具体的压缩算法如下:

```python
qweight = np.zeros(
        (intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32
)
i = 0
row = 0
while row < qweight.shape[0]:
    for j in range(i, i + 10):
        qweight[row] |= intweight[j] << (3 * (j - i))
    i += 10
    qweight[row] |= intweight[i] << 30
    row += 1
    qweight[row] |= (intweight[i] >> 2) & 1
    i += 1
    for j in range(i, i + 10):
        qweight[row] |= intweight[j] << (3 * (j - i) + 1)
    i += 10
    qweight[row] |= intweight[i] << 31
    row += 1
    qweight[row] |= (intweight[i] >> 1) & 0x3
    i += 1
    for j in range(i, i + 10):
        qweight[row] |= intweight[j] << (3 * (j - i) + 2)
    i += 10
    row += 1
```

![image-20230506001134986](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20230506001134986.png)

一个有趣的发现是，量化带来的收益不仅仅是可以完全on-chip运行一个大模型，从而减少offload到ddr, disk的开销，在kernel上也是有收益的，因为在文本生成的生成过程矩阵乘实际上是一个vecxmat的问题，这个问题在GPU上的实现一般是memory bound的，减少存储的bit意味着减少了内存的交互，从而可以获得相当大的性能收益，GPTQ为了证明这一点，提供了一个fp32xint3的cuda kernel实现，效果不错，在kernel里，他这样对权重做解压缩:

```C++
while (k < BLOCKWIDTH) {
    tmp1 = as_unsigned(mat[i]);
    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    /.../
    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;
    res += (scale * scalar_t((tmp2 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp2 >>  3) & 0x7) - zero) * blockvec[k + 1];
    /.../
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;
    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    /.../
    i += width;
    k += 10;
  }
```

显然这样的设计对kernel是不太友好的，GPTQ之后，又有很多有意思的扩展，比如GPTQ-triton，通过triton来提供了fp32/fp16xint4/int8的kernel，但是没有提供int3，更进一步的auto-gptq, 把gptq包装的更易用，也支持了triton的kernel，所以为什么不能有一个GPTQ-tvm呢？

## In Kernel Decompress

这一个小节，分享一下我实现的针对任意bit数据的计算的解压缩方法，虽然tvm并不支持int3,int4比特的数据类型，但我们实际上没必要去给tvm加上这些支持，为了恢复出原有的数据我们实际上只需要支持一些按位的逻辑操作就可以了，而Tensor IR刚好可以做到这一点。

<img src="https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20230506003431698.png" alt="image-20230506003431698" style="zoom: 25%;" />

我们以8bit的存储为例，实际上我们的输入是一个[N, K // 8 * bit]的int8矩阵，这样：

32’s 3bits 可以被存在 12’s int8 (int3)

32’s 4bits 可以被存在  16’s int8 (int4)

32’s 5bits 可以被存在  20’s int8 (int4 + int)

对于4bit来说这个解压的过程不复杂，对于3, 5 bit这种我们需要做跨Bytes的访问，先通过最简单的情况公式化这个过程:

假设我们压缩至bit位，存储的数据类型是 `B[N, K/8 * bit]: int8`, 解压缩的数据类型是`D[N, K]: float16`, 考虑32个elems作为一个组，我们可以计算出他们的group id、组里的位置和mask，以及group stride。

```python
mask = (1 << bit) - 1
group_index = vj // 32
inner_index = vj % 32
group_stride = 32 * bit // 8
```

接下来，通过这些值计算出一个comressed_index(index in B) 和offset。

```python
compressed_index = group_index  * group_stride +  inner_index * bit // 8
offset = ((vj % 32) * bit) % 8
```

> 如果我们不需要跨bytes访问数据，现在就可以恢复出原始的值，如:
>
> ```
> D[vi, vj] = (B[vi, compressed_index] >> offset & mask).astype("float16")
> ```

如果考虑到跨Bytes的访问，我们需要知道在什么情况下需要bytes访问，这个情况是:

```python
need_cross:bool = offset <= (8-bit)
```

如果偏移量≤(8位)，则不需要跨bytes访问数据，如果不是的话还需要一些进一步的信息，如这个数据在第一个字节中占了几位，在第二个字节中占了几位，以及他们各自的掩码。很明显，第一个字节应该包含(8-offset)位，下一个字节包含 (bit - (8-offset))位，这两部分的掩模计算部分也是独立的。对于第一个字节，该数字存储最低几位，掩码应该是0x001或0x011，对于下一个字节，该数字存储最高字节。掩码应该是0x100或0x110，分别对应于不同的规则，抽象出来就是:

```python
# for the first byte, we can compute the mask as
first_byte_mask = (1 << (8-offset)) - 1
# for the next byte, we can compute the mask as
next_byte_mask = ((0XFF << bit) >> (bit - (8 - offset))) & ((1 << (bit)) - 1)
```

当然还得加上rescale来恢复精度，最后的tir表达式可就变成了:

```python
@tvm.script.ir_module
class DecompressGemm:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle, scales: T.handle, zeros: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype="float16")
        B = T.match_buffer(b, [N, K // 8 * bit], dtype="int8")
        C = T.match_buffer(c, [M, N], dtype="float16")
        Scales = T.match_buffer(scales, [N], dtype="float16")
        Zeros = T.match_buffer(zeros, [N], dtype="float16")

        B_decompress = T.alloc_buffer([N, K], dtype="float16")
        B_rescale = T.alloc_buffer([N, K], dtype="float16")

        for i, j in T.grid(N, K):
            with T.block("B_decompress"):
                vi, vj = T.axis.remap("SS", [i, j])
                B_decompress[vi, vj] = T.Select(((vj % 32) * bit) % 8 <= 5, ((B[vi, (vj // 32) * group_stride + (vj % 32) * bit // 8] >> (((vj % 32) * bit) % 8) & mask)).astype("float16"), (((B[vi, (vj // 32) * group_stride + (vj % 32) * bit // 8] >> ((vj % 32) * bit) % 8) & (
                    1 << (8 - ((vj % 32) * bit) % 8)) - 1).astype("int8") | ((B[vi, (vj // 32) * group_stride + (vj % 32) * bit // 8 + 1] << (8 - ((vj % 32) * bit) % 8)) & (mask << (8 - ((vj % 32) * bit) % 8)) & mask).astype("int8")).astype("float16"))

        for i, j in T.grid(N, K):
            with T.block("B_rescale"):
                vi, vj = T.axis.remap("SS", [i, j])
                B_rescale[vi, vj] = B_decompress[vi, vj] * \
                    Scales[vi].astype('float16') - Zeros[vi].astype('float16')

        for i, j, k in T.grid(M, N, K):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float16(0)
                C[vi, vj] = C[vi, vj] + \
                    A[vi, vk].astype("float16") * \
                    B_rescale[vj, vk].astype("float16")
```

## 乞丐版 Dynamic Shape

想要做到实际使用还有一个非常重要的问题要求解，及dynamic shape。因为用户输入的promot的长度是不固定的，所以在实际运算中m的长度并不是一个定值，这就要做到dynamic shape的支持，现有的dynamic shape的解决方案基本都是和对标cublas，针对不同的shape config tune好几个tile的kernel，根据不一样的矩阵大小选择不一样的kernel，而tvm上游，包括很多自动codegen的框架来说，生成的都是静态kernel，以GPU上的矩阵乘的Tile思路为例：

<img src="https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20230506152240446.png" alt="image-20230506152240446" style="zoom:50%;" />

线程通过矩阵分块的形式来完成一个矩阵乘法的计算，一个block算一个BMxBN大小的矩阵，BM,BN,BK的大小就是我们提到的config，profile cublas的kernel的时候，我们经常能够观察到`ampere_h16816gemm_256x128_ldg8_stages_32x3_nn`这样的名字，其实就已经包含了这个kernel的信息，ampere是kernel的provider，这里是ampere架构,h16816是使用到的tensorcore指令位m16n8k16，256x128是BMxBN的值,ldg8是向量化的数量(halfx8->float4), 32是BK的值,3是software pipeline的stage数, nn是A和B的layout。对应到tvm的矩阵乘，也是以同样的方式存在的，如下图这一个kernel：

```C++
__global__ void __launch_bounds__(128) tir_halfxint3_tensorop_128x256x32x1_t0_y2z2_K36864_align8(half *__restrict__ A, signed char *__restrict__ B, half *__restrict__ Scales, half *__restrict__ Zeros, half *__restrict__ C)
{
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> C_wmma_accumulator[32];
    __shared__ half A_shared[5120];
    __shared__ half B_rescale_shared[10240];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> A_shared_wmma_matrix_a[4];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> B_rescale_shared_wmma_matrix_b[8];
    for (int i_0_2_init = 0; i_0_2_init < 4; ++i_0_2_init)
    {
        for (int j_0_2_init = 0; j_0_2_init < 8; ++j_0_2_init)
        {
            nvcuda::wmma::fill_fragment(C_wmma_accumulator[((i_0_2_init * 8) + j_0_2_init)], 0.000000e+00f);
        }
    }
    for (int k_0_0 = 0; k_0_0 < 1152; ++k_0_0)
    {
        __syncthreads();
        for (int ax0_ax1_fused_2 = 0; ax0_ax1_fused_2 < 4; ++ax0_ax1_fused_2)
        {
            *(uint4 *)(A_shared + (((((((int)threadIdx.y) * 2560) + (((int)threadIdx.z) * 1280)) + (ax0_ax1_fused_2 * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8))) = *(uint4 *)(A + (((((((((int)blockIdx.y) * 4718592) + (((int)threadIdx.y) * 2359296)) + (((int)threadIdx.z) * 1179648)) + (ax0_ax1_fused_2 * 294912)) + ((((int)threadIdx.x) >> 2) * 36864)) + (k_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)));
        }
        for (int ax0_ax1_fused_2_1 = 0; ax0_ax1_fused_2_1 < 64; ++ax0_ax1_fused_2_1)
        {
            B_rescale_shared[((((((int)threadIdx.y) * 5120) + (((int)threadIdx.z) * 2560)) + (ax0_ax1_fused_2_1 * 40)) + ((int)threadIdx.x))] = ((((((((int)threadIdx.x) * 3) & 7) <= 5) ? ((half)((((int)B[((((((((int)blockIdx.x) * 3538944) + (((int)threadIdx.y) * 1769472)) + (((int)threadIdx.z) * 884736)) + (ax0_ax1_fused_2_1 * 13824)) + (k_0_0 * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & 7)) : ((half)(((signed char)((((int)B[((((((((int)blockIdx.x) * 3538944) + (((int)threadIdx.y) * 1769472)) + (((int)threadIdx.z) * 884736)) + (ax0_ax1_fused_2_1 * 13824)) + (k_0_0 * 12)) + ((((int)threadIdx.x) * 3) >> 3))]) >> ((((int)threadIdx.x) * 3) & 7)) & ((1 << (8 - ((((int)threadIdx.x) * 3) & 7))) - 1))) | ((signed char)(((((int)B[(((((((((int)blockIdx.x) * 3538944) + (((int)threadIdx.y) * 1769472)) + (((int)threadIdx.z) * 884736)) + (ax0_ax1_fused_2_1 * 13824)) + (k_0_0 * 12)) + ((((int)threadIdx.x) * 3) >> 3)) + 1)]) << (8 - ((((int)threadIdx.x) * 3) & 7))) & (7 << (8 - ((((int)threadIdx.x) * 3) & 7)))) & 7))))) * Scales[((((((int)blockIdx.x) * 256) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_1)]) - Zeros[((((((int)blockIdx.x) * 256) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.z) * 64)) + ax0_ax1_fused_2_1)]);
        }
        __syncthreads();
        for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1)
        {
            for (int ax0_0 = 0; ax0_0 < 4; ++ax0_0)
            {
                nvcuda::wmma::load_matrix_sync(A_shared_wmma_matrix_a[ax0_0], (&(A_shared[(((((int)threadIdx.y) * 2560) + (ax0_0 * 640)) + (k_0_1 * 16))])), 40);
            }
            for (int ax0_0_1 = 0; ax0_0_1 < 8; ++ax0_0_1)
            {
                nvcuda::wmma::load_matrix_sync(B_rescale_shared_wmma_matrix_b[ax0_0_1], (&(B_rescale_shared[(((((int)threadIdx.z) * 5120) + (ax0_0_1 * 640)) + (k_0_1 * 16))])), 40);
            }
            for (int i_0_2 = 0; i_0_2 < 4; ++i_0_2)
            {
                for (int j_0_2 = 0; j_0_2 < 8; ++j_0_2)
                {
                    nvcuda::wmma::mma_sync(C_wmma_accumulator[((i_0_2 * 8) + j_0_2)], A_shared_wmma_matrix_a[i_0_2], B_rescale_shared_wmma_matrix_b[j_0_2], C_wmma_accumulator[((i_0_2 * 8) + j_0_2)]);
                }
            }
        }
    }
    for (int ax0_0_2 = 0; ax0_0_2 < 4; ++ax0_0_2)
    {
        for (int ax1_0 = 0; ax1_0 < 8; ++ax1_0)
        {
            nvcuda::wmma::store_matrix_sync((&(C[((((((((int)blockIdx.y) * 1179648) + (((int)threadIdx.y) * 589824)) + (ax0_0_2 * 147456)) + (((int)blockIdx.x) * 256)) + (((int)threadIdx.z) * 128)) + (ax1_0 * 16))])), C_wmma_accumulator[((ax0_0_2 * 8) + ax1_0)], 9216, nvcuda::wmma::mem_row_major);
        }
    }
}

```

是不是直接用这个kernel，开不一样的block数就可以实现dynamic shape了呢？显然不是。动态shape下，这个kernel会有一些问题，如`for (int k_0_0 = 0; k_0_0 < 1152; ++k_0_0)` 这里的1152实际上是 K / BK，根据K的不同，这个值是不一样的，根据N的不同，数据的leading dimension，及每一行的stride也是不一样的，实际上我们观察到一个model的k和n都是固定的值，只有m是在变化的，针对每个k和n都tune一个kernel(一个model也就三四个shape就行了)，然后对m做pad，就可以解这个问题了，当然 kernel也得tune出很多个，需要针对不同的m去调用不同的kernel，下面是一个端到端的示例。

```python
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


pretrained_model_dir = "facebook/opt-66b"
quantized_model_dir = "opt-66b-3bit"


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=3,  # quantize model to 3-bit
    # desc_act=False,  # disable activation description
    # group_size=128,  # disable group quantization
    desc_act=True
)

# load un-quantized model, the model will always be force loaded into cpu
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask" 
# with value under torch.LongTensor type.
model.quantize(examples, use_tvm=True)

# save quantized model
model.save_quantized(quantized_model_dir)

# load quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_tvm=True)

# or you can also use pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("auto-gptq is")[0]["generated_text"])

'''
output is: auto-gptq is auto-gptq is a easy way to use tool command freeto use, easy to use
'''

```

## 总结

1. gptq-tvm和gptq-triton的后端有什么区别?

   虽然我习惯用tvm做kernel codegen，但是不得不说triton在这种场景下很方便，对不了解mlc的用户更友好，你只需要像写一个python function一样就可以获得不错的性能；反观TVM的DSL，需要自己写schedule，tune起来也比较麻烦，tune完了之后你获得的还不是一个动态的shape，还得自己在外面包一层。当然tvm也有很多优点，triton走mlir，优化看自己也更底层，tvm走codegen，把部分优化交给了nvcc, llvm，这些编译器的优化在某些情况是很玄学的，但好处是可以更好地支持各种设备和场景，triton换个AMD的GPU都跑不起来了，更别说mlc-llm放出来的一些demo。

2. 基于GPTQ的这种解法对于fuse来说并不好，因为基于pytorch来做很难做compile，比如做一些图上的优化，做算子的fuse，对于LLM来说这部分的收益还是相当大的吗，但很多mlc因为没有像tvm这样社区还算活跃，前端基本都是吃的onnx，这让在跑大的llm和quant的llm上会碰到问题，pytorch 2.0的compile接口出现可能会缓解这一问题(triton和tvm都有接入)。

总的来说，这是个大概没什么卵用的玩具，赶紧上车unity和mlc-llm :|
