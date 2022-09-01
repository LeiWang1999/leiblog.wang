---
title: a hgemm tvm schedule
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2022-08-31 12:17:47
---

现在想要用TVM的Tensor Expression复现cutlass的高效GEMM实现，希望能够打到和CUTLASS相当的性能，我们选取2的14次方的正方形gemm乘法，及M，K，N的大小都为16384。

当我们使用CUTLASS Profiler来进行运算，并且用nsight compute dump下来其运行过程中的一些情况，可以拿到他的一些信息，如grid的大小与block的大小等。

对于16384的float16/float32类型数据的gemm，cutlass的grid size是（512, 16, 1）-> 8192个block， block size是（256，1，1），一共是2,097,152个线程，因为最后产生C的大小是（16384，16384），所以平均每个thread需要产生128个C的元素。

从tvm的tensor expression出发，一步一步优化GEMM。

### naive gemm

最navie的情况，一个thread处理一个像素点的数据，这里我们选大小为1024的gemm为例子(因为这么做太慢了，大shape要跑很久)

```python
for m in range 1024:
    for n in range 1024
    	c[m][n] = 0.0
    	for k in range 1024:
            c[m][n] += a[m][k]*a[k][n]
```

对应到tvm的表达式：

```python
    # graph
    nn = 1024
    n = te.var("n")
    n = tvm.runtime.convert(nn)
    m, l = n, n
    A = te.placeholder((l, n), dtype=_dtype, name="A")
    B = te.placeholder((l, m), dtype=_dtype,name="B")
    k = te.reduce_axis((0, l), name="k")
    C = te.compute((m, n), lambda ii, jj: te.sum(A[k, jj] * B[k, ii], axis=k), name="C")
```

tvm可以显式的bind thread，例如我们把第一个维度的loop bind到blockIdx.x，第二个维度的loop bind到threadIdx.x，就完成了每个thread计算一个C的像素。

```python
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    
    i, j = s[C].op.axis
    
    s[C].bind(i, block_x)
    s[C].bind(j, thread_x)
```

运行下来的速度是2.xms是cutlass的二十倍左右，生成的cuda kernel如下：

```c++
extern "C" __global__ void __launch_bounds__(1024) default_function_kernel0(float* __restrict__ C, float* __restrict__ A, float* __restrict__ B) {
  C[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
  for (int k = 0; k < 1024; ++k) {
    C[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (C[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] + (A[((k * 1024) + ((int)threadIdx.x))] * B[((k * 1024) + ((int)blockIdx.x))]));
  }
}
```

### Blocked Gemm

分析上述生成的代码，每个thread需要从global memory读取两个数据，A和B，而显然，A和B中的数据都是可以被复用的，这就导致同一个数据被使用了几次，就从global memory中读取了几次，所需要地总的读取Global Memory的次数为 M\*N\*K\*2 次，写M\*N次。

![image-20220831122502957](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20220831122502957.png)

上图是CUTLASS实现高效GEMM的示意图，cutlass在第一层使用矩阵分块的方式来减少global memory的访问次数，在CPU上优化gemm同样也需要用到矩阵分块，不过cpu上的矩阵分块主要是为了fit L2 Cache的大小，这里的却略有不同，cutlass的图描述地不是很明了，绿色的方块代表的是一个block，负责产出一个C子矩阵的数据，可以通过A的浅蓝色的矩阵部分与B的浅黄色的矩阵部分乘累加得到，因为一个block内的thread共享share memory，**所以可以提前把这两部分浅色的矩阵cache到share memory中**。

$$
\frac{MN}{b_mb_n}(K*b_m + K*bn) =MNK(\frac{1}{b_m} + \frac{1}{b_n})
$$
现在暂时不考虑thread block tile这部分内容，则每个block内的thread各自负责一个计算像素，比如，我们一个block负责产生（32，32）个像素则在16384的gemm场景下：

> M,K,N: 16384，16384，16384
>
> grid_size: 16384, 16 
>
> block_size: 1024 (32,32)

对于grid来说，每一个block需要处理一个小的矩阵，具体到tvm的调度实现代码是：

```python
 	bx, xi = s[C].split(C.op.axis[0], factor=(block_h))
    write_code(str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/2.split_i.cu")
    by, yi = s[C].split(C.op.axis[1], factor=(block_w))
    write_code(str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/3.split_j.cu")

    s[C].bind(bx, block_x)
    s[C].bind(by, block_y)
    s[C].reorder(bx, by, xi, yi)
    write_code(str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/4.bind_block.cu")

    s[C].bind(xi, thread_x)
    s[C].bind(yi, thread_y)
    write_code(str(tvm.lower(s, [A, B, C], simple_mode=True)), "progress/5.bind_thread.cu")
```

把i,j两层循环各自拆分成（\_, 32）,(\_, 32), 再将这两层循环外提，就可以达到矩阵分块计算的效果，直接从global memory读取数据进行计算，一次计算的时间是4849.64ms，比cutlass要慢十倍。

接下来，给A、B、C加上cache，先以C举例，”local“表示结果暂存到寄存器中，则数据会先写到CC，再写回global：

```python
    CC = s.cache_write(C, "local")
```

lower:

```bash
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [268435456], []),
             B: Buffer(B_2: Pointer(float32), float32, [268435456], []),
             C: Buffer(C_2: Pointer(float32), float32, [268435456], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [16384, 16384], []), B_1: B_3: Buffer(B_2, float32, [16384, 16384], []), C_1: C_3: Buffer(C_2, float32, [16384, 16384], [])} {
  allocate(C.local: Pointer(local float32), float32, [1]), storage_scope = local {
    C.local_1: Buffer(C.local, float32, [1], [], scope="local", align=4)[0] = 0f32
    for (k: int32, 0, 16384) {
      let cse_var_1: int32 = (k*16384)
      C.local_1[0] = (C.local_1[0] + (A[((cse_var_1 + (blockIdx.y: int32*32)) + threadIdx.y: int32)]*B[((cse_var_1 + (blockIdx.x: int32*32)) + threadIdx.x: int32)]))
    }
    attr [IterVar(blockIdx.x, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 512;
    attr [IterVar(blockIdx.y, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 512;
    attr [IterVar(threadIdx.x, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
    attr [IterVar(threadIdx.y, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 32;
    C[((((blockIdx.x*524288) + (threadIdx.x*16384)) + (blockIdx.y*32)) + threadIdx.y)] = C.local_1[0]
  }
}
```

发现计算都在c\_local上进行了，但是thread和block都bind在C上，这样直接生成cuda程序会有错误，我们要把c\_local挪到C的第一个维度（因为都已经绑定到thread了，所以这里C的计算没有循环，需要绑定到yi。

```python
s[CC].compute_at(s[C], yi)
```

这样可以生成cuda程序，验证性能，不过只把C缓存到local，对性能没有影响，缓存A和B才是关键，但是缓存了A和B，就会尴尬地发现，share memory的大小不够用了，对于每个block，需要load两个16384，32大小的子矩阵，以单精度浮点数为例子，一共是256KB，一个SM也就128KB的share memory资源，所以不可以像这样一个thread计算一个像素。

| RTX 3090                                  |        |
| ----------------------------------------- | ------ |
| register file per SM                      | 256 KB |
| Maximum number of resident threads per SM | 1536   |
| Maxnumber of blocks per SM                | 16     |
| shared memory size per SM                 | 128 KB |

### Thread Block Tile

这一步需要进一步的决定，每个block里的thread是怎么计算的，对应cutlass efficent gemm中的第二张图，刚才的方法中，一个thread负责计算一个像素点的数据，在计算数据的时候每个thread又需要从global memory中读取两个 16384, 1 的数据，复用性极差。我们只能再进行分块，一次从global memory中load一部分数据进行计算。

![image-20220901141011638](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20220901141011638.png)

在这之后有两种计算方法，第一种是把浅色部分的矩阵按照inner product的方式，及下图所示：

![inner product](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/20220221185923.png)

还有一种是outer product，将最内层的K循环提出到最外层：

![img](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/20220221191615.png)

对应到cutlass的示意图，很明显，我看了很多加速gemm的资料，大家用的都是第二种外积的方式，不过也没有人说为什么要这么用，第一种内积的方式就是把A的浅色矩阵横过来切，我的理解是， 这样算下来有几个不方便的地方，一是外积的方式只需要决定Bk的大小，而外积的方式如果把K固定，那Bm和Bn就有两个参数需要调整，比较麻烦；二是在计算中，总有一些数据会被load多次，造成了额外的访存。虽然这种方式不能缓解inner product的局部性问题，但是在做矩阵分块的时候确实能更好地利用数据，不过外积的方式每次迭代存储的中间结果是一整个矩阵，内积是一小部分矩阵。

> More sophisticated algorithms construct the product hierarchically from multiplications of smaller matrix blocks [7]. By computing outer products on small blocks of the input and output matrices, we can more effectively exploit spatial locality and data reuse. Figure 4 illustrates a blocking scheme with parameters *I*0, *J*0, and *K*0.
>
> cite from : https://patterns.eecs.berkeley.edu/?page_id=158

通过这种方式，可以大大缓解之前从global memory获取数据带来的开销，在block计算每个小C矩阵的的时候，需要从global memory读取数据的次数变成了，设小C矩阵的长宽为Bm,Bn, 则总需要的访存量为：
$$
\frac{MN}{b_mb_n}(K*b_m + K*bn) =MNK(\frac{1}{b_m} + \frac{1}{b_n})
$$
也就是说，每次block只需要load BM\*BK + BK\*BN 大小的矩阵进入share memory，再循环K / BK次以完成整个矩阵乘法。

这样，就有几个超参数需要我们确定，Block size需要设置为多大？一个Block计算多少个数据？下面就这个问题讨论一下。

首先，关于block size的大小，一般会有两个具体的限制作约束，`Maximum number of resident blocks per SM` 和 `Maximum number of resident threads per SM`，也就是 SM 上最大同时执行的 block 数量和线程数量。要到达这个目的有多种方法，其中一个最简单的方法是让尽量多的线程同时在 SM 上执行，SM 上并发执行的线程数和SM 上最大支持的线程数的比值，被称为 Occupancy，更高的 Occupancy 代表潜在更高的性能。

显然，一个 kernel 的 block_size 应大于 SM 上最大线程数和最大 block 数量的比值，否则就无法达到 100% 的 Occupancy，对应不同的架构，这个比值不相同，对于 V100 、 A100、 GTX 1080 Ti 是 2048 / 32 = 64，对于 RTX 3090 是 1536 / 16 = 96，所以为了适配主流架构，如果静态设置 block_size 不应小于 96。虑到 block 调度的原子性，那么 block_size 应为 SM 最大线程数的约数，否则也无法达到 100% 的 Occupancy，主流架构的 GPU 的 SM 最大线程数的公约是 512，96 以上的约数还包括 128 和 256，也就是到目前为止，block_size 的可选值仅剩下 128 / 256 / 512 三个值。

不难发现，cutlass里的绝大多数kernel都选用的是128和256这两个值，当block中的thread数量变多，单个thread能吃到的资源就变小，反而会对编程产生限制，而对于grid size，在大部分gemm的情况下，取最外层的大的loop。

其次，关于一个Block计算多少个数据，由于我们已经确定了block的大小，即一个block里可以有128/256个thread，乘上每个thread处理的像素个数，就可以得到每个block要算几个像素了，从刚才得到的公式中可以知道，当一个block处理的像素点越多的时候，他的访存量就会越小，但显然这个值是有上限的，一个sm上的thread会均摊share memory。

一个有意思地问题是，当一个block处理的元素大小是128的时候，怎么选择这个c矩阵的长宽呢？

<!-- more -->
