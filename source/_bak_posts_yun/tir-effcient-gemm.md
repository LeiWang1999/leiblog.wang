---
title: tir effcient gemm
categories:
  - Technical
tags:
  - CUDA Programming
  - MLSys
date: 2022-09-06 15:31:55
---

上一篇[文章](https://zhuanlan.zhihu.com/p/560729749)中讲到如何利用cutlass优化gemm的思路，使用tvm tensor expression来实现一个高效的矩阵乘法，这里再探索一下直接从TIR Script把这个东西复现一下，对比一下两者的异同。

TensorIR 今年7月再arxiv上放了一篇preprint，感兴趣的读者可以自行阅读：https://arxiv.org/abs/2207.04296

不过写这篇文章的时候，tvm上游(main)分支的tir与paper里还不是一样，siyuan他们另做了许多改进，估计要等paper中了才会被合并到上游（貌似是在投ASPLOS？所以这里还是以tvm上我们可以实际操作的TensorIR Script为例子，优化的思路则不多讲解，和之前的tensor expression是一样的。

PS: 感觉TIR Script的设计和写法更贴近GPU，比tensor expression更抽象，有亿点点摸不着头脑，不过也比直接从tensor ir来构建一个dag要舒服地多，虽然通过自己瞎理解与实验加上在论坛交流了一下，也算是都摸出来怎么实现，但我相信应该还会有更优雅的写法。

代码还是放在: https://github.com/LeiWang1999/tvm_gpu_gemm

<!-- more -->

之前测出来各个阶段运行出来的时间如下：

| stage           | time (ms) |
| --------------- | --------- |
| 0.native_gemm   | 2.15598   |
| 1.blocked_gemm  | 4865.28   |
| 2.thread_tiling | 854.843   |
| 3.wrap_tiling   | 565.302   |
| 4.vectorize     | 423.255   |
| 5.double_buffer | 435.032   |

### native gemm

```python
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K])
        B = T.match_buffer(b, [K, N])
        C = T.match_buffer(c, [M, N])

        for i in range(M):
            for j in range(N):
                for k in range(K):
                    with T.block("B"):
                        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                        with T.init():
                            C[vi, vj] = 0.0
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

这里用到了两个修饰符，`@tvm.script.ir_module`，是把我们自己定义的class修饰成ir_module, `@tvm.script.tir.prim_func`是把main修饰成prim_function，可以把IRModule理解成一个shared library，而prim_func就是library里的可以call各种接口，比如这里定义了global_symbol的值是main，如此就可以在链接库的时候通过main来调用这个prim_func，可见

```c++
tvm/include/tvm/ir/function.h:178
 *  For example, we could set a global_symbol of a function
 *  early to make sure that we can always refer to it by
 *  the symbol name in the generated DLL.
```

`tir.noalias`与传统PL里的指针分析相关，具体可以参考[LLVM Alias Analysis Infrastructure](https://llvm.org/docs/AliasAnalysis.html#MustMayNo)，tvm里貌似没有针对这个flag做具体的优化，而是交给了llvm（目测是显式告诉llvm不存在指针之间的依赖关系，方便compiler进行分析）。

剩下的的结构就是抄抄文档里现成的代码改了，也不难理解， 这里的`vi, vj, vk = T.axis.remap("SSR", [i, j, k])`，其中“SSR” 中的S和R对应的分别是 spatial和reduction，表示i，j是spatial维度，k是reduction的，我觉得这种写法原始的出处应该是来自于ANSOR，详细研究的话可以去读读paper。

<img src="https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20220906171853792.png" alt="image-20220906171853792" style="zoom:50%;" />

另外，上述的tir script仍然有多种写法，首先看一下经过ir_module修饰之后的TIR AST(把TIR AST做一个TIR Script的Codegen)

```python
ir_module = MyModule
print(ir_module.script())
```

输出的结果是：

```python
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[(1024, 1024), "float32"], B: T.Buffer[(1024, 1024), "float32"], C: T.Buffer[(1024, 1024), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(C[vi, vj])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

可以明显发现的是，原来的三成循环被转化成了`for i, j, k in T.grid(1024, 1024, 1024)`，TensorIR基于Python AST设计，在这里把Python的语法做了一些转换，所以我们可以直接在自己写TIR Script的时候，就按照生成的Script的写法来写，也会达到一样的效果:

```python
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K])
        B = T.match_buffer(b, [K, N])
        C = T.match_buffer(c, [M, N])

        for i, j, k in T.grid(M, K, N):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

ir_module = MyModule
print(ir_module.script())
```

输出是:

```python
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[(1024, 1024), "float32"], B: T.Buffer[(1024, 1024), "float32"], C: T.Buffer[(1024, 1024), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(C[vi, vj])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

和原先的写法转换成的AST是一样的，是不是很简单呢（?其我第一眼把grid和block和cuda programming里的grid和block搞混了。。）

关于TIR Script有哪些特殊的写法，似乎现在还没有说明文档，我是通过翻一翻`tvm/python/tvm/script/tir/special_stmt.py`这个文件来找到一些说明，比如之前没有解释到的match buffer：

```python

@register
class MatchBuffer(SpecialStmt):
    """Special Stmt match_buffer(param, shape, dtype, data, strides, elem_offset, scope, align,
                                 offset_factor, buffer_type, axis_separators)

    Note
    ----
    This Special Stmt will perform different behavior depends on the type of param.
    If the param is a var in function parameter, it will create a buffer from DLTensor.
    Else if the param is a subregion of other buffers, then create a subregion match inside a block.

    Example
    -------
    Match buffer from function parameter
    .. code-block:: python
        A = T.match_buffer(a, (128, 128), dtype="float32")

    Match buffer from Buffer subregion
    .. code-block:: python
        A = T.match_buffer(B[0:128, i * 128 : i * 128 + 128], (128, 128), dtype="float32")
    """
```

接着下一步，实例化MyModule拿到一个ir_module，再从ir_module创建一个tir.Schedule。

```python
ir_module = MyModule
sch = tvm.tir.Schedule(ir_module)
```

一个有趣的问题是，tir的Schedule和TE的Schedule是独立的，两者都能够实现相似的功能，比如loop的split和fuse，thread bind等等，有什么不一样的呢？

回顾一下Tensor Expression创建一个schedule的过程：

```python
s = te.create_schedule(C.op)
ir_module = tvm.lower(s, [A, B, C], simple_mode=True)
```

对比一下发现，tensor expression是通过operation创建一个shcedule，之后再通过调用tvm.lower将整个schedule变成一个ir_module，tir的schedule是直接从ir_module创建schedule，再ir_module上进行操作，发现[有人](https://zhuanlan.zhihu.com/p/534062007)已经简单介绍了te和tir两者schedule的区别，这里给个结论：

| 关键步骤      | tvm.tir.Schedule              | tvm.te.Schedule                                              |
| ------------- | ----------------------------- | ------------------------------------------------------------ |
| 创建 Schedule | 从 IRModule 进行创建          | 从 Tensor 的 Operation 进行创建                              |
| 选取 Block    | 采用 DOM 对象文档模型进行选取 | 采用 Tensor 作为索引选取相应 Stage                           |
| 更新 IRModule | 直接对 IRModule 进行修改      | Stage 只是创建了 IterVar 的 Hyper Graph。需要进一步 lower 才能产生新的 IRModule。 |

绑定dimension到线程，编译与运行，生成出来的kernel与测出来的时间和之前基于te的方法是一样的。

```python
block_b = sch.get_block("B")
(i, j, k) = sch.get_loops(block_b)
sch.bind(i, "blockIdx.x")
sch.bind(j, "threadIdx.x")

ctx = tvm.cuda(0)
cuda_mod = tvm.build(sch.mod, target="cuda")

write_code(cuda_mod.imported_modules[0].get_source(), "tmp.cu")
```

| stage           | te time (ms) | tir time(ms) |
| --------------- | ------------ | ------------ |
| 0.native_gemm   | 2.15598      | 2.14892      |
| 1.blocked_gemm  | 4865.28      |              |
| 2.thread_tiling | 854.843      |              |
| 3.wrap_tiling   | 565.302      |              |
| 4.vectorize     | 423.255      |              |
| 5.double_buffer | 435.032      |              |

### Blocked Gemm

这一部分首要内容是做矩阵分块，需要使用split切分loop。

```python
block_b = sch.get_block("B")
(i, j, k) = sch.get_loops(block_b)
by, yi = sch.split(i, factors=[None, block_h])
bx, xi = sch.split(j, factors=[None, block_w])
sch.reorder(by, bx, yi, xi)
sch.bind(by, "blockIdx.y")
sch.bind(bx, "blockIdx.x")
sch.bind(yi, "threadIdx.y")
sch.bind(xi, "threadIdx.x")
```

这部分还是比较简单，这样写代码就可以正常work，但是上一篇文章这里还尝试了cache_write，先把C的内容缓存到local，再写入到global memory，而TIR在这里则略有不同。

```python
block_cl = sch.cache_write(block_b, 0, "local")
```

我们知道使用cache_write之后，ir_module不可以直接生成cuda code，因为`currently “C” is not contained in a thread environment or in the function arguments`，这一点和TE是一样的，所以还需要用compute_at，把C Local的计算挪到C下面。

```python
sch.compute_at(block_cl, yi, preserve_unit_loops=True)
```

但是这么做会抛出一个异常，`Error message: The block tir.Block*#0 is an output block*`,问题出在：

```c++
// tvm/src/tir/schedule/primitive/compute_at.cc:557
// Check condition 4): `block` is not an output block
  if (is_compute_at) {
    CheckNotOutputBlock(self, block_sref, scope_root_sref);
  }
```

我根据我的理解重新写了一下代码，改成split等再C Local上计算，再把C上的计算挪到C Local上，程序可以得到期望的结果：

```python
block_b = sch.get_block("B")
block_cl = sch.cache_write(block_b, 0, "local")
(i, j) = sch.get_loops(block_cl)
by, yi = sch.split(i, factors=[None, block_h])
bx, xi = sch.split(j, factors=[None, block_w])
sch.reorder(by, bx, yi, xi)
sch.bind(by, "blockIdx.y")
sch.bind(bx, "blockIdx.x")
sch.bind(yi, "threadIdx.y")
sch.bind(xi, "threadIdx.x")
sch.compute_at(block_b, xi, preserve_unit_loops=True)
```

但这是一件特别蛋疼的事情，你必须得把cache write写在最前面，这和之前写te的做法是不一样的，我在tvm的论坛上[提问](https://discuss.tvm.apache.org/t/confused-about-cache-related-tir-schedule-primitives/13487)，siyuan给了解答。

其实compute_at是有两个不一样的含义的，一个是`compute_at`，一个是`reverse_compute_at`，te是把这两种混合到了一起，而TIR表示为分开的两个函数，对于第一种情况，只需要把`compute_at`替换成`reverse_compute_at`，即可解决问题:

```python
sch.reverse_compute_at(block_cl, xi, preserve_unit_loops=True)
```

| stage           | te time (ms) | tir time(ms) |
| --------------- | ------------ | ------------ |
| 0.native_gemm   | 2.15598      | 2.14892      |
| 1.blocked_gemm  | 4865.28      | 4820.46      |
| 2.thread_tiling | 854.843      |              |
| 3.wrap_tiling   | 565.302      |              |
| 4.vectorize     | 423.255      |              |
| 5.double_buffer | 435.032      |              |

### Thread Block Tile

在这一部分，我们进一步把K轴根据BK切成两个部分，利用外积的形式计算矩阵：

```python
ko, ki = sch.split(k, [None, BK])
write_code(sch.mod["main"].script(), "4.split.cu")

sch.reorder(ko, ki, yi, xi)
```

接着，把A和B分别cache:

```python
block_b = sch.get_block("B")
block_shared_A = sch.cache_read(block_b, 0, "shared")
block_local_A = sch.cache_read(block_b, 0, "local")
block_shared_B = sch.cache_read(block_b, 1, "shared")
block_local_B = sch.cache_read(block_b, 1, "local")
block_cl = sch.cache_write(block_b, 0, "local")
```

这里当时纯靠自己感悟，没有找到说明怎么用的文档，我的第一个版本是：

```python
block_b = sch.get_block("B")
block_shared = sch.cache_read(block_b, 0, "shared")
block_local = sch.cache_read(block_shared, 0, "local")
```

发现生成出来的代码顺序乱了，及读shared的过程放到了计算之后，感悟了一下应该是这么写:

```python
block_b = sch.get_block("B")
block_shared = sch.cache_read(block_b, 0, "shared")
block_local = sch.cache_read(block_b, 0, "local")
```

第一个参数都得是block_b，而在这种写法下只有A被cache住了，于是我觉得第二个参数`read_buffer_index`是用来解决这个问题，于是就变成了正确的写法，回顾一下TE的cache read过程：

```python
AA = s.cache_read(A, "shared", [C])
BB = s.cache_read(B, "shared", [C])
AL = s.cache_read(AA, "local", [C])
BL = s.cache_read(BB, "local", [C])
```

如果要做类比，那么这里的block\_b应该被看作是TE Schedule里面的`[C]`。

接着要做compute_at，和之前的不一样，这里需要做两层的memory访问，TIR又有一些限制，如果学习TE版本的代码这样写：

```python
ko, ki = sch.split(k, [None, BK])

sch.reorder(ko, ki, yi, xi)

sch.compute_at(block_shared_A, ko)
sch.compute_at(block_shared_B, ko)
sch.compute_at(block_local_A, ki)
sch.compute_at(block_local_B, ki)
'''
te schedule:
    s[AA].compute_at(s[CC], ko)
    s[BB].compute_at(s[CC], ko)
    s[AL].compute_at(s[CC], ki)
    s[BL].compute_at(s[CC], ki)
'''
```

则会抛出一个异常，`The primitive requires all the consumer(s) of the given block to be present under the target loop. However, there are 1 consumer(s) not satisfying the constraint. List of the consumer(s):tir.Block#0`。

我再次感悟，必须先把block_local_A进行compute_at，正确的写法是:

```python
sch.compute_at(block_local_A, ki)
sch.compute_at(block_local_B, ki)
sch.compute_at(block_shared_A, ko)
sch.compute_at(block_shared_B, ko)
```

我觉得这为TIR Script的编写带来了一些限制，实际上，TE不会报错是因为他并不会生成IRModule，而是生成了一个customize过的AST，之后lower到IRModule的时候才会检查一些错误，这是一种懒加载的模式，而TIR是直接在IRModule上进行操作，所以有了这些限制。

最后，对Share Memory的Fecth过程进行并行加速:

```python
aa_yi, aa_xi = sch.get_loops(block_shared_A)[-2:] # loops size is 7
aa_ty, aa_yi = sch.split(aa_yi, factors=[Block_Size_Y, None])
aa_tx, aa_xi = sch.split(aa_xi, factors=[Block_Size_X, None])
sch.reorder(aa_ty, aa_tx, aa_yi, aa_xi)
sch.bind(aa_ty, "threadIdx.y")
sch.bind(aa_tx, "threadIdx.x")

loops = sch.get_loops(block_shared_B)
bb_yi, bb_xi = sch.get_loops(block_shared_B)[-2:]
bb_ty, bb_yi = sch.split(bb_yi, factors=[Block_Size_Y, None])
bb_tx, bb_xi = sch.split(bb_xi, factors=[Block_Size_X, None])
sch.reorder(bb_ty, bb_tx, bb_yi, bb_xi)
sch.bind(bb_ty, "threadIdx.y")
sch.bind(bb_tx, "threadIdx.x")
```

这里有一个很丑的写法是`aa_yi, aa_xi = sch.get_loops(block_shared_A)[-2:]`,其实他的loop长这个样子:

```python
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_shared"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_shared[v0, v1])
            A_shared[v0, v1] = A[v0, v1]
```

刚开始的时候，block_shared_A里面只有两层loop，但是经过compute_at之后就变成了7层loop，作者在[论坛](https://discuss.tvm.apache.org/t/confused-about-cache-related-tir-schedule-primitives/13487/5)里也给出了解释:

```python
I agree that get_loops has different behavior from that in TE. But getting all loops around the block looks better to me, instead of depending on the original loops. Also, that’s the result of interactive mode.
```

最后看一下性能:

| stage           | te time (ms) | tir time(ms) |
| --------------- | ------------ | ------------ |
| 0.native_gemm   | 2.15598      | 2.14892      |
| 1.blocked_gemm  | 4865.28      | 4820.46      |
| 2.thread_tiling | 854.843      | 960.696      |
| 3.wrap_tiling   | 565.302      |              |
| 4.vectorize     | 423.255      |              |
| 5.double_buffer | 435.032      |              |

居然还慢了一些，分析生成的cuda kernel发现，`T.init()` 这部分代码，居然是在最内部初始化:

```c++
for (int k_1 = 0; k_1 < 16; ++k_1) {
      for (int ax0 = 0; ax0 < 8; ++ax0) {
        A_shared_local[ax0] = A_shared[(((((int)threadIdx.y) * 128) + (ax0 * 16)) + k_1)];
      }
      for (int ax01 = 0; ax01 < 8; ++ax01) {
        B_shared_local[ax01] = B_shared[(((k_1 * 128) + (((int)threadIdx.x) * 8)) + ax01)];
      }
      for (int i_1_1 = 0; i_1_1 < 8; ++i_1_1) {
        for (int j_1_1 = 0; j_1_1 < 8; ++j_1_1) {
          if (((k_0 * 16) + k_1) == 0) {
            C_local[((i_1_1 * 8) + j_1_1)] = 0.000000e+00f;
          }
          C_local[((i_1_1 * 8) + j_1_1)] = (C_local[((i_1_1 * 8) + j_1_1)] + (A_shared_local[i_1_1] * B_shared_local[j_1_1]));
        }
      }
    }
```

这样会在运行的过程中产生大量无效的分支，而关于怎么把这部分代码提取出来，我尝试了各种写法，对照着repo里的各种test研究了半天，最后终于找到了对应的schedule primitive（，他就是`decompose_reduction`:

```python
sch.decompose_reduction(block_b, ko)
```

| stage           | te time (ms) | tir time(ms) |
| --------------- | ------------ | ------------ |
| 0.native_gemm   | 2.15598      | 2.14892      |
| 1.blocked_gemm  | 4865.28      | 4820.46      |
| 2.thread_tiling | 854.843      | 593.603      |
| 3.wrap_tiling   | 565.302      |              |
| 4.vectorize     | 423.255      |              |
| 5.double_buffer | 435.032      |              |

速度提升了300+ms，相比te还快了200+ms，这个并不是说tir比te的表现要好，而是在te版本的thread_tiling程序，A shared矩阵分块的时候随意了一些，导致写入A_shared的时候会产生大量bank conflict，笔者只是懒得改TE的代码而已...

### Wrap Tile & Bank Conflict

在这个阶段，我们使用vthread尽可能消除B的Bank Conflict:

```python
tyz, yi = sch.split(yi, factors=[V_Thread_Y, None])
ty, yi = sch.split(yi, [Block_Size_Y, None])
txz, xi = sch.split(xi, factors=[V_Thread_X, None])
tx, xi = sch.split(xi, [Block_Size_X, None])
sch.reorder(tyz, txz, ty, tx, yi, xi)
sch.bind(ty, "threadIdx.y")
sch.bind(tx, "threadIdx.x")
sch.bind(tyz, "vthread.y")
sch.bind(txz, "vthread.x")
```

但是如此运行程序会发现运行时间没有想象中那么好：

| stage           | te time (ms) | tir time(ms) |
| --------------- | ------------ | ------------ |
| 0.native_gemm   | 2.15598      | 2.14892      |
| 1.blocked_gemm  | 4865.28      | 4820.46      |
| 2.thread_tiling | 854.843      | 603.338      |
| 3.wrap_tiling   | 565.302      | 585.533      |
| 4.vectorize     | 423.255      |              |
| 5.double_buffer | 435.032      |              |

对比和te的代码，发现te自动对矩阵A做了transpose优化了访存，那么我们需要改写tir script中的书写，手动做一下A的transpose:

```python
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        AT = T.match_buffer(a, [K, M])
        # A = T.match_buffer(a, [M, K])
        B = T.match_buffer(b, [K, N])
        C = T.match_buffer(c, [M, N])

        for i, j, k in T.grid(M, K, N):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                # C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
                C[vi, vj] = C[vi, vj] + AT[vk, vi] * B[vk, vj]
```

| stage           | te time (ms) | tir time(ms) |
| --------------- | ------------ | ------------ |
| 0.native_gemm   | 2.15598      | 2.14892      |
| 1.blocked_gemm  | 4865.28      | 4820.46      |
| 2.thread_tiling | 854.843      | 603.338      |
| 3.wrap_tiling   | 565.302      | 561.727      |
| 4.vectorize     | 423.255      |              |
| 5.double_buffer | 435.032      |              |

### Vectorize

```python
aa_yi, aa_xi = sch.get_loops(block_shared_A)[-2:]  # loops size is 7
aa_yi, aa_ty = sch.split(aa_yi, factors=[None, Block_Size_Y])
aa_xi, aa_tx = sch.split(aa_xi, factors=[None, Block_Size_X * 4])
aa_tx, aa_vi = sch.split(aa_tx, factors=[Block_Size_X, None])
sch.reorder(aa_ty, aa_tx, aa_yi, aa_xi)
sch.bind(aa_ty, "threadIdx.y")
sch.bind(aa_tx, "threadIdx.x")

loops = sch.get_loops(block_shared_B)
bb_yi, bb_xi = sch.get_loops(block_shared_B)[-2:]
bb_yi, bb_ty = sch.split(bb_yi, factors=[None, Block_Size_Y])
bb_xi, bb_tx = sch.split(bb_xi, factors=[None, Block_Size_X * 4])
bb_tx, bb_vi = sch.split(bb_tx, factors=[Block_Size_X, None])
sch.reorder(bb_ty, bb_tx, bb_yi, bb_xi)
sch.bind(bb_ty, "threadIdx.y")
sch.bind(bb_tx, "threadIdx.x")

sch.decompose_reduction(block_b, ko)

sch.vectorize(aa_vi)
sch.vectorize(bb_vi)
sch.vectorize(sch.get_loops(block_local_A)[-1])
sch.vectorize(sch.get_loops(block_local_B)[-1])
```

最后的性能：

| stage           | te time (ms) | tir time(ms) |
| --------------- | ------------ | ------------ |
| 0.native_gemm   | 2.15598      | 2.14892      |
| 1.blocked_gemm  | 4865.28      | 4820.46      |
| 2.thread_tiling | 854.843      | 603.338      |
| 3.wrap_tiling   | 565.302      | 585.533      |
| 4.vectorize     | 423.255      | 561.727      |
| 5.double_buffer | 435.032      | 455.05       |

慢了20ms，对比和te的版本，是因为切grid的姿势不太一样，如果调整成和te一样的切法速度就会和te一样，挺玄学的（所以量化分析感觉还是挺难，我目测和GPU实际运行的时候调度thread的策略有关，总之，如果做一下grid切分的调整：

```python
(i, j, k) = sch.get_loops(block_b)
bx, xi = sch.split(i, factors=[Grid_Size_X, None])
by, yi = sch.split(j, factors=[Grid_Size_Y, None])
sch.reorder(bx, by, xi, yi)
write_code(sch.mod["main"].script(), "1.reorder.cu")
sch.bind(by, "blockIdx.y")
sch.bind(bx, "blockIdx.x")
```

则速度就会变成：

| stage           | te time (ms) | tir time(ms) |
| --------------- | ------------ | ------------ |
| 0.native_gemm   | 2.15598      | 2.14892      |
| 1.blocked_gemm  | 4865.28      | 4820.46      |
| 2.thread_tiling | 854.843      | 603.338      |
| 3.wrap_tiling   | 565.302      | 585.533      |
| 4.vectorize     | 423.255      | 431.63       |
| 5.double_buffer | 435.032      |              |

这里生成出来的kernel和te based的kernel已经一毛一样了，但是还有8ms的延迟我还是没check出来是哪里的问题。。

### unroll

在tir schedule里没有找到double buffer的primitive，不知道是因为作了什么样的考虑？所以这里就给大家表演一个unroll好了，想展开一段程序，可以:

```python
sch.unroll(ko)
```

则观察产生的cuda kernel,就会被全部展开(不能直接加一个 pragma 吗？不过在loop的层数比较小的情况下，nvcc会自己做一些unroll的优化，所以也不太需要care，。

### 总结

一些发现是，GPU从抽象的角度来讲，所有的线程都是在同一时刻运行的，但是落实到具体的硬件还是一个warp上的thread才是严格并行，如此当你的grid切的维度不一样，对性能还有一些细微的影响，这也是本文的tir script和te性能有一些细微差别的原因（schedule并不是严格一致的，比如grid size te是竖过来分，那tir这边则是尝试横过来分，发现性能还会有一些不痛不痒的差异，但这个差异却很稳定）

总的来说，现在TVM上游的TensorIR的文档还不怎么完善，踩了许多坑，希望能够帮助到有需要的同学吧。虽然基于IRModule的Schedule设计相对于原来TE的懒加载多了许多编程的限制，但这样也能更容易理解程序，熟悉了还好，其实Tensor IR的出现，兴许是为了弥补TE原本的不足，例如还不够贴近硬件（原来的te还很难描述winograd，很难做tensorize，这些才是tensorir真正发挥作用的领域，当然我survey这个东西的初衷不是为了只跑一下fp32的gemm，如果还能继续更下去，那么接下来的路线兴许是，使用tvm利用好vectorize（利用好HFMA2\DP4A指令），tensorize（利用好TensorCore），尝试一下在一些低比特上打到与cutlass相同的性能。
