---
title: TileLang autotuning internal
categories:
  - Technical
tags:
  - TileLang
  - MLSys
date: 2025-06-01 16:28:36
---

最近狠狠重构了一下tilelang的auto tuning, 本文介绍一下tilelang的自动调优部分当作一个小文档，以及顺道回顾一下本作者眼中的机器学习编译领域的自动调优发展。

<!-- more -->

## 机器学习编译领域的自动调优

2018年，天奇在OSDI 2018上发表了机器学习编译开山之作[tvm](https://www.usenix.org/conference/osdi18/presentation/chen)，观察到针对不同的硬件后端编写不同的编译算子(如矩阵乘法)，算子开发人员都需要编写不同的代码来实现，这导致算子开发成本非常高（即使是同一个架构，例如H100的各种阉割版，因为memory配比等不同导致最好的算子实现也会不一样），虽然用户使用的优化手段千千万，针对不同的硬件更需要使用不同的优化策略，但是用户想要优化的计算(算子)确实一样的，例如都是矩阵乘法。于是tvm首先将用户编写的算子变成了计算描述和调度原语两部分，对于同一个算子，上层的计算描述保持一致，针对不同的后端使用不同的调度原语对计算进行调度，最后通过codegen将调度之后的IR再lower到不同的硬件后端生成代码，如此，理想上用户只需要学习tvm提供的这一套基于compute和schedule的dsl就可以优化各种后端了。

![TVM Schedule Example](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/202506011706480.png)

虽然机器学习编译的出现可能可以大大缓解了用户编写高性能计算程序的压力，但是离实际的应用还相去甚远，也因此衍生出了很多有意思的研究方向，例如自动调优, 自动张量化，动态形状等，如今这个分支应该也发展的更加丰富了。本文主要以tilelang的自动调优为例，从我的视角介绍一下自动调优这部分近些年的一些有意思的工作，以及在tilelang里面自动调优的实现和我们的一些roadmap(这个应该挺有意思的)。

<!-- **AutoTVM at NIPS 2018: Learning to optimize tensor programs** -->

不难发现，tvm的玩家会把手动编写目标编程语言算子的问题转化成编写schedule的问题，然而根据目标硬件来应用高性能的调度原语仍然需要足够的领域专有知识。为了应对这一问题，一些自动调优工作被提了出来，自动调优工作的流程一般都会分成三个部分，选择候选解；根据当前的候选解生成代码和编译代码；验证当前候选的性能并通过某种反馈机制来选择下一组候选解，直接空间遍历完或者到达指定的步数。接下来介绍的几篇工作都是follow这个流程。

首篇有关自动调优的机器学习编译工作还是天奇在同年，也是NIPS 2018上的[Learning to optimize tensor programs](https://arxiv.org/pdf/1805.08166), 也就是AutoTVM。AutoTVM需要用户为特定的硬件平台提供搜索空间，并且使用机器学习技术(XGBoost, 没错，这个也是天奇做的..) 来预测不同的运算调度策略对性能的影响，最后选择最优策略。然而，用户仍然需要为特定的硬件平台自己设计一个合适的搜索空间，一个好的搜索空间设计仍然需要丰富的领域专用知识。以如图所示的面向类GPU架构的矩阵乘法优化为例，一个张量计算的高效实现往往需要经过多层分块（Tile），在a中的线程块分块部分，展示了最高级的分块，即将整个矩阵按照计算能力较大的区块进行划分，其中M、N和K代表原始矩阵的维度，而BM、BN和BK则代表在这一级别上块的大小，这种分块的策略确保了每一个线程块所工作的大小固定，不随着问题的变化而增长，方便针对任意问题进行通用的并行展开。图b所示的线程块内的线程束分块进一步细化了分块策略，揭示了更小的子块如何被分配到不同的线程束(Warp)执行。线程束是一组同时执行相同操作的线程，这里每个线程束处理的子块进一步被划分为更小的块，以更好地适应硬件的执行模型。图c部分描述了最细粒度的分块，这里的MMA（矩阵乘法累加器）代表实际的硬件执行单元（例如HFMA2、DP4A等），它们被设计用来并行处理矩阵乘法中最小的数据块。这种三级分块的目的是为了充分利用GPU的并行计算能力，通过不同层次的分块使得存储访问模式和硬件特性高度匹配，从而实现张量计算的高效执行。

![Common Problem Decomposition Methods in Multi-level Cache Architecture](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/202506011709710.png)

为此一个随意设计的典型的AutoTVM模板可以被精简成如表所示(当然这是化简过的，实际上可能还会精确到某一个具体的loop要怎么partition和Binding，要更加复杂)，其中choices策略表示从多项候选中选择其中一项, 则该空间内点的数目一共有187500个候选解，对于每一个候选解，编译器都需要进行计算调度、目标设备的代码生成、编译成可执行文件、生成测试张量，以及多次运行获取一个稳定的运行时间的过程。如果遍历所有的候选解，则在总体的搜索时间上是完全无法承受的。因此，AutoTVM采用了一些机器学习模型作为损耗模型（Cost Model）来指导下一个候选解的派生，其中最为常用的损耗模型之一为进化算法。进化算法能够在不需要遍历整个搜索空间的情况下，有效地探索并快速收敛到最优或近似最优的编译配置。**为了搜索出一个相对优的可行解，在机器学习编译领域常用的进化算法迭代步数为一千或者两千，即使是遍历完这一千次的结果，往往也需要花费小时级别的时间。**

![well defined search space](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/202506011711104.png)

<!-- **Ansor at OSDI 2020: Generating High-Performance Tensor Programs for Deep Learning** -->

AutoTVM仍然需要根据当前的计算模式与目标的硬件信息，自己定制一个搜索空间，考虑到算子的数量与目标硬件的数量，为如此多的组合都设计一个搜索空间依旧是一个困难的事情，并且还是需要一定程度的专家经验。2020年的OSDI上，Ansor作为AutoTVM的后续工作被提了出来（一作lianming zheng, lianming大哥有超级影响力的工作也已经数不过来了），利用生成式设计来自动构建搜索空间，为了用上生成式设计，这篇论文提出了很多新概念，比如Schedule Task，Sketch，包括首次提出了Loop Structure的概念，及把循环区分成Spatial和Reduce，并用SSR这种结构来描述计算，这个概念在之后的[TensorIR](https://arxiv.org/abs/2207.04296)也被沿用了，其实我也挺喜欢用这个概念的。总的来说，Ansor可以自动分析给定的计算图和目标硬件特性，基于此信息自动生成一系列可能的优化策略。包括数据布局转换、计算融合、内存访问模式、并行策略等。这种自动生成的过程大大减少了为每个新模型或硬件平台手动定义搜索空间的工作量。由于Ansor针对单算子的搜索空间是动态生成，程序开发者很难评估一个算子的具体搜索空间大小，但是可以明确的是，上文中定义的AutoTVM的搜索空间是Ansor整个大搜索空间的子集。由此可以看出，虽然Ansor为用户简化了一些过程，但是搜索空间却更大，更难以优化了。但是在当时，Ansor已经能够很好的解决大部分标量计算问题，给定一个计算图，Ansor可以自动生成一个搜索空间，并使用机器学习模型来预测不同策略的性能，最后选择最优策略。用户完全不用干涉搜索空间的设计，可以说是一个真正意义上的自动化调优工具，但是搜索空间还是挺长的，一个算子往往也要几个小时的时间才能搜索完毕。

然而近年来，随着硬件厂商不断地添加新的张量计算单元，Ansor等工作渐渐无法在给定的计算上达到硬件的理论性能上限，因为通过简单的代码生成无法充分利用硬件厂商提供的张量计算单元（比如Tensor Core），于是，自动张量化作为机器学习编译领域的一个重要工作被研究人员们重视起来，这其中当然也少不了自动调优工作的身影。

自动张量化的首要难题在于识别一组标量运算中哪些部分可以进行张量化处理。在2021年，[UNIT](https://arxiv.org/pdf/2101.08458)提出了一种通过TVM的 DSL来描述张量运算的模式，然后对计算表达式的语法树进行节点匹配。匹配成功的节点标识出了可以由张量化运算所替换的部分。将这些节点替换为张量运算节点，即可实现计算过程的张量化。尽管UNIT引入了匹配算法以辅助张量化过程，但它仍旧依赖于用户手动匹配张量化的语法树节点。为了进一步自动化这一过程，2022年，思泽(也是最近的triton-distributed的作者)在ISCA上发表了[AMOS](https://cs.stanford.edu/~anjiang/papers/ZhengETAL22AMOS.pdf)，其结合了UNIT的匹配算法和Ansor的自动调度，作为第一个自动张量化框架被提出来，但是和Ansor有一样的问题，一个矩阵乘法要调优大概8个小时的时间，并且也只能达到英伟达手写库性能的60~80\%，但是时间长和性能差的问题不是没有解法，在后文中我还会介绍两篇工作可以比较好的解决这个问题。

顺应张量计算单元越来越重要这一发展趋势，以及老的tensor expression越来越局限，难以描述一些新算子。2023年，TVM社区，也推出了新的张量中间表示形式——[TensorIR](https://arxiv.org/abs/2207.04296)，一作是国内tvm一哥思远(最近刚毕业在国内当AP了，招学生ing)，Tensor IR亦是TileLang前端的前身。同时，社区还提出了Meta Schedule(TensorIR版的Ansor加上自动张量化), 调优时间上和AMOS体验差不多，性能稍微好些， 但是还不能够达到库的性能。

那么，针对一个如矩阵乘法的简单算子，还有两个很难受的问题没有解决，一个是搜索时间过长，一个是性能还差点意思，接下来介绍两个来自微软亚洲研究院针对这些问题的相关工作。OSDI 22上，微软亚洲研究院发表了一篇名为Roller的论文，题目是[Fast and Efficient Tensor Compilation for Deep Learning](https://www.usenix.org/conference/osdi22/presentation/zhu), 以前的自动调优都假设硬件是一个黑盒子，对于一个搜索空间来说，我们其实很显然就能知道某一些点是十分不高效的，为了筛序出相对高效的点，Roller巧妙的利用了硬件的一些信息，如带宽，合并访存，计算访存比等，遍历一个大的搜搜空间，然后根据这些信息筛选出top k个点，然后遍历这Top K个点选择最优，一般而言这个k为10就足够，这样花几秒钟就可以完成整个搜索空间的遍历，并且能够媲美前文花费七八个小时才能搜索出来的结果，可以将调优速度提高数千倍。

虽然Roller的实现感觉有一些trivial，但是他真的work。笔者在两年前一直用进化算法来完成程序的调优(Roller之前基本上都是)，一个算子要花费数个小时才能完成调优，并且发现性能问题之后改进代码又得花上数个小时，在某一天突发奇想接上Roller之后，调优过程被缩短到几秒钟之后那种救赎感至今难忘...拥抱cost model, 节约阳寿。目前Roller被笔者重构，代码在tilelang下: https://github.com/tile-ai/tilelang/tilelang/carver 。以及MSRA后续还有一个基于Roller的跨算子融合策略Welder。

更进一步，我们再Trace一下这些框架的性能问题。如在类GPU的架构中，一个张量计算的高效实现往往需要经过多层分块，数据首先从全局存储中被读到共享缓存中，然后再从共享缓存中读到寄存器上参与计算。影响这种分块编程方法的性能因素主要有两个：第一个因素是每一层分块的大小，以及软件流水的程度决定了整体的计算访存比，合适的访存比可以使计算单元与存储访问单元能够重叠工作，进而隐藏延迟；第二个因素是带宽的利用程度，Roller指出，在英伟达上一代的GPU(Volta, Ampere等)上，当设备上的高吞吐率量单元Tensor Core被完全利用时，其各级存储的带宽也都会处于完全使用的状态，因此一个高性能的程序想要充分利用硬件上的加速单元，必然需要充分利用好带宽。

针对上述两个对性能产生影响的主要因素，现有的机器学习编译工作可以通过使用调度原语对循环进行分块，以及对迭代器进行线程绑定等操作来完成分块，TVM最新的分支已经支持比较灵活的软件流水的实现，可以很好的控制好一个计算程序的计算访存比。然而，较好得利用带宽仍然是一件非常困难的事情，GPU是多级存储的结构，其不同层级的数据期望访问不一样的访问模式：

- **全局存储**：全局存储是GPU中最大的存储层级，但同时也是访问延迟最高的。为了最大化全局存储的效率，线程块需要进行合并访存（Coalesced Memory Access）。合并访存是指多个线程同时访问连续的内存地址。在NVIDIA GPU上，理想的合并访存至少需要128字节（即32位浮点数的32个连续值）来最大化带宽利用率。
- **共享缓存**：共享缓存（Shared Memory）是位于线程块（Thread Block）内部，供所有线程共享的快速存储区域。为了有效利用共享缓存，需要避免Bank冲突。共享缓存被分为多个独立的Bank，每个Bank可以同时服务一个内存请求。当多个线程同时访问同一Bank的不同地址时，会发生Bank冲突，导致访问延迟。
- **寄存器**：寄存器层级的访存与具体执行的指令对齐，如Tensor Core指令要求的内存访问模式要求每个线程要取对应位置上的数据。

而输入的数据排布往往是固定的行优先或者为列优先存储，这多个级别的缓存每个级别都对最优的访存格式有不一样的需求，这导致简单的内存访问，即从全局存储到共享缓存用分块顺序存取，从缓存取数据到寄存器按照寄存器需要的排布取数据的方法必然会引入Bank冲突，导致内存带宽无法被充分利用。为了解决这一个问题，我们需要引入Swizzle的概念(nvidia在asplos2019的[Swizzle Inventor](https://mangpo.net/papers/swizzle-inventor-asplos19.pdf)中首次提出)，来精心控制一下内存的排布，刚好规避这一问题。其实针对某一个特定的后端，这种Swizzle策略基本上是一样的，所以知道有这个东西就行也不需要调优，要调优这个Layout还是一个有意思+比较难的问题，本作者在OSDI 24上的工作Ladder提出了一个有意思的想法来化解这个Layout问题。

## TileLang的自动调优

回到tilelang本身，上述的自动调优工作实际上的出发点是把schedule隐藏，不希望用户来操作schedule从而实现真正的自动优化和代码生成，不得不说如今tvm based的compiler已经能够做的相当自动了。tvm先有relay/relax的计算图IR来表示算子，针对每个算子有一个纯计算表达的实现，再通过自动调优来完成fuse和代码生产，感兴趣的用户可以体验一下我们之前的工作[Welder/Ladder](https://github.com/tile-ai/Ladder)，应该是这一思路工作的集大成者了(笔者觉得这一条路比现在torch inductor codegen triton要优雅的多，但是torch team并没用用tvm来做这一件事，感觉略可惜)。

其次，随着硬件和算法的设计越来越复杂，tvm提供的schedule渐渐不能够满足要求了，例如如何完成attention的flash版本的fuse，如何描述求逆，有依赖关系的新算子。于是schedule base的策略逐渐开始退环境。现在大家喜欢写的triton和tilelang选择将一些schedule暴露给用户，实践证明这一路线非常正确（虽然我觉得最终形态还是应该回归tvm一开始的设想，compiler自动都做了就好了）。

以tilelang的实现的矩阵乘法为例，我们研究一下现在tilelang和triton的自动调优的形态，一个没有autotune的tilelang程序是这样的:

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Clear local accumulation
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A
                # This is a sugar syntax for parallelized copy
                T.copy(A[by * block_M, ko * block_K], A_shared)

                # Copy tile of B
                T.copy(B[ko * block_K, bx * block_M], B_shared)

                # Perform a tile-level GEMM on the shared buffers
                # Currently we dispatch to the cute/hip on Nvidia/AMD GPUs
                T.gemm(A_shared, B_shared, C_local)

            # Copy result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

kernel = matmul(1024, 1024, 1024, 128, 128, 32)
```

`@tilelang.jit`可以把一个tilelang的程序编译成可以接受torch tensor的jit kernel，生成函数有6个主要参数，分别是M、N、K、block_M、block_N、block_K，其中前三项M、N、K决定了计算的矩阵乘法的shape（M×K矩阵与K×N矩阵相乘得到M×N矩阵），后三项block_M、block_N、block_K则是跟硬件相关的schedule参数，它们决定了每个CUDA线程块处理的子矩阵大小。对于一个给定shape的M,N,K，我们需要知道哪一组block_M、block_N、block_K是性能最好的，这就需要自动调优来帮我们找到最优配置。tilelang目前使用了和triton一样的设计，通过装饰器`@tilelang.autotune`来指定一系列候选配置，运行时会自动选择最优的配置。

```python
import tilelang
import tilelang.language as T

def get_configs():
  return [
    {"block_M": 128, "block_N": 128, "block_K": 32},
    {"block_M": 128, "block_N": 256, "block_K": 32},
    {"block_M": 256, "block_N": 128, "block_K": 32},
    ... 
  ]

@tilelang.autotune(
  configs=get_configs()
)
@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
  ...

```

回顾前文中提到的若干调优方法，这其实是最原始的手动设计空间遍历一个最优解的策略，那么是不是可以用一些更加好的策略呢，比如像Roller那样利用一些硬件信息来约束搜索空间? 我们实现了一个简单例子，在[gemm的example](https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm_autotune.py#L18)中，我们利用一些硬件信息来约束搜索空间:

```python
def get_configs(M, N, K, with_roller=False, topk=20):
    if with_roller:
        arch = CUDA("cuda")
        carve_template = MatmulTemplate(
            M=M,
            N=N,
            K=K,
            in_dtype="float16",
            out_dtype="float16",
            accum_dtype="float",
        ).with_arch(arch)

        func = carve_template.equivalent_function()
        assert func is not None, "Function is None"
        roller_hints = carve_template.recommend_hints(topk=topk)
        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")
        configs = []
        for hint in roller_hints:
            config = {}
            block_m, block_n = hint.block
            warp_m, warp_n = hint.warp
            # block_rows, block_cols represents warp partitioning
            block_rows, block_cols = block_m // warp_m, block_n // warp_n
            config["block_M"] = block_m
            config["block_N"] = block_n
            config["block_K"] = hint.rstep[0]
            config["num_stages"] = hint.pipeline_stage if hint.pipeline_stage > 1 else 0
            config["thread_num"] = block_rows * block_cols * 32
            config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
            configs.append(config)
    else:
        block_M = [64, 128, 256]
        block_N = [64, 128, 256]
        block_K = [32, 64]
        num_stages = [0, 1, 2, 3]
        thread_num = [128, 256]
        enable_rasterization = [True, False]
        _configs = list(
            itertools.product(
                block_M,
                block_N,
                block_K,
                num_stages,
                thread_num,
                enable_rasterization,
            ))

        configs = [
            {
                "block_M": c[0],
                "block_N": c[1],
                "block_K": c[2],
                "num_stages": c[3],
                "thread_num": c[4],
                "enable_rasteration": c[5],  # keep param name for backward-compat
            } for c in _configs
        ]
    return configs
```

虽然有趣，不过这个方法还有一些改进的空间，程序的输入现在是依靠`MatmulTemplate`来模拟一个计算程序，实际上我们可以通过分析tilelang program来得到具体的访存量等metrics；其次，CostModel沿用的Roller，再Hopper以前这个CostModel意外的非常work(甚至在MI300上也挺work的)，但是Hopper之后就差点意思，大概是还需要考虑到TMA的Pipeline。不过现在大家写triton/tilelang大都是一些空间比较小的算子，加上大家对triton/tilelang的期待大部分都是性能够用就行，所以关注度也不是很大，研究的兴趣也不是很高。

对于`@tilelang.autotune`, 我们提供了这些参数,

```python

def autotune(  # This is the new public interface
    func: Union[Callable[_P, _RProg], PrimFunc, None] = None,
    *,  # Indicates subsequent arguments are keyword-only
    configs: Any,
    # profile arguments
    warmup: int = 25,
    rep: int = 100,
    timeout: int = 100,
    # compile arguments
    supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto,
    ref_prog: Callable = None,
    supply_prog: Callable = None,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    max_mismatched_ratio: float = 0.01,
    skip_check: bool = False,
    manual_check_prog: Callable = None,
    cache_input_tensors: bool = False,
):
```

`supply_type`: 张量供给类型，决定如何为kernel提供输入数据，默认的Auto会针对数据类型来选择一个合适的分布，当shape是动态shape，或者像splitk的scheudle需要改变输入的尺寸，则autotune难以给出benchmark的tensor,此时我们需要  通过`supply_prog`: 用于生成测试数据。还注意到，我们还可以提供一个`ref_prog`: 参考程序，用于验证结果正确性，因为有一些config可能会导致程序的正确性问题(大部分发生在AMD的GPU上)，提供了这一参数的话，每次config还需要过一次正确性的验证。以及`manual_check_prog`: 手动检查程序，允许用户提供自定义的结果验证逻辑，这些参数让用户能够精细地控制自动调优过程。

Autotune的主要overhead，针对每个config进行lower和compile，以及evaluate两部分。其中compile可以并行化，evaluate因为和性能相关，需要串行，所以tilelang目前的策略是采用了编译并行+串行evaluate的策略，此部分由@小乱首先设计，把自动调优的速度相比于串行编译提高了两个数量级。

Autotune的cache分为两级，一级是存储在disk上的，他的key是一个相对完备的key，但是hash这些参数的代价实在是有点大，在4090上hash一个flash attention的语法树需要消耗10+ms的时间，但是相比compile要花费的10s(主要是cute的锅，不用担心之后会逐渐去掉cute):

```python
def generate_cache_key(self, parameters: Dict[str, Any]) -> Optional[AutotuneResult]:
    """Generate a cache key for the auto-tuning process.
    """
    # extract parameters from the function signature
    op_parameters = []
    for _, default_value in parameters.items():
        if default_value.default is not inspect.Parameter.empty:
            op_parameters.append(default_value.default)

    if self._kernel_parameters is not None:
        op_parameters += self._kernel_parameters

    func_source = inspect.getsource(self.fn)
    key_data = {
        "version": __version__,
        "op_parameters": tuple(op_parameters),
        "func_source": func_source,
        "configs": self.configs,
        "compile_args": hash(self.compile_args),
        "profile_args": hash(self.profile_args),
    }
    # Sort keys to ensure consistency
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()
```

所以我们还有第二级memory cache, 是对每一个autotune的实例，如果传入的参数一样则就可以避免对整个语法树进行hash:

```python
key = (key_args_tuple, key_kwargs_tuple)
```

如此可以把hash的开销基本上削减到0。

除了使用`@tilelang.autotune`作为一个装饰器来调优一个tilelang程序，我们还可以直接实例化一个AutoTuner来完成自动调优，虽然不太优雅但是能够控制的更加精细。

```python
autotuner = AutoTuner.from_kernel(
    kernel=kernel, configs=get_configs(N, C, H, W, F, K, S, D, P,
                                        with_roller)).set_compile_args(
                                            out_idx=[2],
                                            target="auto",
                                        ).set_profile_args(
                                            supply_type=tl.TensorSupplyType.Integer,
                                            ref_prog=ref_prog,
                                            skip_check=False,
                                        )
best = autotuner.run(warmup=3, rep=20)

# retrieve element
ref_latency = best.ref_latency
config = best.**config**
latency = best.latency
kernel = best.kernel
...
```

最后，浅谈一下triton/tilelang的autotune，仿佛回到了原始暴力搜索的年代，如果追求一个相对快的调优体验，就需要用户针对每个算子和每个硬件部手动设计一份精简的搜索空间(还不能满足任意的shape)，如果是要全面，空间又会很大，搜索起来非常慢。tilelang虽然通过roller一定程度上缓解了这个问题，但是还不够优雅，对tilelang的autotune的期待是可以根据程序的dataflow以及硬件的信息帮助用户生成空间，或者根据当前的形状自动推导出合适的config，相比于Roller时代把硬件变成一个白盒子，tilelang可以把schedule也变成一个白盒子。我们安排了一些小同学就进行这方面的探索，也欢迎大家关于这方面有意思的idea可以基于tilelang来实现或者合作讨论一下，让大家在这方面的体验变得更好。