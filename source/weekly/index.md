---
title: weekly
date: 2021-02-06 13:56:27
---

## 20220424

1. 为了深入了解了一下多面体模型，读了一篇MIT 2019年的《Tiramisu: a polyhedral compiler for expressing fast and portable code》，发现他的设计和Halide和TACO基本上一毛一样，主要是针对多面体模型设计了一个DSL，（Halide的设计是计算和调度分离，然后可以用Autotuning 解决，TACO是数据和算法分离，做稀疏计算的DSL），然后分了四层中间表示来做多面体模型优化。
2. 读了AutoTVM的那一篇《Learning to Optimize Tensor Programs》，在读ASPLOS 22的《A Full-Stack Search Technique for Domain Optimized Deep Learning Accelerators》，Google基于timeloop来做一个加速器最优调度策略搜索的文章。
3. 尝试基于MLIR写一个存算一体的dialect，这个框架更新的太快了，导致很多接口都变了，非常不稳定，定义了dialect和运算后，写lower的transformation，很多头文件和函数都变了，非常难用！

## 20220417

1. 本周看了一点nnfusion的源代码，首先回到一个上古问题，nnfusion的细粒度是怎么实现的？nnfusion的vDevice和vEU，在GPU上对应着SM和CUDA CORE，但是对于GPU来说，写CUDA的kernel函数我们能够控制的粒度是thread，一个thread里的程序会自动分配到多个cuda core上，nnfusion是通过一个叫做persistent programing的技术可以控制每个cuda core的行为（在GPU 社区里好像也挺有名的）
2. 给nnfusion递交了两个代码，第一个是加了几个tensorflow op的测试，刚好用到了之前学的google test，已经被合并了。第二个是发现他项目里有很多拼写错误(typo)，我加了个自动拼写错误检查机器人，代码量有点大还没有完成合并。
3. 陈晓明老师拉了个存算一体编译器的群，这个任务正在推进，让我先讲讲现在NN编译或者工具链的架构和优化的内容，我做了个PPT，比较基础和入门，以后可以留在组会上讲讲。

## 20220410

1. 王颖老师和我提了一嘴专用处理器的编译器敏捷设计，跟我说崔慧敏老师提出的cvm的思路比较好，可以看一看，但是网上没有搜到相关的资料。
2. MLIR中国社区组织了一次线上的交流讨论，没有宣传，是几个社区负责人发邮件通知参会的，我运气好被邀请过去了。结果参会的只有二十个人，都是大咖，杨军、张磊、袁进辉、simbanova的几个，而且崔慧敏老师也在，每个人要自我介绍一下自己基于mlir的工作以及对mlir的优点缺点评价一下，我个没啥贡献的快到我的时候我就退出会议跑路了🤦‍♂️，会议上认识了崔慧敏老师，发邮件打了招呼，询问了关于cvm的事情，是希望做成JVM那种，在加速器之间也能有个类似字节码和虚拟机的机制来迁移。这部分目前还没有系统的发布，只是有一些跨平台的优化技术。有兴趣的话可以和崔老师详细讨论，感觉是一个很make sense的东西，但是我感觉这部分的难点是加速器和编译器中间一般还隔着一层驱动程序，这是一个gap，如果是vendor提供的话还好，但要是vendor提供的，cvm和tengine那种框架有什么本质区别吗？
3. 最近又读了一篇论文，msra的rammer那个组osdi22又中了一篇sparta，我读了他的preprint spargen，据说是之前投apsys被拒的，但是思路应该是一样的，就是说把sparse作为tensor计算的一个属性，而不是说重新做一个sparse tensor来分析，这样针对tensor计算的sparse属性做一些数据流分析，这个和mlir的sparse tensor的概念是一样的，这简直是大势所趋。

## 20220327

1. 学习了google test这个框架的使用。
2. 学习了CUDA编程，把《CUDA C Programming Guide》过了一遍。
3. 继续看了TACO，因为mlir的sparse tensor这个dialect上写的也是参考的taco设计的。
4. TACO也把计算和调度分离了，但是好像没有支持自动调优。

## 20220320

1. 研究了一下tvm现在在稀疏上的工作，ANSOR这篇论文最后总结的时候说不支持稀疏计算的自动搜索，但是我发现tvm现在是有一些对稀疏矩阵乘的自动搜索工作的，《Auto-scheduling Sparse Matrix Multiplication on CPU with Custom Sketch Rule[¶](https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/tune_sparse_x86.html#auto-scheduling-sparse-matrix-multiplication-on-cpu-with-custom-sketch-rule)》。我跑了一下，读了一下代码。稀疏和稠密计算的区别就是稀疏矩阵有多层的访存，这个auto schedule的过程我觉得最大的收益来自于考虑到了这个多级访存的关系，我觉得是很make sense的。

   以BCSR的压缩格式，：X->(128,256) W->(256,512) sparsity: 0.6 BS_R:16, BS_C:1，执行一次计算的时间0.195 ms，相比之下，使用numpy/scipy.sparse来进行同样的数据、同样的压缩格式，运行速度在5ms左右，还是快了很多的。

   **但是这个针对稀疏的auto schedule我觉得还是有一些不足，比如不能自己选取合适的压缩格式，例如bcsr的block的长宽仍然需要人为指定，现在的auto schedule不能搜索这个参数。**

2. 翻了一下tvm的关于sparse的issue，发现陈天奇和杨军之前都在issue里提到taco是一个很好的借鉴框架，于是我就去读了taco的论文《The tensor algebra compiler》，MIT做的，2017年的PLDI。

   主要是设计了一个DSL来做稀疏计算的代码生成（生成C程序，编程时，用户得告诉taco输入tensor有多少，每个tensor有几维及每个维度的稠密稀疏性质，每个tensor是按照什么样的稀疏方式存储的，然后给定计算规则，之后通过编译和链接调用taco生成代码就可以了，虽然是codegen，但我个人理解这个东西是给出这个稀疏计算的算子的高效实现可能长成的样子，然后方便工程师根据他生成出来的C程序进行手动调优到最好性能。

   我刚看到这个东西感觉性能应该不咋样，因为他文章里都没有提到tiling和fusion，支持的数据压缩格式也有限，后来发现他在PLDI 2018上的《Format abstraction for sparse tensor algebra compilers》加了很多稀疏格式的支持，PLDI 22《An Asymptotic Cost Model for Autoscheduling Sparse Tensor Programs》和 OOPSLA 20 《A Sparse Iteration Space Transformation Framework for Sparse Tensor Algebra》加上了一些auto tuning方面的工作。

   于是我现在想测一测现在的稀疏加速框架速度都是怎么样的，发现这个taco有点残破，非常难用！

3. taco做代码生成的时候用到了格结构来做代码生成，但为什么用也没有看明白，我花了半天的时间研究了一下偏序集和格结构。

4. taco生成的c程序用到了openmp，正好这学期选了高性能计算系统的课，这周过了一遍MPI的语法和openmp的用法。

5. 关于MLIR的sparse_tensor的dialect，我发现我的cornell的朋友组里也有在基于mlir做稀疏计算的工作，Washington的朋友也在做同样的工作（他说这个sparse_tensor在设计上不如他们的设计），貌似这个方向还是很受关注的。

下周工作：

1. 把mlir的sparse_tensor的dialect用起来，看看性能怎么样，怎么设计的。
2. 待读论文《MLIR: Scaling Compiler Infrastructure for Domain Specific Computation》

## 20220313

1. 读了《Union: A Unified HW-SW Co-Design Ecosystem inMLIR for Evaluating Tensor Operationson Spatial Accelerators》

   还是gatech(scale-sim)和nvidia(timeloop)两个团队的人打造的模拟器生态。

   Motivation：

   	1. 现有的模拟器都各有侧重点，例如SCALE-sim主要用来仿真脉动阵列， MAESTRO适合应用于仿真的阵列具有可配置的宽高比，Timeloop在模拟的时候可以考虑到复杂的内存层次结构建模，Tetris可以建模3D阵列；
   	1. 新的算子很难适配到原有的生态中，因为cost model和网络的mapper之间互相约束，
   	1. 利用mlir把现有的模拟器都耦合到同一套生态里来（前端使用mlir的生态，可以把一些神经网络的模型转化成mlir的常用的dialect来分析，union-opt可以使用各种不同的搜索策略（他们分别来自于现在不同的模拟器采用的解决方案，backend是做什么他没有说清楚，只是说明了是feature work，这样看起来，MLIR的作用似乎只有在前面接受模型转化成affine dialect的前端部分用到了？

   ![image-20220313231405437](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20220313231405437.png)

2. 偶然发现今年的2月份，mlir多了一个sparse_tensor的dialect，是google提交的，还有一份附带的arxiv，《Compiler Support for Sparse Tensor Computations in MLIR》，正在研究。

## 20220227

1. 本周读完了《Dual-Side Sparse Tensor Core》(ISCA 21')，这一篇最主要的insight就是第一次在稀疏运算相关的工作中引入了outer-product矩阵乘(其他工作基本是基于inner-product的)和bit-map压缩算法来重新设计了一下tensor core，具体的内容也形成了一篇post:https://zhuanlan.zhihu.com/p/471048499 扫码可打开知乎:

   ![8061559B-1363-42FB-8CA1-7F081E852E4C](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/8061559B-1363-42FB-8CA1-7F081E852E4C.png)

   总结：这篇论文的工作量挺大的，在稀疏网络的推理过程上也取得了不错的加速效果，而面积仅仅大了2%。读完也有不少的收获，比如tensor core的矩阵乘大小为什么要选4\*4\*4，一些典型的稀疏硬件方面的工作等。但是他是把整个tensor core都给改成了8\*8的outer-product模式，**虽然文中没有提，但是在图表中可以看出在稠密矩阵乘法上的效果要比原来的sparse tensor core要差**，是拆东墙补西墙了，这里应该是一个tradeoff。

   为什么不用inner-product? 我觉得是因为有之前的工作做过，Sparten,inner-product加上bitmap压缩算法，下周读一下这篇论文。

2. 听了两次meeting，一次是oneflow社区组织的关于MLIR的应用的，比特大陆、sambanova他们都有做自己基于MLIR给AI加速器设计软件栈的报告，但是都是在做比较高层的importer、切图、量化这些的，不是说给加速器设计算子库和驱动（我觉得这个才是问题好么），不过至少说明是有人在使用MLIR来做这些工作的。另外一次是周三晚上的《Real-Time DNN Execution on Mobile Devices with Compiler Optimizations》，虽然标题是说NN优化的，但实际上是讲的今年PLDI 2021上的DNN Fusion的工作，主要内容是对fusion做了分类，分成了One-to-One、One-to-Many、Many-to-Many、Reorganize、Shuffle几种，然后通过代数化简、算子融合来做优化，最后在一些网络上得出的performance和TVM这些比起来简直效果惊人。但是源代码没有公开，不敢相信直接做算子融合就能带来这么大的收益吗？

2. 关注了一个软件所PLCT的buddy-mlir的项目，因为mlir本身是一个lib，这个项目就是搭了一个可以用mlir跑自己的项目的框架，目前可以自己定义dialect、写算子的实现，我本周给这个项目提交了一小部分代码，主要是对他的边缘检测的示例增加了对darwin系统(macos)的支持，作者把我拉进了他们的开发者群，这个项目目前还在起步阶段，还只能跑一个卷积算子，我希望在接下来的时间里可以增加跑网络的feature，并且试试能不能写一个存算一体的dialect.

下周计划：

1. 读一篇inner-product+sparse tensor的硬件加速方法：《*SparTen*: A Sparse Tensor Accelerator for Convolutional Neural Networks》
1. 看看tvm里是怎么处理稀疏运算的?..
1. 借助buddy-mlir学习一下mlir怎么用.

## 20220220

1. 本周在macbook上手工优化了一下gemm，主要是从循环展开、向量指令、数据pack等方面来做加速，形成了一篇post：https://zhuanlan.zhihu.com/p/468304964，扫码可打开知乎：

   ![image-20220220210018024](/Users/wanglei/Library/Application Support/typora-user-images/image-20220220210018024.png)

2. 简单了解了一下Tensor Core的工作机制，A100卡上的Sparse Tensor Core的稀疏加速用的是上上周读的Bank Sparsity的方法，简单来讲，对于W\*A，把大矩阵W拆分成很多个1\*4的小块，然后强制让稀疏度为50%，即每4个元素，去除掉其中绝对值最小的两个值，这种稀疏压缩方式成为(2:4 bank sarsity)，对原本的tensor core也只需要做很小的修改，像下图中加一个mux四个有值的下标来选出与之匹配的矩阵A中的元素进行运算。

   ![image-20220220210511461](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20220220210511461.png)

3. 《Dual-side Sparse Tensor Core》是指出这个sparse tensor core要求稀疏度是固定的50%，而且只考虑到了weight sparsity不能考虑到activation sparsity，于是魔改Sparse Tensor Core的一些工作，在Accel-Sim（GPGPU的模拟器）上进行了验证，但是论文太长，具体是怎么做的也没整明白。

下周计划：

1. 继续看完《Dual-side Sparse Tensor Core》
2. 看看tvm里是怎么处理稀疏运算的？

## 20220213

关于稀疏硬件mapping的survey：

1. 关于现在的图神经网络训练框架，比如DGL，在A100的卡上跑能不能吃到他的硬件sparse加速，这个答案询问了DGL的作者是不能，sparse tensor core很难适配上，这个单元设计出来是为了network pruning的，所以单纯做稀疏矩阵加速运算不太好适配，但可能还是可以适配的？
2. 关于FeatGraph，他的作者之一在知乎上看到了我写的论文笔记，同时也是dgl的作者之一，跟我讲featgraph不是一个特别值得follow的工作，提供的insight也有限，那个时候能中可能主要是这个能用jit来优化gnn的饼，当然他觉得给sparse kernel做auto-tuning还是有很大空间的，只是这篇的search sparse不一定对，而且抽象粒度也不是很够，于是我去了解了一下软件上做稀疏加速的方法，学习一了一下一些基本的稀疏格式（CSC、CSR、BCSR）和他们的Auto Tuning方法，除此之外，还看了一片FPGA19的Paper，《Efficient and Effective Sparse LSTM on FPGA with Bank-Balanced Sparsity》提出了一种叫做bank sparsity的压缩格式，主要侧重在对任意的稀疏率，PE都负载均衡，而且硬件实现比较友好，比如对于tensor core而言，加一个multiplexer就可以了，不知道TVM的稀疏是否考虑到了这些特征呢？

下周的工作：

1. 看看tvm里是怎么处理稀疏运算的？
2. 还有一些魔改sparse tensor cores的文章，看一下去年微软亚研院的《Dual-side Sparse Tensor Core》。

关于存算一体编译器的survey：

1. 回答一下陈晓明老师在上次去会议上提到的问题。

   a . 思考了一下为什么tvm被称为编译器，到底做了哪些事？我本来以为AI编译器是指前端接受模型到生成硬件指令的端到端的软件，但是回来仔细思考了一下发现，tengine、ncnn、tnn、paddle lite这些端到端的软件大家都称之为NN推理引擎，只有TVM、OPENVINO这些被称为编译器，知乎上也有人提出了这个问题，即推理引擎和AI编译器的区别在哪儿，根据我的学习和总结，我觉得最本质的区别就在于适配硬件的解决方法，比如对于一个卷积操作，像Tengine、NCNN的解决方法是手写汇编，手动并行、写向量指令来极致压榨处理器的性能，如下图：

   ![image-20220213225801630](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20220213225801630.png)

​		但是这种方案的可复用性差，首先向量指令不同的指令集不一样，即使是同样指令集的处理器，cache大小也不一样。所以Tengine的卷积目录下，有riscv的实现、cortex-a/cortex-m \* arm v7/v8 的实现，而且像这样能把处理器性能压榨到极致的工程师也很少，与爆炸的算子和硬件的组合矛盾。

​		TVM借鉴了Halide的思想，设计DSL将运算和调度分离，像这样：

![image-20220213230233829](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20220213230233829.png)

​	之后通过一些auto tuning的办法找到这些参数，现在已经可以远超人类手写的平均水平了。

​	b. 为啥TVM要用到llvm，tvm用llvm生成cpu侧运行的代码，我觉得最主要的原因是要使用llvm去生成各个处理器对应的vector指令。

2. 思考了一下存算一体编译器，主要和现在的推理框架做的东西都差不多，Tengine的开源协议是Apache2.0，所以可以直接拿来用。假设前面的模型转换和切图、量化等都使用Tengine，那么我们实际上要实现的就是一个计算图到指令的生成软件，有点像Bangc这种东西，我觉得需要设计这样的软件是一个DSA现在的通病，也是一个痛点，但是MLIR说致力于解决这个问题（虽然现在还不知道咋解决的吧，还需要survey一下，是不是可以用MLIR来设计呢？）

下周的工作：

1. 已经开启了一个小项目，在mac上通过手写汇编的方法来优化一下gemm试试。

## 20220128

本周找了一篇2020编译器的综述《The Deep Learning Compiler: A Comprehensive Survey》

1. 第一个思考，深度神经网络编译器到底做了什么事情

具体在前端和后端的每个阶段做的优化手段综述里有一章图片解释：![image-20220128232443844](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20220128232443844.png)



2. 深度神经网络编译器和推理引擎的区别是什么

对一个模型做端到端的推理除了tvm，还有像ncnn、tengine、paddle lite这些推理框架，但是大家一般不把后者称为编译器。

这个结合知乎上的讨论：[深度学习编译器和推理引擎的区别是什么？](https://www.zhihu.com/question/401766029)

发现大家对于编译器和推理引擎的界定并没有很明确，但tvm和第二类软件框架也有明显的不同，侧重在于优化加速的方案：

a . tvm都是在ir层面的优化，最后的输出也是生成llvm、OpenCL、Cuda的中间表示，再配合AutoTVM/AutoScheduler(AutoTVM是需要用户提供一个Schedule来定义一个搜索空间，比如某一个循环分块的tiling的步长，然后tvm用机器学习的方法来迭代几次找到最合适的参数，2020年的工作Ansor把自己写Schedule这个工作也省去了)。

b. Tengine、NCNN这种推理引擎的优化手段是把不同的算子都在不同的设备上实现一遍，不用llvm。比如ncnn对接很多处理器都是靠手写汇编的方式来不断提升性能，Tengine也是在针对不同架构的处理器，比如armv7/v8都有不同的gemm的加速实现代码；这种方法虽然理论上可以获得性能最好，但是能写出高效算子实现的有经验的工程师没那么多，tvm的自动生成的方法能够超过工程师们的平均水平，在设备爆炸的时候优势会很明显，比如Tengine现在的cuda代码运行效率就很低，因为大佬不用，所以大家写的代码质量不高。

3. 不能直接用LLVM来加速卷积运算吗？非得在前面多个tvm？

首先，LLVM框架本身自带的Pass都在这里：https://llvm.org/docs/Passes.html 

其中没有用来加速矩阵乘的，所以LLVM本身是没有对矩阵乘法的优化的，但是也有人基于LLVM来做过这方面的工作，这个项目叫polly、（[Polyhedral optimizations for LLVM](https://polly.llvm.org/)），也是比较有名的框架，但是这个框架本身不能完成网络的端到端推理。

其次，LLVM的中间表示是基于三地址码，现在的神经网络模型都是基于计算图的，两个之间的差别太大，有很多优化在计算图上才能更好的被发现，所以在三地址码之前多一个中间表示会更好，这个就是TVM的Relay IR，来做一些计算图层面的优化。

4. DDR和卷积Buffer的通信下界问题

对于NVDLA的卷积Buffer，这些逻辑都是在kmd（NVDLA的驱动程序，挂在操作系统内核里，负责配置寄存器）中实现的，看代码分析他的数据流些困难（是把weight固定，还是特征图固定），因为寄存器的说明太少了。按理说对接Tengine的工作是管不到这部分内容的，所以其速度快两倍的原因需要上板子再看一下(开发板在雁栖湖，回学校看看)。

虽然这是一个科研问题，但是我看陈晓明老师的HPCA20《 Communication Lower Bound in Convolution Accelerators》似乎已经把这个问题从理论层面解决了啊。。

5. 关于存算一体编译器的定位问题

首先，Tengine的开源协议是Apache 2.0 可以随意使用，包括商业等等都可以的，所以至少基于Tengine来完成模型解析和切图这些工作都是没有问题。

假设设计完了指令，编译器也可以生成指令了，那么我们是要把生成的指令保存到内存里，然后存算一体的RAM的外围电路会取指执行？还是说生成的指令也要vendor来写具体实现？

春节假期间：

1. 调研一下现有的编译器框架的代码生成，或者有没有自定义指令集相关的内容。



## 20220123

近些日子在看看图神经网络这种非常稀疏的网络运算系统中有没有什么自己可以做的编译优化，其实在编译现在主流的图神经网络训练框架DGL的时候就不难注意到其依赖项里是有TVM的，这是不是说明现在的DGL也在使用TVM来进行自动调优呢？带着这个疑问我翻了一下DGL的代码，发现和tvm有关的部分只有一个叫做FeatGraph的框架，顺藤摸瓜找到了胡玉炜大佬发表在SC20上的Paper：

[《FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems》](http://link.zhihu.com/?target=https%3A//www.csl.cornell.edu/~zhiruz/pdfs/featgraph-sc2020.pdf)

在2021年6月亚马逊云科技 Community Day 上，AWS的张建老师有一个[《图神经网络和DGL在实际落地项目中的挑战和思考》](http://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1yK4y1M7B2%3Fshare_source%3Dcopy_web)的Talk，指出现在主流的图神经网络框架DGL的自己裁剪的Gunrock之后制作的minigun来做运算加速的，但是根据代码大胆猜测一下实际上DGL只在在0.3～0.4中才有使用的是minigun来做一些加速，在0.5中就不使用minugun了，而是将主要的运算抽象成了SpMM(稀疏稠密的矩阵乘）和SDDMM（sampled稠密稠密矩阵乘）两种运算，这项工作在DGL达到版本0.6的时候结合tvm的高效代码生成转变为了FeatGraph发表在SC20上，而现在DGL已经前进到了0.7版本了。

我读了FeatGraph的论文和代码（加上benchmark的代码，有效代码只有一千多行），主要就是用tvm的底层api封装了spmm和sddmm这两个操作，不过一作是tvm的作者之一，论文的作者还有李沐等，通过tvm找到最有调度方案的方法来加速，在cpu和gpu上对现有的mkl这种手工的库都取得了一定的加速效果。**但这还是和稠密算法的加速方案是一样的，并不能看到专门针对稀疏做了什么优化，软件层面加速稀疏算子似乎是一件非常困难的事情**，不知道直接使用可以支持硬件稀疏运算的A100训练会不会比V100训练图网络要快很多，但是我们没有A100的卡，没有办法实践。

找了一篇综述《Computing Graph Neural Networks: A Survey from Algorithms to Accelerators》，刚好是今年发的，还在读。

下周：

1. 继续看这篇论文，看看稀疏加速大家的一些工作

2. 需要思考：从稀疏运算到稀疏架构的映射这一条路径这个想法，是指针对A100这种硬件上支持了稀疏运算加速的架构来生成高效的cuda代码？需要与cudnn这一类库对标？

## 20220109

考完算法课之后，用dgl运行了几个图神经网络在Cora数据集上跑了跑，dgl这个框架本身也使用了TVM来进行自动的调优。关于上周，稀疏架构是什么的问题，我觉得有两个选项吧：

1. A100这种专门做了稀疏加速的训练GPU
2. HyGCN这种专门针对图神经网络设计的，但是一般只有模拟器性能，和软件这边搭便太少了

下周安排：

1. 弄清楚GNN在dgl里是怎么放在GPU上运行的，是将邻阶表转化成邻阶矩阵进行计算吗？

## 20220103

目前只剩一门在1.7号的卜老师的算法考试。

这段时间感觉看CIRCT和MLIR没什么头绪，问了一下王颖老师有没有什么想法。王颖老师说，tvm的稀疏加速做的不太好，可以考虑一下从加速框架到稀疏加速专用结构的这条映射通路。因为稠密tensor加速结构的自动生成有很多人做，比如flextensor，interstellar，稀疏的还没有，我觉得是挺不错的idea，然后现在的图网络就是非常sparse的一系列模型，所以在253上建了个dgl的环境，在学习图神经网络。

另外，之前伯克利那个把nvdla用chisel重写的yuda，现在入职了壁仞计算核设计组做gpgpu，好像那个chisel的sodla不准备继续搞了。

## 20211205

1. 本周继续调研了MLIR的内容，又写了一篇笔记，正在做组会汇报的PPT。

## 20211128

1. 写完了编译的最后一个大作业，把上星期写的笔记完善了一下。
2. 调研了MLIR：
   1. 看了一下作者在LLVM 2019年的大会上做的报告，根据作者的描述是为了解决现在各个厂商的AI框架都有自己的一套IR，MLIR似乎是想做类似ONNX这样把IR统一的工作，但是关于MLIR大家的评价不是像LLVM、TVM那样一边倒的支持，算是褒贬不一吧，有人认为MLIR不过是把原来各大AI框架的碎片化变成了MLIR内部的碎片化，这一点来看似乎在走ONNX的老路，犯了和ONNX类似的错误；有人说他是一个优秀的编译器库，但不是一个优秀的编译器框架，就我目前的调研来看，我比较支持这个观点，不明白MLIR的定位究竟是啥，因为也没有看到基于MLIR很实际的case。
   2. MLIR在2020年发过一篇Paper：标题是[MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054)，又是重新给MLIR定了一个方向，是因为摩尔定律的终结的现状下，各个硬件厂商都在做DSA，MLIR可以更好的解决异构系统的问题，但是怎么解决我还没有悟到？
   3. 还有一些知乎文章，主要是官方的一些文档的内容也比较少。
   4. 在253上编译了一个mlir的镜像，把他的tutorial过了一遍。

## 20211121

学习LLVM、复习编译程序高级教程这一门课，整理了一篇笔记，还没有写完：

## 20211114

1. 本周看了一篇OSDI20上的《RAMMER: Enabling Holistic Deep Learning Compiler Optimizations with *r*Tasks》内容是讲微软亚洲研究院开源出来的一个NNFusion的推理框架，将网络推理的粒度切的更细了，从而得到了更快的加速比，是非常不错的工作。

## 20211107

1. 本周舍友（专项技术中心）组会上有一个分享AI编译器综述的，我去听了。具体内容是清华大学发的20年的一篇综述《The Deep Learning Compiler: A Comprehensive Survey》，其次寒武纪的有个MLU让他们在做工具链，我听了一下内容和我之前做的那一套差不多，他们是在基于tvm来做，对接的是bangc，现在可以跑lenet和vgg这两个网络，加了一下做相关工作的同学的联系方式。
2. 把LLVM官方的Tutorial之自己动手写一个简单的编译器过了一遍，加深了一些了解，学习编译。
3. 《新时代中国特色社会主义》结课，写了结课的论文。《高科技企业管理》结课，为了写结课论文看完了《做卓有成效的管理者》这一本书。

## 20211031

1. 本周认识了王颖老师的一个学生，交流了一通之后发现读的论文还是有很多相似点的，他目前的idea是做动态神经网络的即时编译，我也跟了一下这个新网络，主要是看了清华大学黄高课题组做的一个综述和报告。
2. 编译程序高级教程的作业要用LLVM完成一些任务，在完成任务的时候从LLVM官方的Tutorial开始，在重新学习LLVM。
3. 在追李沐大神的带你精度经典论文系列，本周看了Transformer。

## 20211017

1. 读完了Simba这一篇，我觉得Simba就是一个多核的NVDLA这个IDEA的实现，但是感觉没什么做的地方了，但是对多核之间的软件问题可能可以有研究，但也没有学习的平台。
2. 抽空看了NASA，是加速NAS的训练过程之中的推理过程，因为NAS的训练过程中网络结构一直在变，与平时推理的时候，网络结构不变，输入一直在变的情况是不一样的。于是针对这样的推理特性，设计了一种NAS加速器。而NASGuard：因为NAS搜索出来的网络会出现多分支的情况，这算是对这种robust网络结构设计的加速器，有调度电路等等的。这两个都不是之前听说的用神经网络架构搜索的思路搜索一个架构，而是针对某中类型的网络设计的电路，但是这种思路似乎也有，关键字是automl architecture,找到了一篇，还没看。
3. 在学写模拟器，研究Scale-Sim这一篇论文，是ARM开源出来的一个脉动阵列的模拟器。

## 20211010

本周主要都是在写作业，太多了。

1. 思考了一下怎样决定图的切分，使得整个网络推理的时间最短这个问题？但本质上是因为加速器缺少一个能完成数据remap的硬件单元，不应该靠软件来解决这个问题，感觉意义不大
2. 第二个是多核的方向，在读Simba这一篇、学习了Chiplet是个什么概念。

## 20211003

1. 本周看了DianNao里的几篇之前做NN加速器的Reference和DaDianNao，感觉寒武纪他们并不是一开始做这方面工作的。而且大家在体系结构方面用的技术好像也都差不多，不过重在工作扎实，性能很好，然后开始迭代，这个才是重要的吧。
2. 看了一篇异构系统调度的论文FlexTensor、是北大CECA团队基于TVM做的一个自动算子调优的工作，不过也是自动调tiling的参数之类的，对之前提到的怎样合理切图的工作没什么启发。似乎没有找到像类似的需求，需要再思考一下这个问题是不是只是一个很简单的工程问题。
3. 之前提到把chisel版的nvdla做出来的人现在在拿这个创业，现在是伯克利的nprocessor初创公司的CEO，他们约了台湾做ONNC的那家公司做一个报告在下周一（十月四号晚上七点），让我感兴趣可以伪装成nprocessor的员工去旁听。
4. 根据我的猜测，网易有道部门要做的关于OCR之类的加速器的应用应该是服务于他们的那个智能笔，扫一扫识别单词之类的智能产品。

## 20210926

1. 本周主要是完成了Tengine+NVDLA的这一套工具链的直播课程
2. 上次提到的那个国产公司startfive居然最近A轮融了十个亿，这钱这么好融的吗？
3. 在看一些经典的论文，细看了DianNao。

## 20210918

1. 今天上午与兵器导控所的人进行了腾讯会议的交流，了解了一下他们的需求：

   1. 目前他们之前已经做过一个芯片，买了ARM A9、ARM M7的IP，现在想把nvdla_large做成一个ip挂载到原来的系统上。
   2. 目前是想对nvdla做一个评估，所以就算性能差也没有关系，但是性能还是尽可能的好比较好。
   3. 目前他们团队还是只有一个人在搞这个事情，想完成nvdla的vcs和verdi的仿真（这部分工作我们之前应该有过积累）
   4. 在FPGA上实现远行验证，把驱动程序和umd都给跑起来（他们的板子是VCU440，是Xilinx最大的板子应该，上面有5000k的LUT资源，而且他们买了ARM A9的IP）
   5. 但是他们在ARM A9上跑的是国产的操作系统，名字我没有听清。
   6. 他们也提到可不可以做多核的dla这样的思路（毕竟他们还有M7这个ip，可以给每个dla挂个小核？），刚好可以提供一个验证平台。

   我和他们说了我们这边最好能促成合作的意愿，而且之前我们这边也有过一些积累。他们表示很乐意促成合作，具体的细节要和韩老师聊一下，问问能不能要一下韩老师的手机号码。

2. 本周的一个小插曲，之前伯克利有一个大佬用chisel重写了nvdla的联系我想让我帮忙看看能不能评估一下chisel版本和官方的verilog版本哪个好，主要是看timing和资源使用之类的，我看他的nvlarge综合出来的资源暂时是比verilog少的，但我觉得从verilog这边出发优化应该能做的更少？ 主要的收获是了解到他目前在伯克利的产业孵化skydeck做创业，公司名字好像是叫nprocessor，有三条产品线，分别是soDLA商用版授权，cora的开发，还有把arm赞助的arm ip集成到chipyard的框架里。

3. 另一个小插曲是，国产公司startfive（sifive china）之前有一块板子，没猜错的话应该是惊鸿 7000，里面有一个riscv和large，他们想找人把片上的large跑起来。这个活我觉得还挺有意思的，因为板子没赛灵思的板子那么吵，但是我暂时没和他们联系。startfive好像也不怎么景气，好像帮他们做工具链的团队也跑路了。。

## 20210912

1. 本周完成了cmod的变异，成功的搭建了vp的环境，但是问题是cmod的代码很难看得懂。请教了之前的那个做AI芯片的公司，他们说内部分析DLA还是看VMOD比较多，看VMOD的工程师懂了之后CMOD的代码很容易就可以理解了，所以分析硬件还是得看vmod。
2. 接触了一下网易的远程开发环境，他们的环境还是挺舒服的。通过openvpn连接到内网，然后用vnc连到一个Ubuntu的物理机器，zcu104就接到这个物理机器上，远程开发的时候相比在所里没有噪声这是比较舒服的。其次，关于如何远程控制开发板的开关，使用一个小米家具的无线开关，在手机上可以远程遥控开发板的电源，很无敌。。
3. 这周，北京兵器导控所的一个人，他们也想调研NVDLA来做一个演示，好像是不太在意性能的那种，问我个人或者课题组有没有意愿合作什么的，具体的要等下周商量。。

## 20210905

1. 本周花了几天的时间完善了PPT的内容

2. 上周提到的那家AI公司在研究多个DLA的调度情况，感觉是一个好的方向；交流中还了解到他们也是先进行CMOD的开发，在QEMU里跑仿真再做硬件（他们是RISCV+DLA这样的一个系统。目前看来，研究CMOD代码还是比较重要的，比如可以考虑在模拟器上挂两个DLA试试？

3. 网易的NPU开发部门想设计加速器，来加速自己的OCR、NMT，TTS，CAPT等应用，虽然里面也有NN，，但感觉用NVDLA效果应该不理想吧？接下来协助一下他们将NVDLA在FPGA上部署、实现之前的基于Tengine的工具链，看看他们是怎么开发的。

4. 学校生活上，又成了班级党支部的书记可能事情会有点多，选课基本上满足预期：

   <img src="/Users/wanglei/Library/Application Support/typora-user-images/image-20210905234400829.png" alt="image-20210905234400829" style="zoom:50%;" />

下周计划：

1. 学习QEMU、在服务器上搭建vp环境。

## 20210828

本周我了解了一些关于会议、期刊论文的一些知识。并且学习了一些QEMU的技能，我决定在QEMU上把NVDLA的CMOD模型挂上去来做仿真（就是构建一个官方那样的VP环境），这样可以脱离开发板来开发软件程序，也应该是一个标准的流程。

真的有一些公司是基于NVDLA做加速器设计的，本周接触了一些企业的人：

1. 和浙江杭州的一家不愿意透露姓名的AI芯片公司的compiler leader交流，获得了不少有用的信息，他们流片了Large，并且加上了RUBIK和BDMA（因为Large的官方的Spec文件里是没有BDMA和RUBIK的，然后还加了一些自己的模块，他们结论是速度不比RK和海思的差，这里海思应该指的是NNIE吧）他们内部也有评估板，但是也不肯借给我玩。。

   然后，sifive arm有一块惊鸿7100评估板上面有一个large，不过买不到这个板子，他提醒我Tengine可能有，因为之前Tengine那边有一块，因为这个评估板上有sifive自己的nne加速器，软件是找Tengine来设计的。结果问了一下Tengine的人，，已经还回去了，不过可能今年会拿出来卖？Tengine感觉做这个团队人很少，无法支撑一个可靠的芯片。

<img src="https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210828192048729.png" alt="image-20210828192048729" style="zoom:50%;" />

<img src="https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210828193238940.png" alt="image-20210828193238940" style="zoom:50%;" />

2. 网易也准备基于NVDLA搞加速器，感觉他们把这个工作想的太简单了，还想让我去打工：

   ![image-20210828193441354](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210828193441354.png)

   <img src="https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210828193621554.png" alt="image-20210828193621554" style="zoom:50%;" />

3. 一个叫思尔芯的公司，感觉不厉害。

4. 还有一个在台湾国立交通大学留学的老外，叫Nagendra也在调研NVDLA看来：

   <img src="https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210828194003932.png" alt="image-20210828194003932" style="zoom:50%;" />

下周计划：

1. 继续学习QEMU
2. 完善PPT

## 20210822

这两周完成了 Yolox-Nano 的量化与调试，把代码整理优化总结、合并到了Tengine的Github仓库里，写了教程与宣传文章，发布在了知乎和博客。

## 20210817

上周把几个能加的算子都调了调，然后自己量化了一个yolov3-tiny，可以work了。不过标准的v3-tiny是leaky-relu，都切给cpu去跑了，这个时候耗时1秒7，如果换成relu的话，一帧需要600多毫秒。但是yolo-v3-tiny-relu这个版本，我训练的精度不太行，还没法做demo。

现在准备第一阶段进行小结，准备先发布一个能用的版本，在写文档啥的。

## 20210808

1. 本周继续为Tengine添加了NVDLA的Relu、CONV、ElementWise、FullyConntect等四个算子的实现。

2. 先说结论：加上上周写的pooling的算子一共五个算子已经具备了运行resnet18-cifar10的条件，在板卡上成功进行了推理，一张图耗时只有10ms，比我毕业设计的时候快三倍，应该是Tengine对计算图做了很多优化，比如把Batchnormalization层和conv融合了。

3. 为了跑模型，需要解决模型转换和模型量化的问题，两者Tengine都提供了工具。Tengine的量化工具与TensorRT的原理一样，都是需要sample一系列图片作为校准集校准，但是NVDLA应该是考虑到硬件设计的简单化，在驱动层面实现pooling层之后没有加scale的选项，而Tengine量化的时候avgpooling的输入会成input_scale，输出会除output_scale，这导致模型用DLA推理的时候失真严重，经过两天的排查之后发现了这个问题（比如把pooling单独切到cpu上运行发现精度恢复正常，那就是pooling有问题了，顺便也证明了切图可以正常work）和Tengine的量化专家探讨了半天，我认为怎么tensorrt量化的就可以正常work呢，结果发现tensorrt做量化的时候应该做过特殊处理，他量化之后得到的参数默认得avg_pooling的input_scale和output_scale一致，应该是运用了一些透传规则。量化专家建议我试试从前往后传和从后往前传scale两种方式，及尝试输入的scale直接赋给输出的scale、或者输出的scale赋给输入的，，没想到真的能解决这个问题，而且实验发现从前往后传比从后往前传的效果要好一些。


下周计划：

1. 试试跑yolo，Tengine那边想直接跑yolox，但是其中的激活函数都是hardswish，他们请旷视的工程师下周帮忙训练一个都换成relu的，虽然精度会降低但对加速器友好，希望能顺利。

## 20210801

本周学习了解了Tengine，并且开始进行了为其添加 NVDLA 后端的工作。我在 https://github.com/LeiWang1999/Tengine 进行工作，到今晚已经完成的工作有：

1. 使用 Tengine 实现了计算图自动切图，将 NVDLA 支持的算子定义，然后 Tengine 会自动把计算图切分成几个子图 (subgraph)，我们只需要关心这个子图怎么用 NVDLA 去跑就行了，其他的子图会自动用别的设备（比如说CPU去运行）。

2. 将 Tengine IR Graph 转换到了 NVDLA 用的 EngineAST 。原本 NVDLA 的软件栈由编译器进行模型到中间表示的转化，之后进行优化，接着序列化成 Loadable 文件，Runtime 进行反序列化，调度硬件资源，中间由一个 Loadable 文件作为媒介，我的做法是将中间的序列化和反序列化的过程抹去，将 Compiler 和 Runtime 的程序都整合到Tengine里，这样可以使用 NVDLA 原有的中间表示。

3. 首先，我从一个最简单的 Max Pooling 算子开始实现，已经完成了整个工作流程，将 Tengine 的 IR 转化成 NVDLA 的 EngineAST（中间有许多Mapping的过程，包括Tensor的Mapping、IR的转化），最后将EngineGraph emit 成 Runtime 所需要的数据，使用 Runtime 分配内存和推理。整个过程略微复杂，在作业的过程中我分别调试Tengine和NVDLA原有软件栈的程序，对比运行中的数据才使得错误较少。

4. Pooling OP的Test，我测了几组数据都是对的。其中 NVDLA 的数据摆放问题也比较关键，将输入数据给NVDLA的时候，要按照他的要求摆放，取出数据也要进行还原，研究了一天左右，最终还是work了。

   > 我也了解到在部署领域这种数据摆放的问题还是很常见的，一般加速器都需要特殊的数据格式，但比如芯原底层，有个 tensor process 单元，非常快的做这个事，NVDLA则需要手动来移内存，这个公式想了我半天。。。

接下来的任务：

1. 支持更多的算子，pooling既然已经work了，接下来开始添加卷积和Relu这两个算子，然后就可以跑简单的网络了，说起来NVDLA支持的算子其实也有十个左右，，是个体力活。
2. 这个过程还是比较复杂的，但我觉得是十分有意义的工作。文字描述有限，等下次组会的时候给大家分享一下。

## 20210725

完成了 NVDLA 在 ZCU 102 开发板的移植工作，关于这部分技术细节新写了文章：https://zhuanlan.zhihu.com/p/392974835

## 20210718

1. 本周阅读调试了 ONNC 的开源版本的代码，尝试添加算子，ONNC 的代码写的很规范，添加算子比NVDLA的那套流程要简单明了。而问题是， ONNC 只支持 full 版本，意味着不可以上FPGA开发板调试。但是如果是添加 Sigmoid 这类算子的话，不需要改动 Runtime 的代码，这样就可以在原本的 vp 环境中验证，但是要加一些他不支持的算子，需要在 Runtime 的代码里添加 emulator 的 op 实现（cpu 运行的算子都会放到用emulator 去执行），意味着需要改动 Runtime 的代码，而且不好调试，难度大。而且即使添加完了算子，需要解决问的问题还有，高版本的 yolo 似乎也没有 onnx 的版本，量化问题等。

2. 而为了解决这个问题，我找到了新的方向，可能是因为老师转发了 NVDLA 的那边公众号文章，一位业界大佬（Tengine 推理框架的负责人）看到了我的毕业设计：

   <img src="https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210719124508514.png" alt="image-20210719124508514" style="zoom: 67%;" />

   具体来说，我们需要把 NVDLA 支持的算子交给 NVDLA 实现，其他算子交给 CPU 实现，这个过程叫做**计算图切图**，Tengine 本身有一个 NPU 的后端，支持的是 TIM-VX，这样的机制已经做得比较完善了，而且前端支持绝大部分的框架、也完成了量化等（因为 TIM-VX 只支持 unit8），他们有意向找一个开源的FPGA加速器的后端，于是联系了我，我觉得这个如果能学会了对很多加速器设计都是刚需。

   有关Tengine的 NPU 后端讲解：https://live.csdn.net/room/weixin_43476455/lt1qpikr

   ![image-20210719124629891](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210719124629891.png)

   下周任务：在 ZCU 102 上完成 NVDLA 的移植，学习 Tengine。

## 20210704

1. 本周读了一篇论文《MAESTRO: An Open-source Infrastructure for the Cost-Benefit Analysis of Dataflows within Deep Learning Accelerators》，算是一篇综述，利用 MAESTRO 来评估像是 Eyeriss、NVDLA 这些硬件结构的调度性能，这篇论文的收获是知道了一些经典的AI加速器论文，例如 Eyeriss、FlexFlow等，制定了接下来暂时要阅读的论文列表：

   <img src="https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210704195406392.png" alt="image-20210704195406392" style="zoom:50%;" />

   很巧的是卢美璇师姐本周看了一篇 Time Loop 的论文，其工作看起来是 MAESTRO 的延续，仔细分析了一波这部分的工作都被英伟达的一个团队所垄断了，~~难怪每次都要评估NVDLA~~。

2. 将分析 NVDLA Compiler 工作机制进行了延续，总之结论就是想跑 Yolo 非常困难。首先，NVDLA的编译器接受的是 Caffe 的模型，里面有一些运算也是不能用加速器来做，比如最后的 Softmax，就是用 CPU 来算的，但是他的前端只有一个 Caffe 的 Parser ，这导致只能跑一些很简单的分类网络，对于目标识别网络，例如 Yolo、虽然也可以转成Caffe的模型，但是里面常见的 LeakyRelu、NMS 这些算子 Compiler 这边是不支持的。

   但是根据我的调研，对这些算子做支持的工作也有人已经做过了，是由台湾的一个团队研发的 ONNC、有看到别的台湾人用 ONNC 在 ZCU 102 这些板卡上跑 YOLO、我也和他们发邮件聊了一下，ONNC 也分开源和商业授权两个版本，开源的直支持 nv_full、并且不能量化、支持的算子也有限，商业授权版本的 ONNC 就可以支持 YOLO 这些目标检测网络的算子、支持量化等、可以实现一个端到端的调度，支持的算子也是开源版本的两倍。**但是这个台湾人说且商业授权大概要 800~1000萬 (人民幣約 200~250萬)一年。**

   另一个，就是官方提供的 YOLO V3 的 Demo 我也调研了一下，他原本是在 firesim （aws fpga）平台上运行的，用的是 NVDLA Large 挂在 RocketChip 上，因为 AWS 服务器注册需要信用卡，服务器使用起来也貌似有点贵，我在本机搭建了一下环境。这部分是官方把 Darknet 的代码进行了更改，让 NVDLA 支持的操作调用 Runtime 去执行：

   <img src="https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210704203114826.png" alt="image-20210704203114826"  /> 

   专门拆出了三个 Loadable，应该是 Yolo 里能用 NVDLA 进行处理的三大块，以不能用 NVDLA 进行处理的几个算子做的切片，这个 Loadable 的生成不是用编译器生成的，根据官方的 issue 里的描述，是使用的内部工具。

   这样在 NVDLA Large 下，跑个 Yolo V3 应该问题不大了，跑其他的网络/NVDLA 配置就要研读这部分代码，反向研究一下这个 Loadable 该怎么自己生成。

   于是，根据这两周的调研，想要推理 YOLO 网络，就需要把 Leaky RELU 等不支持的算子使用 CPU 去调度，我能想到的有四条路线：

   1. 直接操作寄存器读写，这个需要对 NVDLA 的寄存器很熟悉，然后需要手动把网络的每一层数据都量化、提取成单独的二进制文件。该方案的问题是灵活度太低，针对不同的网络的工作量太大（手上有一份 ZCU 102 Rev 1.0 板卡、nvdla_small 的 demo、可以运行的版本似乎只有可执行文件、但是提供了量化好的 Yolov1 tiny 的权重二进制文件）。
   2. 基于 NVDLA Compiler/Runtime 更改，为 Compiler 加上不支持的算子的代码，Runtime也要做一定的更改，但是没有看到基于这套方案的工作，首先 Compiler 需要大改，因为只有 Caffe 的 Parser 限制了很多网络的类型，其次这个代码用 Makefile 组织，这个组织结构错综复杂，改起来太麻烦了。
   3. 基于 ONNC 更改，首先商业授权版本的可以跑 Yolo 等，可以以开源版本作为切入点，首先其前端是 ONNX 的 Parser、模型天生支持的算子就比 Caffe 要多很多，其次是以 CMake 组织现代化一点，然后还做了 Python 的接口，代码相对 NVDLA 的软件栈要规范很多，学习的收获应该会大一点。但是对于 ONNC，还没有 Debug 调试，理解其运行的机制，这些特性是否是优秀的还是 YY 阶段。其次，这部分工作只是 Compiler、在板卡上的运行时还需要对应的修改，工作量也是很大的。
   4. 基于 darknet-nvdla 的修改，这部分工作，我觉得能够搞明白还是比较有意思的，他对 darknet 的源代码修改量不大，官方提供的 demo 是在 firesim 上跑的 riscv + large，这个配置使用 chipyard应该是可以在 VCU 118 上复现的（刚好 352 有一块），运气好的话，这个 demo 能跑起来。 问题是如果要自定义一些东西，比如想跑别的 Yolo 模型，或者不使用 Large、用 Full 的配置，可以预见的困难就有，怎么模型拆分，编译成对应的 Loadable 文件，这其中还涉及到一个 YOLO 网络的 INT8 量化的问题，难度看起来也很大。

   **下周的工作：**

   1. 读一篇 Eyeriss v1 的论文
   2. 读一下 ONNC 的开源版本的代码

## 20210627

1. 本周看了浩博师兄发给我的一篇论文 《NASGuard: A Novel Accelerator Architecture for Robust Neural Architecture Search (NAS) Networks》，论文的阅读笔记放在了博客：https://leiblog.wang/NASGuard-A-Novel-Accelerator-Architecture-for-Robust-Neural-Architecture-Search-NAS-Networks/

   总结：Robust NAS是研究人员用神经网络架构搜索的方式来研究神经网络的结构本身针对神经网络模型的对抗训练攻击的防御性而产生的。虽然本篇文章是和 NAS 相关的论文，但不是针对 NAS 漫长的训练过程设计的加速器，而是作者总结近年来典型的 Robust NAS 网络，发现其存在较多的多分支结构、分支与分支之间存在较多的互联情况等问题，这些问题的存在使得常见的加速器的PE阵列资源利用率变得不高，以及因为互联造成了并行度的限制，导致需要多次与外存交互。这篇论文提出了一个分支调度电路、以及一个权重等数据的 prefetch 缓存电路，结合编译器来自动分配PE阵列的资源、提前将需要的数据进行缓存，是一个以面积换效率的工作。

   本文我比较感兴趣的点是作者在对设计的电路进行评估的时候使用的一些方法；例如评估系统的各个模块的面积占用使用的是Synopsys DC；电路仿真与性能评估使用的是一个叫 MAESTRO 的模拟器；这些方法的指导我觉得是很有意义的，找到了关于 MAESTRO 的两篇论文，其中一篇还与 NVDLA 有关，下周看看。

2. 调试学习了 NVDLA 的编译器部分，因为 NVDLA 支持的算子有限，师兄说想让我看看能不能让 NVDLA 不支持的算子在 CPU 上跑。我读了几天的代码，还没有分析的透彻，已经分析了的部分也放在了博客里：https://leiblog.wang/NVDLA-Compiler-Analysis/

   总的来说，它接受了 Caffe 的模型之后会把模型转换成 Network 对象，然后将 Network 变成一个 canonical AST、这个就和计算图差不多，canonical AST 又会被转化成 engine AST、engine AST 是最关键的 IR，和 canonical AST 不同的是它会包含硬件的信息，比如 Lenet5 第一层卷积在 canonical AST 上单独的一个节点，到 engine AST 上的时候会变成 conv 和 sdp 两个节点，因为 conv 是一个引擎，还有一个加 bias 的操作是用 SDP 引擎去做的。之后的量化、融合，都是根据这个 AST 做的各种变换。代码里还有一些设计模式的理念，比如工厂模式。

   ![image-20210627194944502](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210627194944502.png)

   把不支持的操作用 CPU 计算我觉得是可以的，问题是它这个前端只支持 caffemodel，像 yolo 是没有caffe版本的就无法解析，工作量太大。之前调研过的 onnc 因为是支持onnx的模型的就没这个问题，但是他的nvdla_small版本需要商业授权，我发邮件过去也没人回我，我又发邮件给一个买过他们商业授权的团队（ITRI），他把我的邮件转发给了他认识的 ONNC 的工程师，但还是没回我。但是 ONNC 这个团队在两个月之前还在 IEEE 上发了篇论文。。。

   

## 20210523

1. 花了两天在 Arm A9 上把 Caffe 编译出来，将论文最后缺失的数据补全了：

   ![w](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210523213136884.png)

2. 论文写好之后发给我们这里的负责的导师，又先后改了两版。就是给老师的发过去，反馈有点慢，有一天多的延迟。。在这个时间里，重构了一下自己的博客。
3. 最近在看《计算机体系结构：量化研究方法》第六版的，原先那本讲蜂鸟处理器的，因为现在不调研RISCV了，就先不看了。

## 20210516

**本周的工作：**

1. 学习了一下 ONNC 的使用，在 253 上测试了一下他们的编译器和 VP 环境，感觉他关键的，可以给 FPGA 使用的 nv_large/nv_small 没有开源出来，关于他们在issue里提到的商业授权，我也没找到入口。没办法运用到自己的项目中来

2. 以前问那个日本研究者要的（这个有点巧，上周发现的使用 ONNC 推理 YOLO 的项目 [ITRI YOLO](https://ictjournal.itri.org.tw/content/Messagess/contents.aspx?&MmmID=654304432061644411&CatID=654313611227352166&MSID=1037365734414623040)  也是他们），用 NVDLA small 配置推理 yolo 的项目的镜像，我烧到SD卡，查到师兄采购的 ZCU 102 上发现跑不起来。根据我在 Xilinx 论坛上的提问，发现是因为：https://forums.xilinx.com/t5/ACAP-and-SoC-Boot-and/Booting-ZCU-102-from-SD-Card/td-p/926649

   现在市面上的 ZCU 102 是 rev 1.1 的，老版本的 ZCU 102 是 rev 1.0 的，1.1 把 ddr 升级了，而当时那个项目构建的时候使用的是 1.0 的 bsp 包，所以跑不起来。但是把镜像文件烧录到他的 sd 卡里能够看到他的文件系统，可以看到他的代码，但是我逛了一圈里面没有源代码，也是预构建好的可执行文件，但是看他那个样子，应该是硬写寄存器来做推理的。

   如果要修复的话，思路是有，但还是有些麻烦的，这个坑以后有机会再踩吧。

3. 结合浩博师兄的建议完善了一下毕业论文，现在毕业论文基本完成写作，还差一点数据：

   1. 首先学会了静态时序分析的基本方法之后，我为论文加上了 Timing 报告分析：分别生成了 10 Mhz 、 25 Mhz、50 Mhz、75 Mhz、100 Mhz、150 Mhz 的 Timing 报告和功耗报告，发现在 Zynq 7045 上最高是可以跑到 100 Mhz的、但是再提高到 150 Mhz 就会出现 Slack。而 ZCU 102 那块板卡，只能跑到 75 Mhz，100 Mhz 的 Timing 都过不去，这一点也在之前有人发表过的论文上得到了证实。
   2. 精度损失情况，这个有点麻烦，因为训练和量化的精度当时没有保存，还要再跑一轮。
   3. 运行速度对比，关于这个，我还要在 A9 处理器上编译一个 Caffe 、 推理看看速度。

## 20210509

**本周的工作**：

1. 学习邱志雄老师开的课[《数字集成电路静态时序分析基础》](https://www.bilibili.com/video/BV1if4y1p7Dq)，学习了 TCL 语言，还有静态时序分析的基本概念，在Vivado里添加虚拟时钟约束来生成时序报告等。
2. 🤦‍♂️另外本周女朋友的毕业设计没做出来，我帮忙看了好几天，快做完了。
3. 我在搜索有没有曾经用 NVDLA 推理Yolo的项目，发现了一个[ITRI YOLO](https://ictjournal.itri.org.tw/content/Messagess/contents.aspx?&MmmID=654304432061644411&CatID=654313611227352166&MSID=1037365734414623040)，在YouTube上还有演示视频，看了一下介绍，他的前端使用的是一个别的工具链 [ONNC](https://github.com/ONNC/onnc)，好像是 ONNX 官方做的，可以接受 ONNX 的模型，然后也会生成 Loadable ，运行的时候需要对原本sw的代码打个patch，对于 NVDLA 不支持的算子，会使用C++实现，所以能够实现Yolo这样的模型。关于onnc，他也提供了修改过sw代码的 vp docker容器，下周试一下编译onnc看看流程。但是看issue，**好像以前 nvsmall 的配置需要申请商业授权，原因是原本NVDLA没有开源出关于small配置的软件代码，所以onnc组织自己写的，比较累。但是不清楚现在是否还需要申请商业授权了。**

## 20210505

**本周的工作：**

1. 首先上星期说到编译出来的runtime接受jpeg文件会失败的问题，我经过调试发现它内部读取图像的时候会做一个 RGB 转 BGR 的操作，根据注释是因为Caffe内部调用的是Opencv，存取图像是以BGR的格式：

   ```c++
   info.out_color_space = JCS_EXT_BGR;                    // FIXME: currently extracting as BGR (since caffe ref model assumes BGR)
   output->m_meta.surfaceFormat = NvDlaImage::T_B8G8R8;
   output->m_meta.channel = 3;
   ```

   这就会造成一个libjpeg库的一个Error。但由于libjpeg是连接库，看不到里面的代码，不知道这样会造成啥问题，所以我就把这句话注释掉了，这样就能接受jpeg的图片作为输入，对精度的影响不大，测了几张图也都是对的。

2. 本周还尝试提高 NVDLA 的工作频率，NVDLA有两个输入时钟，一个是给 CSB 的 csb_clk，一个是工作的 core_clk, csb的时钟给比较低也没问题，core时钟就不清楚了，我添加了信工所的王兴宾博士的联系方式、他们流片过了。他说在 ASIC 仿真里，core_clk 能给很高，可以到1Ghz，我就在 FPGA 里给了 500Mhz，结果是读写某些寄存器的时候都会卡住，系统会挂。**然后我学习到了原来即使 ASIC 仿真里能给很高的时钟，不意味着在 FPGA 验证上也能给一样的时钟**，对于 NVDLA，我发现我上周能正常工作的 100Mhz，Timing报告已经不是很好了。

3. 把 NVDLA 在 FPGA 上 Map 的一些技术点整理了一篇博客。

4. 写毕业设计的论文

**下周的工作：**

1. 改 Runtime 的代码，使其运行的时候能够推理多张图片再看看耗时。
2. 继续写论文。



## 20210425

本周，延续上周的计划，学习了如何把 NVDLA 的中断映射到已经移植到开发板上的Ubuntu系统上。在这个过程中我学会了使用 petalinux 工具从头新建一个内核驱动程序，并且发现了以前把NVDLA的KMD程序挂载到开发板上的一系列问题：

1. `compatible`属性不对应，这里网上所有的教程都是老版本的，现在NVDLA主分支的`compatible`值与以前不同了，如果不改`insmod`的时候就没什么效果。
2. `reversed memory`，这里 ZYNQ 7000 和 ZYNQ MPSoc 也不一样，参考了 Xilinx Wiki，把这里也改对了。
3. 我用的kernel版本比官方的驱动程序要老，里面有个初始化dma的函数，返回的值两个kernel居然是相反的。。。这会导致初始化失败，还好翻issue翻到了类似的问题。
4. 以前说过的32位处理器的内核程序里不能进行64位的除法问题也需要解决一下。

然后就能把nvsmall的kmd程序挂到处理器上了，之后需要进一步编译runtime，runtime需要的链接库只有libjpeg.a一个，幸好之前移植了Ubuntu操作系统，有包管理工具特别方便，于是我自己把libjpeg编译了，他用的是libjpeg6。

于是，能够用nvdla_runtime自动推理了：

1. 首先我运行了一下Lenet5，推理了几幅图都是正确的，推理一幅图的时间是280ms左右，这个与我上学期用HLS写的加速器速度差不多。
2. 之后，我运行了一下量化过后的resnet18-imagenet、好像是因为内存不够，不能运行。
3. 然后，我测试了一下resnet18-cifar10、可以运行，但是libjpeg库有一些问题，我必须把图像转成单通道的pgm才可以进行推理，虽然结果是对的，具体的问题需要debug调试。但是推理一幅图的时间居然只要300ms左右，resnet18比lenet5明明复杂很多倍，看来花在访存的时间比较多。
4. 我看一篇论文：The Implementation of LeNet-5 with NVDLA on RISC-V SoC 。这个能跑到500Mhz，一幅图只要200us，而我上周复现的yolo加速器最高只能到150Mhz，这次实现的NVDLA只给了100Mhz，看来还能更快（只是不知道这个最大频率是怎么测试出来的，我只知道理论上的最大工作频率可以通过环路边界算出来）。

然后，我刚上板成功，本校要求28号之前要上传毕业论文初稿，于是学习了Latex，使用的是ucasthesis这个框架，还在写。

## 20210418

**本周的工作**

1. 在253上制作了一个chipyard的docker镜像，chipyard是伯克利的一个项目，集合了riscv-tools/rocketchip等等等等的工具包，可以在上面运行rocketchip的方针，很方便。学习了一下chipyard如何仿真，如何编译出rtl，想调研这个项目是考虑存在烧写 rocketchip + nvsmall 的可行性，之所以在253上编译，是因为其仿真很吃内存。和zynq器件一样，在这个环境下读写寄存器是可行的。但是runtime需要在rocketchip上运行riscv-linux，这需要生成SD卡接口来存放BOOT和Image文件，很明显ZYNQ器件的开发板没有直接约束在PL端的SD卡接口，所以这个项目放弃了。

2. 认识了两位使用NVDLA作为毕业设计的同学：同学A是基于我上述的chipyard仿真来的，他也只打算跑仿真不打算上板，但是他对NVDLA的寄存器配置，数据在内存中的排布理解的比较透彻，并且其在本周实现了lenet的寄存器配置，仿真时间是几分钟左右。同学A对FPGA不是很熟悉，预计下周交换一下成果，我帮助他在开发板上测试一下速度怎么样。**同学A和我一样认为使用寄存器硬配置整个网络有两个主要缺点：a. 需要配置的寄存器数量太多 b. 要手动把二进制存储的的权重数据塞到内存里 **同学B是做异构计算的，用OpenRISC 1200做主控，NVDLA似乎只是拿来用一下，还在调试，没有什么成果。

3. 本周阅读了一下这篇Paper：["Research of Scalability on FPGA-based Neural Network Accelerator"](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFDTEMP&filename=1019228234.nh&uid=WEEvREcwSlJHSldRa1FhdXNXaEhoOGhUTzA5T0tESzdFZ2pyR1NJR1ZBaz0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4IQMovwHtwkF4VYPoHbKxJw!!&v=MjE5NTN5dmdXN3JBVkYyNkY3RzZGdFBQcTVFYlBJUjhlWDFMdXhZUzdEaDFUM3FUcldNMUZyQ1VSTE9lWnVkdUY=)

   使用HLS实现了一个加速器，能跑yolov2，对应的项目也开源了：https://github.com/dhm2013724/yolov2_xilinx_fpga

   我在zynq7045板卡上将其复现。之所以调研这个项目，是想学习怎么在嵌入式操作系统中控制PL的设备，答**案是MMIO(Memory-mapped I/O)，使用这个技术可以使系统设备访问起来和内存一样。** 

4. 知道了MMIO技术，我开始动手实践上上周周报里的想法，即自己动手解析loadable，然后利用这些结构体模拟runtime，自动配置寄存器。首先在板卡上烧了一个petalinux，但是其没有包管理工具很不方便，于是学习了一下怎么制作ubuntu，解决方案是更换文件系统，这样就可以在板卡上运行ssh/apt等工具了。

   然后是长达几天时间的移植，把原先在SDK上跑的C语言代码移植成C++的，然后和上上周写的Parser对接，在这个过程中还分析了一下kmd的运行流程，如下的思维导图：

   ![execute task](http://leiblog.wang/static/image/2021/4/execute_task.png)

   在我的努力下，在SOC上已经可以配置整个workflow的2/3的工作了：

   ![a](https://leiblog.wang/static/image/2021/4/CBCA2D625B18949F0213B08DB1A4DE85.jpg)

   剩下的1/3主要是还有两个问题没有解决：

   1. nvdla_parser这个项目还没有从loadable里拿出weight的数据
   2. 上了操作系统之后，原本NVDLA会产生的硬件中断被操作系统屏蔽了。。。。这里有两个想法，一个是用寄存器查询来代替，一个是调研一下能不能拿到这个中断。

**下周计划：**

1. 从loadable里解析出权重数据
2. 解决一下中断的问题

## 20210411

1. 本周看完了《The Chisel Book》，把上面的项目基本实现了一半，感觉chisel和verilog建模电路比起来区别还是挺大的。
   1. 首先网上有一些chisel的教程，但是他们的chisel版本是不一样的，这居然会导致教程里的很多语法（尤其是在写scalatest的时候）完全不一样了。所以有时候运行别人的项目，需要安装对应的chisel版本以及对应的scala版本，索性使用idea能够自动cover这个问题，但这样会造成机器上安装了好几个版本的chisel和scala，我觉得这是一个小缺陷，兼容性不太好。
   2. 再记录一点学习chisel的感觉，我原本以为chisel是和hls一样的东西，但实际用过发现完全不一样。之前不久xilinx把hls开源，是基于的llvm实现的，hls和chisel的关系应该是可以使用hls增加生成chisel的feature，和verilog是同一等级的。chisel开发起来效率会高很多，但学习曲线有点陡峭，首先要理解特质、隐式转换等scala的特性才能把chisel玩的很溜，而且比verilog要抽象很多，有时候不知道自己写的是个什么东西，暂时不像verilog那样有建模的感觉，总之我的chisel用的还是一点都不熟练，还需要多做项目提升一下。
2. 本周还买了一本《手把手教你设计CPU——RISC-V处理器篇》，在读。
3. 进一步了解了rocket-chip，它在官方的仓库里，有给rocc的接口那边挂载nvdla的加速器的代码，但我很惊讶google不到多少条有关的信息。

## 20210405

1. 把上学期用PYNQ做的那个手写数字识别的项目写了个简单的文档，然后在Github上Public了：https://github.com/LeiWang1999/Pynq-Accelerator

2. 本周主要写了一个项目：https://github.com/LeiWang1999/nvdla-parser ，对应的更新了一篇博客：https://leiblog.wang/NVDLA-Parser-Loadable-Analysis/ 

   上周提到nvdla的loadable文件利用flatbuffers来将对象进行序列化和反序列化，这篇博客记述的是我使用Flatbuffers的C++接口读取了由compiler编译而成的loadable文件进行的分析，把前面提到的几个比较重要的结构体都提取出来了，然后封装成了一个Parser对象。

这样，我觉得就可以在32位的机器上实现一个简单的工作流程：首先通过Parser读取loadable里的网络的配置信息，然后通过上上周修改的kmd的代码接受Parser里的配置信息，就可以正确的读写寄存器，剩下的就是要把输入图像塞到正确的位置。

但是，这个方案做起来有两个问题：

First. loadable文件放在哪里？例如我用compiler生成了针对nvsmall的fast-math.loadable，但是在sdk上，因为arm的裸机没有文件系统是不能读取本机的文件的。这里据我所知有两种解决方案，都需要使用到SD卡，一种是利用SD卡做文件系统，把fast-math.loadable文件放在SD卡上，另一种是在SD卡上构建Linux（Ubuntu）。

Second. 上述的工作流程，Parser部分的代码是C++实现，移植kmd的代码是C语言实现，在sdk上尝试使用extern "C" 等各种来解决，碰到了很多问题，Parser的程序我也想过使用C语言写一份，但是flatbuffers没有提供友好的C语言接口。

综上所述我认为下周需要研究一下上一个操作系统。

## 20210328

本周的主要的时间花在scala上，基本掌握了其语法和一些性质，并且刷了一些codewars上的题目，我觉得已经具备学习chisel的基础了，下周学习chisel。

上周提到在`ZYNQ 7045`上的ARM处理器上使用裸机读写NVDLA寄存器的方式来做的尝试，问题在于需要配置的参数过多，而怎么生成这些参数，以及需要疏通权重和图像等数据在内存中的存放是这个方案的两个痛点，而关于第一个问题，上周的周报里有提到一个github项目：https://github.com/flw-1996/flw 内有解析loadable的方法，这周我把这个项目跑了起来。

理解了一些loadable文件是如何组织的，可以通过nvdla/sw仓库下提供的和loadable有关的头文件来从文件中读取loadable对象，并且通过阅读他的代码学习到loadable的组织方式其实是Google的flatbuffers，这是一种类似protobuf的数据序列化工具，所以在C++侧可以使用这个工具来解析，主要数据都存储在loadable->blob里，例如blob的第一块数据内存放的就是上周我给出的`engine->network`结构体的内容。

此外，上周说要自己制作pynq镜像、失败了。我原先的petalinux为了不影响本机环境安装在docker里，因为本机是ubuntu20.04，安装老版本的petalinux坑有点多，而官方手册给出的构建pynq镜像的方法在环境上不仅需要petalinux还需要vivado全家桶，再执行它的脚本生成，于是我直接把ubuntu20.04格式化换成18.04了（但考虑到后续可能会有新板卡，这个坑就先不踩了

下周计划：

1. 学习flatbuffers的使用方法
2. 学习Chisel

## 20210321

1. 根据上周所说的，我找到了riscv+nvlarge的项目：https://github.com/sifive/freedom 这个项目里有一些riscv的chisel项目，其中一个是使用rocketchip+nvlarge的项目，我按照教程生成了相关的rtl，但我发现large版本的nvdla的lut消耗有四十多万个，板卡不够。其次，zynq板卡上的qspi等都是固定在ps端的，riscv的qspi没办法约束到pl去，（这个方案可能可以在amazon的FPGA云上实践，但这个方案我一直没有实践，一方面是因为需要购买amazon的账户，使用服务器应该也需要付费，另一方面不利于本科生毕业设计的实现（需要作出一定的效果））。

2. 本周意外的发现TVM前几次的commit把USE_LLVM的开关默认设置成ON了，这导致在`runtime only build比如编译VTA(我在水群的时候发现有人按照官方的教程不编译，帮忙看了一下)的时候失败，因此给TVM水了一个PR（https://github.com/apache/tvm/pull/7657），还和陈天奇大大简单互动了一下。

3. 经过这两三周的探索，我觉得我主要输在板卡上，导致没有一个能跑起来的demo。期间我还调研了一些项目，例如DAC SDC 2019的亚军，西安交大用的PYNQ实现的类似NVDLA的设计(https://github.com/venturezhao/XJTU-Tripler)，并且做了RSIC的指令集，结果我发现2019换了板卡，用的是Xilinx的Ultra96系列，处理器是64位的A53，与上周移植petalinux有相同的困难，并且它核心的运算部件mpu和vpu是网表文件，没有开源成rtl。

4. 本周还尝试使用nvsmall直接读写寄存器，我把sw/kmd也就是用户和系统内核的接口那部分的代码移植到了Arm处理器上，也就是用`Vivado SDK`开发。里面对寄存器的定义非常清晰，但我在阅读源码的过程中越发感觉到hw的master分支是写到一半突然不写的感觉，不明白sw对应的到底是hw的哪个版本。

   1. 例如我使用官方给出的sw的寄存器地址头文件(他给了两个头文件，`opendla_initial.h`、`opendla_small.h`，按理来说应该使用nvsmall的时候需要使用opendla_small.h这个头文件)，测试读取`NV_HW_VERSION`这个寄存器，使用small的头文件读出来的内容是不对的，而使用initial的结果是正确的。

   2. 在master分支发现，官方自从某个commit开始，生成的rtl就只能支持int8，不支持fp16的rtl生成，并且阅读issue(这里忘了是哪个issue了，是[redpanda3](https://github.com/redpanda3)提的)之后发现，其实官方给出了那么多spec的定义文件，能够正常work的只有nvlarge和nvsmall的版本，其他的都要修改部分，然后这个大佬还自己写了一份chisel版本的nvdla，也开源了。

   3. 我问师兄师姐们要来了lenet_nvfull版本的log，对照着代码基本理解了其思想，umd（runtime部分，）接受的nvdla loadable文件和img等共同生成了一个`engine context`,这里面有几个关键的结构体：

      - engine->network,里面定义了网络的基本信息，例如网络对应的几个OP(六个processor对应了BDMA、CONV、SDP、PDP、CDP、RUBIK)这些操作出现的位置，例如lenet的只有conv、pooling、relu，第一层layer是conv、第二层relu、然后pooling，那么network的结构如下：

      ```c++
      // Custom Config
      int
      network_config_lenet(struct dla_network_desc * network){
      	network->operation_desc_index = 6;
      	network->surface_desc_index = 6;
   	network->dependency_graph_index = 5;
      	network->lut_data_index = 8;
   	network->roi_array_index = -1;
      	network->surface_index = -1;
   	network->stat_list_index = -1;
      	// network->reserved1 = [ -1 , 0 , 1 , 2 , -1 , -1 ];
      	network->op_head[DLA_OP_BDMA] = -1;
      	network->op_head[DLA_OP_CONV] = 0;
      	network->op_head[DLA_OP_SDP]  = 1;
      	network->op_head[DLA_OP_PDP]  = 2;
      	network->op_head[DLA_OP_CDP] = -1;
      	network->op_head[DLA_OP_RUBIK] = -1;
      	
      	network->num_rois = 1;
      	network->num_operations = 10;
      	network->num_luts = 0;
      	network->num_addresses = 10;
      	network->input_layer = 0;
      	network->dynamic_roi = 0;
      	network->reserved0 = 0;
      	
      	return 0;
      }
      ```
      
      - 此外还有三个比较重要的结构体、consumer、surface、operation，分别代表每个index(指网络需要的操作数，conv+relu+pooling算三个index)的时候网络的配置信息，比如卷集核大小之类的，还有网络的存储位置，**每一个index需要配置的参数可能有将近50个！！！**
      - 这些参数根据原代码的逻辑原本应该是由DMA从内存中自动搬运，因为img和loadable（里面应该包括了权重等信息）已经由runtime搬运到内存里去了，于是我自己把这个逻辑重写了一下，变成每次index取我自己定义的结构体数组。每个结构体的成员的参数应该如何设置？一方面可以根据自己的理解，例如已经清楚的卷积核大小等已知的网络信息，另一方面可以对照着debug模式跑出来的log也可以确定，其实在确定这些参数的时候，我发现了一个好玩的项目https://github.com/flw-1996/flw `nvdla_depicter` 它可以模拟runtime，然后把这些结构体都输出，我是对着这它跑出来的log简单配置了网络，但是工作量巨大，我只简单实现了一层conv，寄存器的配置顺序与log是一致的。这还有一个问题，就是weight和img的数据应该如何读取？我设想在SD卡里放置caffemodel，然后再运行一个Parser，但感觉这个步骤太复杂了，如果可以在板卡上构建一个**PYNQ**是再好不过的。
   
   综合以上，我发觉又有数不清的东西需要去学习，例如Chisel，NVDLA/HW的工程我觉得太乱了，想custom需要安装过多的软件包，如java、perl...过于复杂。前文中有提到了一个chisel实现的nvdla，我觉得这才是正解，只需要掌握chisel就可以了。
   
   下周先暂停瞎折腾NVDLA，做两件事：
   
   1. 学习Scala、Chisel
   2. 尝试移植一下PYNQ

## 20210311

本周根据网上找到的一篇文章 https://vvviy.github.io/2018/09/17/nv_small-FPGA-Mapping-Workflow-II/ 根据这篇文章我尝试在上周的工程上看看能不能在petalinux上把sw里的内容运行起来。

在petalinux安装过程中就碰到了一些问题，我使用的是vivado2020.1,但是vivado2019.2之后export到sdk的文件就从原本的hdf变成了新的xsa，虽然用起来感觉更规范一些，但是很遗憾不能被2019.2以前的petalinux版本支持，为了保证和教程中的步骤走（因为不同版本的petalinux支持的linux内核版本不同，也就造成了sw那个仓库里有些函数(sw的kernel版本是4.13)在对应的内核版本已经废弃或更改，但我对linux内核编程也不熟，所以为了避免踩坑，我把vivado也换成了2018版本。）

然后成功生成了镜像，但是在`insmod opendla.ko`的时候,出现了下面的error：

![](http://leiblog.wang/static/image/2021/3/A7EB3EEB28E45AE3A72B0BF6BE37DBF9.png)

我搜了一下，是指32位处理器使用64位除法的时候回报这个错。这时候我才明白为什么走这条路线的教程都是用的`zynq +UltraScale MPSoC`，和我使用的ZYNQ7045的不同在于其处理器是64位的A53，而ZYNQ7000系列的处理器是32位的，这也就带来了很多问题。。

我发现kmd的代码中只有一行代码是用到64位除法的，及获取时间戳，再除个1000获得纳秒的数量级，在debug模式用来输出运行时间用的，解决方案自己用32位的数据类型写一个64位的除法，我直接把这个除1000删掉了。

这样，insmod果然能够正常工作，进入umd的目录下面，开始make runtime，出现了浩博师兄之前跟我说的，有的代码是调用的连接库，比如我就失败在了libprotobuf.a这个库上，他是给的64位的已经编译好的库，也许我在linux上重新编一个能够解决，但我预感到之后还会有很多坑，于是放弃了对petalinux上的测试。

但好在，虽然ZYNQ7045的处理器有点拉垮，但是LUT的资源很多，下周开始尝试浩博师兄说的，把RISC-V也烧进去的解决方案！

## 20200307

1. 本周首先在实现了cifar10-resnet18网络的量化，该网络的量化假期在家里的时候一直没成功，回学校之后发现可能是因为量化时候给的校准集和训练用的数据集不一样的锅，把校准数据集改成从训练集里sample了若干张图片后，网络work了。**但过程中发现如果使用per-kernel编译结果会失真，表现为无论给什么输入，输出都不变，但使用per-filter编译就能很好的work，但之前训练的针对MNIST和IMAGENET的两个模型在这两种编译模式下结果给相同输入则输出是相同的，这一点有点迷惑，还没了解这部分编译的机制。**
2. 浏览了一些Paper，运用tanh、leakyrelu等存在负数输出作为激活函数的CNN网络，帮肖航师兄找的，感觉这个方向灌水蛮严重，分类网络的话一般都是用RELU或者Sigmoid了，找到的几篇也都是用在奇奇怪怪的领域，什么COVID-19检测，波谱图分类之类的，或者是比较不同网络使用不同激活函数的效果。
3. 学习了NVDLA的Hardware部分，首先根据官方的手册Make了一份nvsmall的RTL代码，刚开始综合占用的LUT有四十多万个，后来发现是因为其中的RAM是行为级描述，其实issue里也有，把这部分ram替换成BRAM资源就可以了，然后再综合LUT有7万多个，但我身边的板卡都是一两千的，LUT顶多7w。之后联系我们这的学院老师报销了一块ZYNQ7045的第三方板卡，解决了资源不够的问题，之后还有很多技术点，例如要globaldefine几个宏，关闭clockgating，还要把CSB接口封装成APB在转AXI、要补几根信号线。之后，NVDLA和ZYNQ器件的BlockDesign能够成功生成比特流文件，但没在SDK上验证是否能够work，因为对相关的寄存器还不太熟悉。

下周任务：

1. 在ZYNQ上建PetaLinux，看看能不能编译出nvdla_runtime

## 20210228

1. 过去的几周没有周报需要，我在刷一些题目，codewars打到了4kyu、巩固了C++的基础，还打了一会儿hackthebox。

![](https://www.codewars.com/users/LeiWang1999/badges/large?logo=true)

2. 买了一本《LLVM编译器实战教程》回来看，学习了一些LLVM和CLang的知识。

1. 今天返回校园，下周需要完成的目标
   - 完成cifar10-resnet-int8的量化
   - 帮助肖航师兄找到一个运用负数输出的激活函数的CNN网络
   - 学习一下NVDLA HW部分。

## 20210202

量化之前在cifar10上的预训练的resnet-18网络失败，在Tensorrt端验证量化后的准确率接近90%，但是网络在./nvdla_compiler中指定int8量化，会报error:https://github.com/nvdla/sw/issues/211

经过调试以及对比可运行的其他网络，我发觉该Bug可能是因为紧跟着数据输入的第一层卷积层的BatchNormal的参数过大，因为之前训练cifar10为了方便我去除了输入图像的transform，及输入图像除256归一化到0～1之间，以及减去图像均值。

> 这里需要表明，之前为了方便是因为以前没有注意到./nvdla_runtime 有normalize和mean这两个参数：
>
> 例如，我们把输入的图像归一化到0～1之间，并减去均值128，就可以这么写:
>
> ./nvdla_runtime --loadable cifar10-fp16.nvdla  --image resnet18-cifar10-caffe/
> Image/raw/175_4.jpg   --normalize 255 --mean 128,128,128 --rawdump
>
> 其中normalize是uint8，所以最大值是255，公式如下（这些是读源代码的总结），而这两个参数只对fp16的模型管用。
>
> ```c++
> if (outTensorDesc->dataType == NVDLA_DATA_TYPE_HALF)
> {
>   half_float::half* outp = reinterpret_cast<half_float::half*>(obuf + ooffset);
>   // 这里是公式
>   *outp = static_cast<half_float::half>((float(*inp) - float(appArgs->mean[z]))/appArgs->normalize_value);
> }
> else if (outTensorDesc->dataType == NVDLA_DATA_TYPE_INT8)
> {
>   char* outp = reinterpret_cast<char*>(obuf + ooffset);
>   //*outp = char(*inp); // no normalization happens
>   // compress the image from [0-255] to [0-127]
>   *outp = static_cast<NvS8>(std::floor((*inp * 127.0/255.0) + 0.5f));
> }
> ```

这里只要采取其一就好，于是我加上了减去均值的操作解决了该BUG。

但是，经过量化过后的网络存在严重的失真现象，测试了若干张图像就没有一个对的，甚至大部分数据的输出都是一致（与不指定输入图像的输出相同）。

下面开始排错：

1. 有没有可能是因为我量化调用的是PythonAPI，与lenet当时用的TensorRT提供的C++ DEMO不同？

   于是我用Python又进行了Lenet的量化，经过测试，两者是有细微的差异，但是用Python接口的量化后的calibtable进行推理，不会出现失真。

2. 有没有可能是因为上周写的脚本calibtable的python脚本出错？

   在第一个问题的验证中，证明该脚本无错。

3. 我怀疑是因为指定int8的时候，输入的图像被经过特殊处理，因为在上文截取出的代码：

   ```c++
   else if (outTensorDesc->dataType == NVDLA_DATA_TYPE_INT8)
   {
     char* outp = reinterpret_cast<char*>(obuf + ooffset);
     //*outp = char(*inp); // no normalization happens
     // compress the image from [0-255] to [0-127]
     *outp = static_cast<NvS8>(std::floor((*inp * 127.0/255.0) + 0.5f));
   }
   ```

   输入图像经过二分之一的压缩，我觉得这个操作很迷，但是runtime必须在buildroot环境下才能运行，不能像compiler一样debug。如果是这个bug，那么lenet能够work就可以解释，因为他其实只有0和255两个值，输入图像是黑白的，无论怎么缩放都不会出问题。

于是为了验证是不是第三个原因，我对之前训练的resnet18-imagenet的网络进行量化，但这次的量化结果很好，于是为什么resnet18在cifar10上不work，这个问题可能要学会调试runtime才可以解决。

另外，本周还训练了一个lenet网络，把之前的relu层换成了sigmoid层，



## 20200131

本周的科研工作着重于NVLDA的int8推理，踩了和填上了若干个坑。

首先，我修改了一下`nvdla/sw`中./nvdla_compiler的makefile使其可以进行单步调试，在调试的过程中我发现如果不提供--calibtable参数，只指定编译int8的化，它内部是有一个简单的量化算法的，但经过测试量化的效果很不理想、测试的resnet18和lenet5两个模型，除了输入层的参数保持在1，其他层的参数都被量化到127？可能这个算法没有完善吧。

之后我开始了解calibtable是什么东西，它的全称应该是calibration table，是tensorrt做int8量化的时候会生成的校准表。但要是要单独安装一个tensorrt有些麻烦，于是我先尝试使用：https://github.com/BUG1989/caffe-int8-convert-tools ，该库模拟了tensorrt的KL量化算法，可以生成calibration table文件，但也几年没维护了，**运行起来有一些bug**，新库已经合并到了Tencent ncnn、还好我也在qq群里，和作者（圈圈虫）交流后解决了这个库问题之后，我发现./nvdla_compiler需要的calibtable参数，接受的是一个json文件，而calibration table 转json的脚本放置在sw/umd/utils里，**但这个脚本也有坑**，在脚本的开头指定了应当用python3运行，但我阅读代码之后发现，居然用的python2的语法，并且只支持tensorrt生成的calibtable文件转换。

于是我在253上build了一个tensorrt镜像，以lenet5网络为基础，实现了网络的量化（这部分量化的算法更高级，需要数据集作为数据，貌似是以数据集为基础做的量化，精度保持的蛮高），**但运行脚本转换的时候又发现了一些BUG！**

这个bug可以看nvdla/sw的issue：https://github.com/nvdla/sw/issues/201

该issue去年6月就有人提出了，但是现在都没有解决方案，我自己对比官方提供的可以量化的resnet50，自己写了个脚本解决了这个问题，于是int8的lenet5网络能够在runtime里跑出正确的结果了。

![](http://leiblog.wang/static/image/2021/1/oBdNEC.png)

所以calibtable的生成方法，首先需要在tensorrt环境下生成对应的calibration table、并且能够预估出量化后网络的精度，之后在nvdla环境下进行calibration table到json的转换，最后在caffe环境下，利用自己写好的脚本对转换出来的错误的json文件进行校准，然后在nvdla环境下进行compile和运行。

此外，本周还完成了智能计算系统的第四章课后作业。

下周安排：

量化cifar10和resnet18、imagenet的resnet18网络，其实还是蛮复杂的，这个过程中我又发现了很多坑

1. 原先训练的cifar10网络的batchnormal层的配置不能被compile，我今天debug了半天发现了问题，修改了一下，这个问题已经修复。
2. 原先训练的cifar10网络的transorm里有scale、会把图像缩放到-1~1之间，会造成各种各样的麻烦问题，我现在把这个scale删掉，正在重新训练。
3. imagenet的网络是从网络上找的现成的，但如果要用tensorrt做量化、那么需要训练的数据作为辅助、可能还是得重新训练。

## 20200124

1. 本周协助肖航师兄训练了针对imagenet数据集的resnet18网络，针对上周的cifar10的resnet18做了一些修正工作。并且将测试模型阶段利用Caffe的Transformer对图像进行预处理、然后验证的一些方法整理记录成了一篇博客、
2. 本周持续学习智能计算系统，视频课快看完了、了解了书中的深度学习处理器的一些设计理念和方法、以及设计智能硬件的时候需要考虑到的性能、仿存优化工作等，完成了第三章的课后习题。

## 20200117

1. 本周协助肖航师兄使用caffe训练了针对cifar10的90%精度的resnet18网络。
2. 完成了本科阶段的课程设计4的报告和答辩工作。
3. 学习《智能计算系统》，内容有生成对抗网络的基本概念、训练方法、SVM支持向量机


## 20200110

   1. 本周完成了本科阶段的课程设计4，在这次设计中，我基于上周调研的通用加速加速器设计方案，利用PYNQ-Z1做了一个手写数字识别的简单系统:首先是写了一个Web应用、可以通过鼠标手写数字：

   <img src="https://leiblog.wang/static/image/2021/1/dHi9t4.png" alt="img" style="zoom: 33%;" />

   然后通过以太网把相关的参数和图片传输给PYNQ，PYNQ可以分别使用CPU和使用FPGA进行推理，可以显示准确率和耗时、这一套流程可能会应用于本科毕业设计的展示。

2. 本周正在协助肖航师兄利用caffe训练一个针对cifar10的网络，考虑到以后caffe会常用向浩博师兄申请到了GPU集群，制作caffe的docker镜像、其中碰到了很多问题：caffe官方提供的makefile不支持V-100的卡、默认的python接口python2，经过了一两天的折腾，成功解决了这些问题，在253上制作了针对v100卡和python3接口的caffe镜像。

## 20210103

针对毕业设计题目《深度神经网络硬件加速系统设计与优化》，调研了现有的一些有关的开源工作的解决方案。

1. https://github.com/awai54st/PYNQ-Classification

   该项目做的比较完善，能够把通过caffe或者theano预训练的模型导入，然后通过利用hls制作的硬件加速器进行推理。

   这样通过HLS制作的加速器虽然不是通用加速，及每针对一个结构不同的模型都需要重新更改hls代码，但作者提供了自己设计的cifar10、lenet-5两个模型的hls代码，在本周调研的第四个项目里，实现了通用的卷积池化全联接的加速器，但相对于专用加速器来说，又存在着一些问题。

   调研该项目的时候碰到了一些问题：

   - 需要在pynq上编译caffe和theano框架，但安装的过程需要增加pynq的ram空间，作者在readme里指出需要外接存储扩大swap 的空间，但讲的比较含糊，暂时还没有能够解决。

   - 作者给出的hls程式中，不同工程中通用的代码，比如卷积和池化、貌似(这部分还有待商榷，得等我明天去实验室研究，我用vscode打开主程序寻找定义，会被定向到syn/systemc的源程序里，但我觉得这应该是hls自己生成的，还没仔细研究)是使用systemc编写，对systemc的语法我比较生疏，以前没了解过。

2. https://github.com/wbrueckner/cv2pynq

   该项目是网友写的针对pynq的opencv框架，比直接调用自己从官网编译的opencv速度要快，并且能够应用PYNQ的`base overlay`进行硬件层面的加速。

   - 貌似已经停止运维了，给到的安装教程最新到v2.3,而最新的镜像已经到了v2.6版本，我在v2.5版本的pynq镜像上安装失败。
   - 加速针对的base的overlay，即使倒退了版本成功编译，在自己定制overlay的情况下应该也不好使用。

3. https://github.com/WalkerLau/Accelerating-CNN-with-FPGA

   这个项目是计算所的VIPL组的同学写的、但是部署的板卡是上万元的Xilinx Ultrascale+ MPSOC ZCU10X、对于我身边的板卡而言，本身参考价值不是很大。

4. https://github.com/mfarhadi/CNNIOT

   该项目是一个通用卷积运算的加速器框架，我读了源代码，并且跑通了整个流程。

   他的例子也是一个使用lenet进行手写数字识别，与1刚好有很多不同之处。

   - 对于数据的输入，1中的方案是在pynq上编译caffe和theano、然后读取模型的数据、4的方案就更人性化了、把模型的权重等信息直接保存为npy文件，方便导入。
   - 卷积、池化、全联接层的加速是通用加速，即卷积核大小、通道数量，都可以自定义。

   - 对于输入数据和参数配置，使用的是dma进行传输。

   运行的结果还是不错的，单用sdsoc的arm进行推理，一张图片需要1.2s；调用加速核，一张图片需要160ms左右（但还是很慢啊！

   经过我的思考，这应该是通用加速的弊端、首先不能预先知道卷积核的具体参数，就没有办法提前进行优化。但这个项目总体的思路还是不错的、**下周预计基于这个项目学习并做一些可展示的工作**。

   本周除了以上的调研，还帮助肖航师兄发现了nvdla输入任意图片都会得到相同的结果的bug、实际上是nvdla不支持jpeg格式的图片，另外推进了《智能计算系统》的学习进度，开始完成第三章后的习题了。

## 20201228

1. 本周确定了毕业设计的题目《深度神经网络硬件加速系统设计与优化》。
2. 完成了《智能计算系统》第二章节最后一题的程式编写，及在不使用任何编程框架的前提下，将上周提到的使用PyTorch训练的四位全加器功能的MLP，用C++实现了一份，锻炼了C++程式编写的能力。

下周安排：

1. 完善所写程式的文档
2. 完成校外毕业设计、客座协议的有关工作
3. 要复习期末了

## 20201219

在学习《智能计算系统》的时候，发现课后的习题是没有参考答案的，整理了自己的习题答案和笔记记录在Github：https://github.com/LeiWang1999/AICS-Course 、在博客里也有一份post，详细的记述了一些题目的公式推导，以及编写的相关程式，包括自己构建数据集、使用PyTorch训练了一个实现四位全加器功能的MLP。

本周简单调研了NVDLA和Caffe、协助肖航师兄用caffe构建了两个简单的模型用于Nvdla的debug。

下周安排：

1. 确定毕业设计的具体方向、拟定题目
2. 继续调研nvdla
3. 继续学习《智能计算系统》、完善习题和笔记

**毕业设计题目构思**

1. 基于SDSOC的神经网络硬件加速器设计

> 说明：SDSOC是指ARM+FPGA这种架构的别名，首先，这个题目不指明具体的器件、网络和框架、普适性比较强。
>
> 我的初步想法是使用PYNQ这样的器件、无论是手写也好、用加速器框架生成也好，加速一个网络，然后做一个简单的web页面远程监控PYNQ，传输图像，显示关闭硬件加速和开启硬件加速的推理耗时差异和结果。
>
> 或者做实验室那边的一些项目、这个题目应该也能适应。

## 20201211

很抱歉空了三周没有和老师和师兄汇报，这段时间内忙于各种学生工作与党务，空出很少的时间来学习知识这周刚忙完，首先汇报一下这三周做的一些事情。

和科研无关的部分：11月14日-11月24日，我主持了一项校里会有领导来看的本学院的年度党日活动，督促各支部组织表演、借服装、借场地、借耳麦，做视频等等，累死了。这段时间里，参选了我们学校的大学生年度人物，这个荣誉研究生和本科生都参评最终选10名，评上了有8000元，其实我是冲着钱去的，要做PPT答辩，做个人展示视频，幸运的是以综合评分第一名的成绩评上了。之后被老师要求在迎新晚会上做汇报，接受校媒的采访，在学校一二九表彰大会上做学生代表发言，就占用了一些时间，还有一个星期的毕业实习，去了解了华为的鸿蒙OS开发应用流程，听了华为和龙芯的几位工程师做的汇报。

和科研有关的部分：

前段时间问浩博师兄要了两篇综述，第一篇有关神经网络压缩和硬件加速器的，刚刚阅读完，并且画了一份思维导图：

![](http://leiblog.wang/static/image/2020/12/7IrFTz.png)

在阅读综述的过程中，积累到了一些关键词和知识盲点，其中有比较熟悉的技术如：神经网络架构搜索，知识蒸馏，PE Array。相对简单的技术有**SVD(奇异值分解)**、**群卷积**等我详细了解了一下这些东西有哪些用途以及如何实现，但是像张量稀疏化等搜了一下就能看出来特别复杂的我就没多了解。

另一份是有关机器人的综述，我大概看了一下几乎全是我不懂的词，就没有继续看，最近在看陈云霁老师的《智能计算系统》一书。

## 20201114

编译原理部分：将中间代码生成和代码生成这部分看完了、之后关于生成代码的优化等编译领域的前沿研究方向，短时间内不会再去看，有关编译原理的学习告一段落。

现在每周阅读了哪些文章/博客，都记录在我的博客里

![](http://leiblog.wang/static/image/2020/11/peyftV.png)

关于TVM部分本周在有了编译原理的基础上，在看TVM关于开发者的指南，本周看了两篇：Adding an Operator to Relay ｜ Adding a Compiler Pass to Relay，关于第二篇，学习了了Pass是什么概念：

> Pass - 流程，在llvm里有类似的概念，其接受的是TVM的IR(IRModule)，输出的也是TVM的IR，Pass可以用来优化代码，Pass的输出的和输入相比在功能上完全相同，只是在性能上得到改进。这部分通常是给我们发挥的地方。

其他，我尝试把上次用Pytorch训练出来的网络使用VTA编译，在graph_pack上就报错了，目前还在找出错的原因，期望下周能解决。

## 20201107

编译原理部分：本周继续学习了语义分析和语法制导翻译的内容。

TVM部分：本周较为深入的学习了tvm的reduce操作，reduce的基本用法，以及自定义规约的行为，笔记记录在博客里。

## 20201101

本周主要时间花在学习编译原理上（本周学习内容是龙书第三章的结尾到第四章语法分析的一半，一些语法分析的算法，自顶向下和自底向上，目前停在LR文法上）学习的速度比较慢，因为参考了很多资料（本来国防科大的网课讲得太枯燥了，还参考了斯坦福的编译原理课程，最后还是觉得龙书写的最好）。

另外本周还再学习了VTA，包括VTA的结构、上周学习了HLS加速器的设计之后对VTA的四个模块也更熟悉一点了，简单看了看代码，但我到现在只跑通过官方提供的resnet模型，根据我搜集的资料，VTA的一些问题大概在于：

1. VTA.build之前需要量化模型，这部分是在前端做的，但有时候前端对有些op的量化不支持。
2. VTA对op的支持也有限

上周训练darknet、数据集和label的标注都准备好了，但是没有训练需要的GPU设备，还在和这边的老师借一下。

在学校其余时间,还做了很多与中心科研无关的事情：

1. 校内维护的网站使用的豆瓣接口被豆瓣关闭了，我用node.js重新写了个豆瓣的爬虫，已经上线使用了。
2. 学校要评年度人物准备素材、党支部党日活动策划等等。

希望能尽早做完这些事情，安心科研。

## 20201023

本周完成的工作：

1. 终于在jetson上成功编译了tvm，(迫于校园特殊的网络环境，花了点时间钻研出了让jetson用以太网接到PC，然后把PC的以太网桥接到Wifi让jetson依靠Win/Ubuntu上网的解决方案，对Pynq同样适用，当然这部分与研究无关）
2. 学习了darknet，因为以前帮老师做的那个网络是由学长训练，不知道为啥把darknet转成了tensorflow，以前的模型直接用tvm转化成IR会有错误（这边感觉是给TVM提个PR的机会？），所以准备把这个模型用darknet重新训练一下。
3. 处于了解除了VTA以外的加速器设计，学习了Xilinx2020年的暑期班(北京大学CECA)深度学习加速器设计及优化实验，自己动手把卷积、脉动阵列等加速器都写了一遍。

## 20201018

本周无实质进展，参加了一个星期的电子设计竞赛，现已比完，下周开始继续学习，我今天看了一些文章，内容是关于使用HLS设计加速器的，想学着手动设计一下简单的加速器。

## 20201010

本周先使用Pytorch训练了一个AlexNet网络，然后分别使用TVM和Torch部署，对比了一下速度，真的有不少的提升(记录在本周新写的博客)，然后在jetson nano上编译tvm碰到了一些坑，还没解决。TVM社区本周新出了TVMC这个命令行工具，学习了一会儿，还是蛮好用的。今天是大学最后一次参加电子设计竞赛、带带学弟学妹们，下周目标暂定成功在nano上编译tvm。

## 20201004

本周首先读了TVM在18年发表在osdi上的那片原论文，看完等于是帮助我复习了一个月前在网上听陈天奇老师做的那份报告，但论文里有更丰富图表和细节。我注意到两件事情：

第一个是在验证FPGA的时候用的是PYNQ，并且在Resnet网络的推理上取得了很可观的加速，然后我就想找找看这段的程序网上有没、发现官方给的VTA的唯一一个关于部署到FPGA的Tutorial(Deploy Pretrained Vision Model from MxNet on VTA)里就是做的resnet。

但是这个教程我有很多问号，AutoTuning和推理的代码都是在主机上写的。虽然通过RPC这些工作确实都是在pynq板上运行，但如果用这个思路的话，PYNQ板的推理工作就离不开主机不是么。我觉得流程应该是主机使用RPC进行AutoTuning，然后把tune好的模型(或者加上生成的ip)传到pynq上，然后在pynq端写推理程序，这边我还没搞懂。

第二个是注意到，TVM在论文里做过ARM A53的加速评估，我身边有一块jetson板卡，前几天在本科阶段给我很多资源的老师想让我帮她把一个用YOLOv3做detection的项目部署到jetson上，我想用tvm试一下，这算做学习的一条支线。

除了读了这篇论文，本周还踩了很多坑，学习了一些编译，学习了detection的一些基础知识(比如mAp是啥，iou是啥)，还学习了yolov1到yolov2、v3的一些改进(这部分还没看完)。

本周期望完成的小目标：

1. 挑一个自己以前训练过的模型用tvm推理看看，对比一下速度。
2. jetson nano上编译tvm。

## 20200927

周报汇报晚了，并不是偷懒喔，这一周我每天忙于各种琐事，很少有闲下来的时间。

汇报与科研有关的：

1. 本周每天都有在学习编译原理，这门课有些枯燥
2. 之前用的PYNQ-Z2，使用tvm有些问题(比如每次使用一次rpc之后，都需要重新编译tvm才能使用，这一点论坛上也有一些同样的状况)，于是我和digilent申请了一块PYNQ-Z1

汇报与科研无关的：

1. 本周的本校推免事宜，已获得本校名额

2. 本周受老师要求给新生做了一个多小时的线上课堂

3. 下周我在修的专业写作课程，有一次汇报的机会。我准备讲tvm、花心思做了一份好的ppt。

   ![ppt](http://leiblog.wang/static/image/2020/9/QQ20200927-171808@2x.png)



## 20200918

上周末周报汇报完之后，我学习了如何使用 C/C++/Python 混合编程，然后自不量力的从前端开始读tvm的源码，阅读笔记记录在博客：http://leiblog.wang/TVM-%E4%B8%80-load-onnx/ ，因为读到一半卡住了所以并没有写完。tvm本身是一个编译器，代码中有非常多的编译原理概念，所以我暂时放下了源码的阅读工作开始学习起了编译原理(参考资料为 国防科技大学编译原理网课 + 龙书第二版)，tvm的使用也还在慢慢熟悉。

## 20200911

1. 根据官方指南，使用RPC和VTA在PYNQ-Z2板上运行了一些程序。
2. 和浩博师兄交流了一下毕业设计方向（浩博师兄提议，毕业设计为该框架贡献一些力量(如现在的tvm使用vta直接部署到FPGA，支持卷积比较好，但是对反卷积等其他操作不支持，如果扩展了这些操作，再做一个相关算子的演示系统，比如场景分割或者换脸，就已经很好了)，或者利用该框架实现一些功能，对此我很赞同）
3. 开始阅读了部分TVM源码，但我还没接触过Python与C++混合编程，准备先学这个
4. 下周学习Python、C++混合编程！


## 20200904

本周分别在Windows WSL和MacOS上成功编译了TVM，并根据官方的文档学习了其基本使用，设计理念以及相关算子(Matrix Mul、Padding、 Conv2D等)的TVM实现，目前还在学习TVM基本的张量的操作语法、调度的知识，学习的时候编写的有关程序记录在 https://github.com/LeiWang1999/tvm

另在bilibili看了几份陈天奇本人关于tvm设计理念的报告。

下周我要组织学院学生党支部的一系列会议，学习进度可能会有拖沓。

本周还在肝课程设计！但还会抽空学学tvm。