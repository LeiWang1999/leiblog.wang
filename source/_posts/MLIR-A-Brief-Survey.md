---
title: MLIR | A Brief Survey
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2021-12-01 14:11:35
---

MLIR 是Google在2019年开源出来的编译框架。不久之前意外加了nihui大佬建的MLIR交流群，不过几个月过去了群里都没什么人说话，说明没人用MLIR（不是。现在刚好组里的老师对MLIR比较感兴趣让我进行一下调研，于是就有这篇比较简单的调研报告啦！

MLIR的全称是 **Multi-Level Intermediate Representation**. 其中的ML不是指Machine Learning，这一点容易让人误解。

一些你可以帮助你了解MLIR的资源：

1. [MLIR官网](https://mlir.llvm.org/)

2. 目前，MLIR已经迁移到了LLVM下面进行维护。

   https://github.com/llvm/llvm-project/tree/main/mlir/

3. 如果想要引用MLIR，使用这一篇Paper：[MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054)

4. [LLVM/MLIR Forums](https://llvm.discourse.group/c/mlir/31)

5. [LLVM Discord 在线聊天室](https://discord.gg/xS7Z362)

MLIR SIG 组每周都会有一次 public meeting，如果你有特定的主题想讨论或者有疑问，可以根据官网主页提供的方法在他们的文档里提出，有关如何加入会议的详细信息，请参阅官方网站上的文档。

<!-- more -->

### 一、Background

#### 谁是MLIR的作者？

如果你和我一样曾经担心MLIR会成为Google众多开源到一半又被腰斩的工程之一，那么MLIR的作者是Chris Lattner这一事实可能会打消你的想法。Chris 同时也是LLVM项目的主要发起人和作者之一，Clang编译器的作者，在Apple工作了十年，是Apple开发用的Swift语言的作者。排个序的话，MLIR应该是Chris大佬继LLVM、CLang、Swift之后第四个伟大的项目。

![img](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/261316515121404.jpg)

> 2010 年的夏天，Chris Lattner 接到了一个不同寻常的任务：为 OS X 和 iOS 平台开发下一代新的编程语言。那时候乔布斯还在以带病之身掌控着庞大的苹果帝国，他是否参与了这个研发计划，我们不得而知，不过我想他至少应该知道此事，因为这个计划是高度机密的，只有极少数人知道，最初的执行者也只有一个人，那就是 Chris Lattner。
>
> 从 2010 年的 7 月起，克里斯（Chris）就开始了无休止的思考、设计、编程和调试，他用了近一年的时间实现了大部分基础语言结构，之后另一些语言专家加入进来持续改进。到了 2013 年，该项目成为了苹果开发工具组的重中之重，克里斯带领着他的团队逐步完成了一门全新语言的语法设计、编译器、运行时、框架、IDE 和文档等相关工作，并在 2014 年的 WWDC 大会上首次登台亮相便震惊了世界，这门语言的名字叫做：「Swift」。
>
> 根据克里斯个人博客（http://nondot.org/sabre/ ）对 Swift 的描述，这门语言几乎是他凭借一己之力完成的。
>
> 克里斯毕业的时候正是苹果为了编译器焦头烂额的时候，因为苹果之前的软件产品都依赖于整条 GCC 编译链，而开源界的这帮大爷并不买苹果的帐，他们不愿意专门为了苹果公司的要求优化和改进 GCC 代码，所以苹果一怒之下将编译器后端直接替换为 LLVM，并且把克里斯招入麾下。克里斯进入了苹果之后如鱼得水，不仅大幅度优化和改进 LLVM 以适应 Objective-C 的语法变革和性能要求，同时发起了 CLang 项目，旨在全面替换 GCC。这个目标目前已经实现了，从 OS X10.9 和 XCode 5 开始，LLVM+GCC 已经被替换成了 LLVM+Clang。

#### MLIR的Motivation是什么？

在MLIR的Paper里，主要是讲了两个场景，第一个场景是NN框架：

![image-20211201155715453](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20211201155715453.png)

> Work on MLIR began with a realization that modern machine learning frameworks are composed of many different compilers, graph technologies, and runtime systems.

比如说，对于Tensorflow来讲，他其实是有很多Compiler、Runtime System、graph technologies组成的。

一个Tensorflow的Graph被执行可以有若干条途径，例如可以直接通过Tensorflow Executor来调用一些手写的op-kernel函数；或者将TensorFlow Graph转化到自己的XLA HLO，由XLA HLO再转化到LLVM IR上调用CPU、GPU或者转化到TPU IR生成TPU代码执行；对于特定的后端硬件，可以转化到TensorRT、或者像是nGraph这样的针对特殊硬件优化过的编译工具来跑；或者转化到TFLite格式进一步调用NNAPI来完成模型的推理。

第二个场景是各种语言的编译器：

![image-20211201162659492](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20211201162659492.png)

每个语言都会有自己的AST，除了AST以外这些语言还得有自己的IR来做language- specific optimization，但是他们的IR最后往往都会接到同样的后端，比如说LLVM IR上来做代码生成，来在不同的硬件上运行。这些语言专属的IR被叫做Mid-Level IR。

对于NN编译器来说，首先一个显而易见的问题是组合爆炸，就比如说TFGraph为了在TPU上跑，要写生成TPU IR的代码，在CPU上跑要写对接LLVM IR的代码，类似的，其他PyTorch这样的框架也需要做同样的事情，但是每个组织发布的自己的框架，比如Pytorch、TensorFlow、MXNet他们的IR设计都是不一样的，不同的IR之间的可迁移性差，这也就代表着大量重复的工作与人力的浪费，这个问题是一个软件的碎片化问题，MLIR的一个设计目的就是为这些DSL提供一种统一的中间表示；其次，各个层次之间的优化无法迁移，比如说在XLA HLO这个层面做了一些图优化，在LLVM IR阶段并不知道XLA HLO做了哪方面的优化，所以有一些优化方式可能会在不同的层级那边执行多次（嘛，我觉得这个问题还好，最后跑起来快就行了，编译慢点没事）；最后，NN编译器的IR一般都是高层的计算图形式的IR，但是LLVM这些是基于三地址码的IR，他们的跨度比较大，这个转换过程带来的开销也会比较大。



### 二、有关MLIR的细节

### 三、已经使用MLIR的项目

#### CIRCT

### 总结

tengine ncnn 这种是根据指令和高层次的 ir 意图，手写算子；mlir 试图不手写，直接从高层次的 ir 编译过去；前者问题是体力活，新芯片新指令新架构要继续肝；后者试图喝咖啡就把这事干了；现在 mlir 在造轮子苦逼的阶段，肝的内容比手写还多一些
比较完善后，mlir 可以将多数优化的工作转为手写 schedule 和 pass；蓝领变白领

nihui建的mlir交流群：677104663（嘛可能不活跃就是了
