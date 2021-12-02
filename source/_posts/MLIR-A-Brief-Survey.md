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

#### Motivation

在MLIR的Paper里，主要是讲了两个场景，第一个场景是NN框架：

![image-20211201155715453](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20211201155715453.png)

> Work on MLIR began with a realization that modern machine learning frameworks are composed of many different compilers, graph technologies, and runtime systems.

比如说，对于Tensorflow来讲，他其实是有很多Compiler、Runtime System、graph technologies组成的。

一个Tensorflow的Graph被执行可以有若干条途径，例如可以直接通过Tensorflow Executor来调用一些手写的op-kernel函数；或者将TensorFlow Graph转化到自己的XLA HLO，由XLA HLO再转化到LLVM IR上调用CPU、GPU或者转化到TPU IR生成TPU代码执行；对于特定的后端硬件，可以转化到TensorRT、或者像是nGraph这样的针对特殊硬件优化过的编译工具来跑；或者转化到TFLite格式进一步调用NNAPI来完成模型的推理。

第二个场景是各种语言的编译器：

![image-20211201162659492](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20211201162659492.png)

每个语言都会有自己的AST，除了AST以外这些语言还得有自己的IR来做language- specific optimization，但是他们的IR最后往往都会接到同样的后端，比如说LLVM IR上来做代码生成，来在不同的硬件上运行。这些语言专属的IR被叫做Mid-Level IR。

对于NN编译器来说，首先一个显而易见的问题是组合爆炸，就比如说TFGraph为了在TPU上跑，要写生成TPU IR的代码，在CPU上跑要写对接LLVM IR的代码，类似的，其他PyTorch这样的框架也需要做同样的事情，但是每个组织发布的自己的框架，比如Pytorch、TensorFlow、MXNet他们的IR设计都是不一样的，不同的IR之间的可迁移性差，这也就代表着大量重复的工作与人力的浪费，这个问题是一个软件的碎片化问题，MLIR的一个设计目的就是为这些DSL提供一种统一的中间表示；其次，各个层次之间的优化无法迁移，比如说在XLA HLO这个层面做了一些图优化，在LLVM IR阶段并不知道XLA HLO做了哪方面的优化，所以有一些优化方式可能会在不同的层级那边执行多次（嘛，我觉得这个问题还好，最后跑起来快就行了，编译慢点没事）；最后，NN编译器的IR一般都是高层的计算图形式的IR，但是LLVM这些是基于三地址码的IR，他们的跨度比较大，这个转换过程带来的开销也会比较大。

最后，MLIR的这一篇Paper的标题是摩尔定律终结下的编译器基础架构，我理解的其含义是在摩尔定律终结，工艺已经到达临界的情况下，针对各种领域设计的DSA用的多了，异构系统的情况下的软件栈需要巨大的人力是一个很大的问题，而MLIR可以提供一种类似脚手架的存在，能够快速融合现有的编译器设计快速的打造软件生态。

### 二、使用CLion远程调试MLIR项目

最好的看代码的方法当然是打断点调试。为了不污染环境，MLIR是我在服务器的Docker环境下构建出来的，需要在本地远程连接调试，踩了一点坑，这部分如果有人有类似的问题，可以参考。

远程调试，一般我会考虑两种方案，一种是用VSCode的Remote SSH插件，远程连接上去看文件很方便，但是本地和远程的文件不同步，很容易乱；一般看代码和文件的时候喜欢用VSCode，在开发的过程中我喜欢用CLion，相当于是在本地开发然后通过SFTP把文件传到服务器上运行，就是环境配置起来有些麻烦。

其实本质上是用CLion远程调试LLVM项目。

首先，先把LLVM的代码Clone下来,并创建build目录来保存编译出来的内容：

```bash
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
```

然后，使用CLion打开`llvm-project/llvm`这个文件夹，此时会根据llvm下的CMakeLists.txt来在本地构建系统，这不是我们期望看到的，可以参考CLion Full Remote Mode这一篇文章来配置远程环境：

https://www.jetbrains.com/help/clion/remote-projects-support.html

然后，为了编译MLIR，我们需要在`Preference->Build,Execution,Deployment->CMake`里，更改为远程设置，在CMAKE OPTIONS里填上：

```bash
-G "Ninja" ../llvm -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON 
-DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" -DLLVM_ENABLE_ASSERTIONS=ON
```

然后，把Build Directory换成`llvm-project/build`的绝对路径。

最后，在Deployment下添加这两个Mappings：

![image-20211201220459184](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20211201220459184.png)

然后，CLion会自动在远程端进行MLIR的编译，编译完成之后的CLion页面如下：

![image-20211201220545776](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20211201220545776.png)

之后我们就可以随意打断点调试了，跑个Toy Tutorial的Ch1康康：

![image-20211201222150521](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20211201222150521.png)

### 三、有关MLIR的细节

#### MLIR的表达式结构

在学习LLVM的时候，我们可以将一段C/C++程序使用CLang生成AST、生成LLVM IR与Pass优化，再到最后的代码生成来学习整个WorkFlow，但是目前来讲，官方没有提供C/C++代码生成MLIR的流程，虽然类似的工作有一些人来做：

- CLang AST -> MLIR :  https://github.com/wsmoses/Polygeist
- C -> MLIR : https://github.com/wehu/c-mlir

还有MLIR官方也在推进的CIL，但是这些目前来讲都是野路子。MLIR提供了一个Toy语言，可以方便我们学习MLIR的流程，关于Toy语言详细，可以移步[Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/).

这样跟随Toy Tutorial，我们把MLIR的一些细节过一下，从以.toy结尾的源文件到最后的Lowering需要经过如下的步骤：

.toy 文本文件 -> Toy AST -> MLIRGen -> Transformation -> Lowering -> JIT/LLVM IR.

首先是AST部分，对于如下的Toy语言，没有用ast.toy这个文件，有点复杂而且是会抛异常的。如下来自Ch2的Codegen.toy的例子能更好的贯穿文章：

```python
# User defined generic function that operates on unknown shaped arguments
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}
```



打印出该语言的Toy抽象语法树：

```python
  Module:
    Function 
      Proto 'multiply_transpose' @../../mlir/test/Examples/Toy/Ch2/codegen.toy:4:1
      Params: [a, b]
      Block {
        Return
          BinOp: * @../../mlir/test/Examples/Toy/Ch2/codegen.toy:5:25
            Call 'transpose' [ @../../mlir/test/Examples/Toy/Ch2/codegen.toy:5:10
              var: a @../../mlir/test/Examples/Toy/Ch2/codegen.toy:5:20
            ]
            Call 'transpose' [ @../../mlir/test/Examples/Toy/Ch2/codegen.toy:5:25
              var: b @../../mlir/test/Examples/Toy/Ch2/codegen.toy:5:35
            ]
      } // Block
    Function 
      Proto 'main' @../../mlir/test/Examples/Toy/Ch2/codegen.toy:8:1
      Params: []
      Block {
        VarDecl a<2, 3> @../../mlir/test/Examples/Toy/Ch2/codegen.toy:9:3
          Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @../../mlir/test/Examples/Toy/Ch2/codegen.toy:9:17
        VarDecl b<2, 3> @../../mlir/test/Examples/Toy/Ch2/codegen.toy:10:3
          Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @../../mlir/test/Examples/Toy/Ch2/codegen.toy:10:17
        VarDecl c<> @../../mlir/test/Examples/Toy/Ch2/codegen.toy:11:3
          Call 'multiply_transpose' [ @../../mlir/test/Examples/Toy/Ch2/codegen.toy:11:11
            var: a @../../mlir/test/Examples/Toy/Ch2/codegen.toy:11:30
            var: b @../../mlir/test/Examples/Toy/Ch2/codegen.toy:11:33
          ]
        VarDecl d<> @../../mlir/test/Examples/Toy/Ch2/codegen.toy:12:3
          Call 'multiply_transpose' [ @../../mlir/test/Examples/Toy/Ch2/codegen.toy:12:11
            var: b @../../mlir/test/Examples/Toy/Ch2/codegen.toy:12:30
            var: a @../../mlir/test/Examples/Toy/Ch2/codegen.toy:12:33
          ]
        Print [ @../../mlir/test/Examples/Toy/Ch2/codegen.toy:13:3
          var: d @../../mlir/test/Examples/Toy/Ch2/codegen.toy:13:9
        ]
      } // Block
```

不难察觉其和LLVM的抽象语法树的组织形式一样，一个文件对应一个Module，Module里包含若干Function，Function里包含若干基本块BasicBlock，BasicBlock里包含了若干条指令。

.toy 文本文件 -> Toy AST -> **MLIRGen** -> Transformation -> Lowering -> JIT/LLVM IR.

MLIR的语言详细参考官方文档中的 Lang Ref : https://mlir.llvm.org/docs/LangRef/

`ch2 ../../mlir/test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo`

其根本上是一个类似图的数据结构，节点就是Operations，边就是Value.每个Value可以是Operation的返回值或者是Block的参数，Operations被包含在Blocks里，Block包含在Regions里，Regions包含在Operations里，如此往复，生生不息。

Operations顾名思义可以代表很多内容，高层次比如说函数定义、函数调用、内存分配、进程创建；低层次比如说硬件专属的指令，配置寄存器和逻辑门（应该是指CIRCT这样的HLS项目），这些OP可以被我们随意的扩充。



#### Dialect

#### Lowering

### 三、已经使用MLIR的项目

但是MLIR官方给的toy语言不足以支撑其在一个实际工业应用下MLIR可能发挥的作用，只能说可以帮助我们更好的了解MLIR。有关MLIR具体可以解决什么样的问题，解决的怎么样了，需要看一下当下使用MLIR的一些项目。

#### CIRCT

### 总结

tengine ncnn 这种是根据指令和高层次的 ir 意图，手写算子；mlir 试图不手写，直接从高层次的 ir 编译过去；前者问题是体力活，新芯片新指令新架构要继续肝；后者试图喝咖啡就把这事干了；现在 mlir 在造轮子苦逼的阶段，肝的内容比手写还多一些
比较完善后，mlir 可以将多数优化的工作转为手写 schedule 和 pass；蓝领变白领

nihui建的mlir交流群：677104663（嘛可能不活跃就是了
