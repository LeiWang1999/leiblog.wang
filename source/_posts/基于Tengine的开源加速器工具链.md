---
title: 基于Tengine的开源加速器工具链
categories:
  - Technical
tags:
  - Tengine
  - NVDLA
date: 2021-08-15 19:37:46
---

## 前言

在之前的文章里笔者已经记述了怎样在FPGA上映射由英伟达开源的加速器NVDLA。但是NVDLA的官方发布的工具链很弱，只能端到端地运行极为简单的分类网络，而现在在绝大部分的深度神经网络应用里分类往往只是其中一小部分。例如我现在想利用加速器去运行yolo，但其中有许多加速器并不支持的算子，加速器支持的convolution、pooling、relu等等算子最好都要用加速器运行，而那些不支持的算子则需要Fallback到CPU去运行。可是这一步说起来容易，但却很少有人能够实现。

<!-- more -->

有意思的是，对于这件事近年来一些机构也基本上在可能的方向上有过尝试，例如：

1. 台湾工研院的做法是硬刚NVDLA的寄存器配置，将yolov1-tiny之中可以使用加速器运行的层单独提出来并且量化，手动根据加速器需要的摆放格式放到内存，然后通过配置寄存器的方法完成加速器侧的推理，之后拿到数据之后用cpu算其他的layer。这种原始的方法部署需要对硬件理解的极为深刻，显然是一个非常懂硬件的一开始能够想到的可行方案。但我想一个人没有一两个月是完不成一个网络的部署的，而如果需要推理的目标网络换成另一个则需要做太多的重复工作。
2. NVIDIA 官方与 Sifive 合作有介绍过一个[firesim-nvdla](https://github.com/CSL-KU/firesim-nvdla)的项目。他们在FPGA云服务器平台上面跑了一个RiscV+NVDLA。并且魔改了darknet，利用内部工具将yolov3中三个能够用dla运行的子图提取出来生成了Loadable文件，成功运行了yolov3。工具链是不开源的，并且局限在darknet。
3. 台湾的一家公司Skymizer则野心在制作一套全新的编译器--ONNC，与官方的编译器只能接受caffe的模型不同，ONNC的前端接受的是ONNX模型。可惜的是这家公司开源的编译器也只能跑分类网络并且只能支持full配置，而且无法完成量化。但是其商业版本发布的内容来看支持的算子也很少，并且也只能跑yolov1-tiny。在我的调研里发现，其商业版授权更是高达一年两百多万人名币！起初我以为他们看我是大陆人想宰我，结果认识几个台湾朋友也都反映是这个价格。一些初创公司的第一版加速器设计参考NVDLA，想必也为了短时间内度过难关、寻求应用而不得已花了不少钱购买授权吧。

这反映出来一个加速器设计设计完成之后的通病，缺少工具链的支持。一个大的公司可以砸人为自己的加速器设计一套独有的工具链，但DLA不支持的算子数量可以高达近百个，实现他们不仅是个体力活，并且实现的过程还需要考虑到在不同设备上的运行效率。其次，还要具备强的模型接受能力，支持量化，能够自动的将计算图中的算子做切分，可以用DLA运行的调度DLA运行、不可以用DLA运行的用CPU去跑。

以上的种种问题对于一个小的团体来说，许多都能在开源社区得到解决，而在[Tengine](https://github.com/OAID/Tengine)就是这样一个框架。

这片文章要介绍的是笔者利用Open AI Lab开源的边缘设备推理框架Tengine，为NVDLA打造一套新的工具链。并且为了与上面的几种方案对比，本篇文章也会利用 Tengine 演示一下如何利用完成模型的转换、量化，调度DLA跑一个yolo网络！

## 正文

应该少有读者会同时了解这两个框架，先简单介绍一下两者。

### NVDLA (NVIDIA Deep Learning Accelerator) 是什么？

NVDLA 是英伟达于2017年开源出来的深度学习加速器框架，他的官方[repo](https://github.com/nvdla)在这里。可惜的是，这个项目被开源出来一年后就草草停止维护了，留下了开发到一半突然停工的代码。NVDLA的核心是MAC阵列、其有若干个处理单元来决定MAC阵列的调度与运算：

- Convolution Core – optimized high-performance convolution engine.
- Single Data Processor – single-point lookup engine for activation functions.
- Planar Data Processor – planar averaging engine for pooling.
- Channel Data Processor – multi-channel averaging engine for advanced normalization functions.
- Dedicated Memory and Data Reshape Engines – memory-to-memory transformation acceleration for tensor reshape and copy operations.

并且这些处理单元都暴露出了一组可配置寄存器接口。

!["Small" and "Large" NVDLA systems side by side, with SRAMIF disconnected on "small" system, and a microcontroller on "large" system.](http://nvdla.org/_images/nvdla-primer-system-comparison.svg)

前文中提到他软件设计工具链的缺陷，主要是其Compiler、Runtime的程序写的非常乱，明显是根据公司内部制定的设计文档，很多人一起完成的开发过程，整个架构设计和代码风格真的不咋地，但是其内核驱动代码写的却很棒。

硬件这部分，RTL代码生成我也不是很懂为何要使用如此多的工具链去搭建tmake，利用C语言的宏来处理字符串决定哪些模块应该生成，哪些模块不应该生成，而不是用 Verilog 内部的宏定义完成这部分工作。其次，其发布时间是2017年，那么实际的研发时间应该在2015年到2016年，其中的很多设计也正是借鉴的寒武纪最初的 DianNao、DaDianNao结构。而现在，不说寒武纪已经迭代了好几个版本并且上市了，而在这之后的一些研究，例如eyeriss系列围绕着数据复用做出来的大量让人惊艳的工作，NVDLA当时的设计肯定是没有考虑到的，所以其性能肯定不如当下的商用加速器。（话说能商用肯定搞钱去了，干嘛开源给大家玩

**虽然看起来NVDLA的缺点很多，但其仍然极具学习价值！**

NVDLA自上而下十分规范，其硬件设计有许多单元，PE阵列、Global Sram，软件设计包括驱动程序，主机上的 Compiler、设备上的驱动程序与 Runtime等等。在很多学术论文里都是关键的那部分，比如作者使用 Runtime 做了哪些事，PE阵列我怎么优化的，我这里放了一个 Sram，但他们的设计细节又不会公开，而也没有参考设计，基本无法复现。例如此时此刻，一个刚开始科研的小白想到了一种dataflow，可能在某些任务上比之前所有的dataflow都要更好，但是他拿什么去验证呢？NVDLA里，上述的这些单元都是实实在在的代码，研究价值还是极高的，在这里需要感谢英伟达。

NVDLA 的硬件是可以配置的，比较典型的有Full、large、media、small这几个版本，修改[spec](https://github.com/nvdla/hw/blob/master/spec/defs/nv_small.spec)即可。本文使用的是其最小的small配置完成设计，一些算子其实是无法实现的，读者可以自行调整设计。

1. NVDLA的硬件最大的可以配置为32\*32，Small映射PE阵列的大小为8\*8。
2. Small没有使能查找表，导致只可以支持Relu这一种激活函数。
3. Small没有使能RUBIK引擎，RUBIK在DLA里的作业是做数据重排，如此不可以支持反卷积等op的实现。
4. 没有使能Global Sram第二缓存。

至于 NVDLA 如何部署到开发板卡上，看我之前的文章：

#### 怎样部署 NVDLA 到自己的 FPGA 芯片上

本设计没有使用到任何外设，所以跟开发板无关，跟芯片有关。NVDLA 对 LUT 的资源要求较高，一般的器件只需要保证 LUT 的数目大于八万就可以部署small了。

https://zhuanlan.zhihu.com/p/378202360

这篇文章针对的是 ZYNQ 7000、ZYNQ MPSOC 器件，对于这些器件，可以参照我的教程使用 Xilinx 的 petalinux 制作 Linux 挂载驱动程序。对于别的器件，就需要使用buildroot这些第三方工具了。

从现有的经验来看，在我的repo的issue里有位老个在Intel的FPGA上也完成了部署（真的🐂）。如果是纯逻辑资源的器件，不妨试试chipyard，可以烧写一个riscv+nvdla进去，官方也提供了非常简单的生成教程。

而如果你使用的是和笔者一样的ZCU102-rev-1.1，可以直接联系我我把SD卡的Image拷贝给你（或者之后会放到云盘上供大家下载。

上板成功之后，我建议换成 Ubuntu 的文件系统，怎样更换请看这篇文章：

https://zhuanlan.zhihu.com/p/392974835

而之后为了方便大家开发与调试，我将开发板通过以太网桥接到了自己的开发机器上，完成了开发板通过主机的wifi访问互联网，已经有了Windows、Ubuntu、MAC三个版本的解决方案，非常舒服：

https://zhuanlan.zhihu.com/p/378814739

### 为什么选择 Tengine

其实是Tengine选择了我XDD

