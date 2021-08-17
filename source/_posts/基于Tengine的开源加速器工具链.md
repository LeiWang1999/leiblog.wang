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
2. NVIDIA 官方与 Sifive 合作有介绍过一个[firesim-nvdla](https://github.com/CSL-KU/firesim-nvdla)的项目。他们在FPGA云服务器平台上面跑了一个RiscV+NVDLA。并且魔改了darknet，利用内部工具将yolov3中能够用dla运行的子图提取出来生成了Loadable文件，成功运行了yolov3，并且在large配置下yolov3可以到7帧。但是工具链是不开源的，并且局限在了darknet这个框架。
3. 台湾的一家公司Skymizer则野心在制作一套全新的编译器，叫做ONNC。与官方的编译器只能接受caffe的模型不同，ONNC的前端接受的是ONNX模型。可惜的是这家公司开源的编译器也只能跑分类网络并且只能支持full配置，而且无法完成量化。但是其商业版本发布的内容来看支持的算子也很少，并且也只能跑yolov1-tiny。在我的调研里发现，其商业版授权更是高达一年两百多万人名币！起初我以为他们看我是大陆人想宰我，结果认识几个台湾朋友也都反映是这个价格。一些初创公司的第一版加速器设计参考NVDLA，想必也为了短时间内度过难关、寻求应用而不得已花了不少钱购买授权吧。

这反映出来一个加速器设计完成之后的通病，缺少工具链的支持。一个大的公司可以砸人为自己的加速器设计一套独有的工具链，但DLA不支持的算子数量可以高达近百个，实现他们不仅是个体力活，并且实现的过程还需要考虑到在不同设备上的运行效率。其次，还要具备强的模型接受能力，支持量化，能够自动的将计算图中的算子做切分，可以用DLA运行的调度DLA运行、不可以用DLA运行的用CPU去跑。

以上的种种问题对于一个小的团体来说，许多都能在开源社区得到解决，而在[Tengine](https://github.com/OAID/Tengine)就是这样一个框架。

这片文章要介绍的是笔者利用Open AI Lab开源的边缘设备推理框架Tengine，为NVDLA打造一套新的工具链。并且为了与上面的几种方案对比（主要是卷一卷onnc），本篇文章也会利用 Tengine 演示一下如何利用完成模型的转换、量化，调度DLA跑一个yolo网络！

## 一、NVDLA 与 Tengine 简介

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

NVDLA自上而下十分规范，其硬件设计是一个较为完整的体系结构。有寄存器接口，有PE阵列、有多级缓存设计，软件设计包括驱动程序，主机上的 Compiler、设备上的驱动程序与 Runtime等等。在一些学术论文里往往是关键的部分，比如作者的设计里 Runtime 做了哪些事，PE阵列是如何优化，缓存如何设计，但他们的设计细节又大多不会公开，基本无法通过看论文的方式复现。一个刚开始科研的小白看到了一篇论文，认为其有更好的设计方案，但是他拿什么去验证呢？NVDLA就可以作为一个典型的参考设计。NVDLA里，上述的这些单元都是实实在在的代码，研究价值还是极高的，在这里需要感谢英伟达的开源工作。

NVDLA 的硬件是可以配置的，比较典型的有Full、large、media、small这几个版本，修改[spec](https://github.com/nvdla/hw/blob/master/spec/defs/nv_small.spec)即可。本文使用的是其最小的small配置完成设计，一些算子其实是无法实现的，读者可以自行调整设计。

例如，在small的spec文件里：

```C
#define MAC_ATOMIC_C_SIZE_8
#define MAC_ATOMIC_K_SIZE_8
```

表示PE阵列的大小为8\*8，而NVDLA的硬件最大的可以配置为32\*32。

```C
#define SDP_LUT_DISABLE
```

表示Small没有使能查找表，导致只可以支持Relu这一种激活函数。

```C
#define RUBIK_DISABLE
#define RUBIK_CONTRACT_DISABLE
#define RUBIK_RESHAPE_DISABLE
```

表示Small没有使能RUBIK引擎，RUBIK在DLA里的作业是做数据重排，如此不可以支持反卷积等op的实现。

```C
#define SECONDARY_MEMIF_DISABLE
```

表示没有使能Global Sram作为片上的高速缓存。

#### 怎样部署 NVDLA 到自己的 FPGA 芯片上

本设计没有使用到任何外设，所以跟开发板无关，跟芯片有关。NVDLA 对 LUT 的资源要求较高，一般的器件只需要保证 LUT 的数目大于八万就可以部署small了。

https://zhuanlan.zhihu.com/p/378202360

这篇文章针对的是 ZYNQ 7000、ZYNQ MPSOC 器件，对于这些器件，可以参照我的教程使用 Xilinx 的 petalinux 制作 Linux 挂载驱动程序。对于别的器件，就需要使用buildroot这些第三方工具了。

从现有的经验来看，在我的repo的issue里有位老哥在Intel的FPGA上也完成了部署（真的🐂）。如果是纯逻辑资源的器件，不妨试试chipyard，可以烧写一个riscv+nvdla进去，官方也提供了非常简单的生成教程。

而如果你使用的是和笔者一样的ZCU102-rev-1.1，可以直接联系我我把SD卡的Image拷贝给你（或者之后会放到云盘上供大家下载。

上板成功之后，我建议换成 Ubuntu 的文件系统，怎样更换请看这篇文章：

https://zhuanlan.zhihu.com/p/392974835

而之后为了方便大家开发与调试，我将开发板通过以太网桥接到了自己的开发机器上，完成了开发板通过主机的wifi访问互联网，已经有了Windows、Ubuntu、MAC三个版本的解决方案，非常舒服：

https://zhuanlan.zhihu.com/p/378814739

### 为什么选择 Tengine

Tengine（不是阿里云那个Tengine），是由Open AI Lab开源的面向边缘设备推理的Runtime框架。在考虑Tengine之前，我首先考虑的是对接另一个推理框架TVM，但是TVM已经有一个开源的FPGA加速器后端VTA，并且TVM非常庞大。而Tengine的后端对接过程使人感到舒适。

<img src="/Users/wanglei/Tengine/doc/architecture.png" alt="architecture" style="zoom:75%;" />

在前端的模型转换部分，可以把各大主流的训练框架训练出来的模型转化到Tengine的tmfile格式，这就包括了在前言里提到的Caffe与ONNX，格局打开，并且在模型转化的时候，Tengine会做一些图优化，如果对接DLA后端在模型转化的时候需要把fuse_conv_relu_common关掉，因为现在还不支持conv和relu合并的op实现，在之后重构IR的时候会支持的。

```C++
int graph_opt(graph_t graph)
{
    fprintf(stderr, "graph opt begin\n");

    ir_graph_t* ir_graph = (ir_graph_t*)graph;

    if (fuse_conv_unsqueeze(ir_graph) < 0)
        return -1;
    if (fuse_relu_eltwise(ir_graph) < 0)
        return -1;
    if (fuse_conv_bn(ir_graph) < 0)
        return -1;
    if (fuse_fc_bn(ir_graph) < 0)
        return -1;
    if (fuse_bn_scale(ir_graph) < 0)
        return -1;
    if (fuse_conv_relu_common(ir_graph) < 0)
        return -1;

    fprintf(stderr, "graph opt done.\n");
    return 0;
}
```

模型的量化，Tengine也有现成的工具实现PTQ（Post Training Quantization Dynamic, 动态离线量化），并且与TensorRT的量化原理一致，用起来也更方便，不用像tensorrt那样手撸一堆代码，并且量化之后的scale，weight都会保存到生成的tmfile里，在推理的时候可以很方便的从ir中拿到量化的数据，包括量化完成的权重和偏置，以及每一层的scale。

与其他边缘设备推理框架的不同，Tengine的架构设计考虑到了对NPU的支持。在Tengine架构图的右下角出现的TIM-VX是一款NPU，而Tengine将这部分对接的代码开源，给本工作的设计提供了很多参考。Tengine针对这种异构体系结构的存在，存在一个特别关键的机制--**自动切图**。

例如，对于一个典型的Lenet5网络：

![lenet](/Users/wanglei/Downloads/lenet.png)

NVDLA不支持Softmax，那么Tengine在知道要使用NVDLA来做推理的情况下就会自动将原来的计算图切分为两个子图，第一个子图不包括Softmax，那么在DLA看来就是一个最后一层是FullyConnect的网络，第二个子图只有Softmax这一个节点，交给CPU运行。

> 不过，无事闲来大佬曾经说过：一个有经验的部署玩家在实际的部署过程中如果碰到最后一层Softmax，因为不影响分布就会直接去掉。                     	

Tengine还有许多别的优点，在下一章慢慢介绍。

## 二、NVDLA后端对接代码走读

在这一小节，我将介绍一下笔者进行后端对接的流程。而这个过程中的很多细节可以去看一下OpenAILab的后端对接公开课，由闲来大佬主讲：

https://www.bilibili.com/video/BV1dM4y1P7oM

本文的所有代码基本上都在Tengine的`source/device/opendla`下：

```bash
source/device/opendla
├── CMakeLists.txt
├── include
├── lib
├── odla_define.h.in
├── odla_device.cc
├── odla_device.hpp
├── odla_dump.c
├── odla_dump.h
├── odla_executor.cc
├── odla_executor.hpp
├── odla_graph.cc
├── odla_graph.hpp
├── odla_limit.hpp
└── op
```

include和lib文件夹都可以Follow写在官方仓库里的教程，这里贴一个大概的流程：

> 首先，**这里演示的整个编译的过程都在开发板卡上运行**，否则需要交叉编译；例子都是以root的身份来运行的；如何使用开发板连网可以参考[这篇文章](https://zhuanlan.zhihu.com/p/378814739)。

### 2.1 对接思路简介

在对接之前，先讲解一下原来的NVDLA的软件栈是如何工作的，主要分为三个部分：

![DL training software produces a model, which the compilation tool takes and turns into a loadable, which is used by runtime environment. In runtime, UMD submits with ioctl()s to KMD, which is sent to NVDLA with register writes.](http://nvdla.org/_images/nvdla-primer-sw-flow.svg)

#### 2.1.1 Compiler

在三个部分里，Compiler的代码数量是最多的，但却也是最需要我们重构的。NVDLA的编译器前端可以接受Caffe的模型，在内部会通过Parser转化为一个较为通用的Network对象，接着 Network 会被转化成 `CanonicalAST`, 接着 `CanonicalAST`又会被转化成`EngineAST`，`EngineAST`就是整个编译阶段最核心的IR。

例如，下图是我绘制的Lenet5的两个AST的结构，Canonical是通用的，所以该语法书就是一个很简单通用的结构，而EngineAST则每一个Node都被映射到硬件的单元，例如原来CanonicalAST里的`Convolution Node`会被应设为`ConvlutionOPNode`和`SDPBiasOPNode`两个节点，bias、weight和input也被从Node里抽离出来，然后，编译器会有很多PASS来针对EngineAST来工作，比如量化、比如简单的merge；等到所有的Pass都走完，最后得到的IR会被emit成为需要给Runtime的数据，以Flatbuffer序列化协议来组织，最后生成一个Loadable文件，交给Runtime做推理。

<img src="/Users/wanglei/Desktop/nvdla_ast.jpg" alt="nvdla_ast" style="zoom: 25%;" align="center"/>

那么，如何对接呢？首先，因为笔者马上研究生就要开学了，所以想先做出一版可以使用的，于是我决定在第一版使用官方的程序，于是在下文中进行编译的时候是把官方的core代码做成了lib进行链接，调用里面的 Function；另一方面，我想学习一下他的IR是怎么设计的，于是不从Network这么高的层次下手（当然这也导致无法很方便的做到OP的拼接，但也无所谓，反正要重构的）；然后就是决定接入CanonicalAST还是EngineAST了，因为是调用的官方提供的函数来构建Graph，所以EngineAST的创建过于依赖CanonicalGraph，于是笔者选择把Tengine提供给后端的ir_graph接入CanonicalAST，然后转成EngineAST，然后所有的Pass照走。

#### 2.1.2 Runtime

Runtime 的代码量不大，首先是需要将Compiler生成的Loadable进行反序列化，然后为输入输出tensor开辟内存，将输入的Tensor填充数据，递交任务给nvdla，拿到数据。

因为笔者即将开学，所以第一版为了偷懒决定将Compiler和Runtime这两部分代码做成链接库直接调用其提供的函数完成作业，并且我简化了一些流程：

1. Compiler和Runtime对接的时候原先需要由Compiler离线序列化生成Loadable文件，再由Runtime反序列化，我将这个步骤去除，data直接导入给Runtime：

   ```C++
   this->loadable.priv()->getSerializedDataSize(&loadableSize);
   if(!loadableSize){
     fprintf(stderr, "No Loadable Generated. \n");
     return -1;
   }
   NvU8 * buffer  = (NvU8 *)NvDlaAlloc(loadableSize);
   if (buffer == NULL) {
     fprintf(stderr, "Failed to allocate buffer for loadable. \n");
     return -1;
   }
   // deserialize Loadable image
   this->runtime->load(buffer, 0);
   ```

   当然要是需要生成Loadable，也是可以支持的，注释掉这边的代码就好：

   ```C++
   NvDlaFileHandle file = 0;
   std::string fileName = std::string(this->profile->getName()) + ".nvdla";
   fprintf(stdout, "Dump loadable data to : %s . \n", fileName.c_str());
   NvDlaFopen(fileName.c_str(), NVDLA_OPEN_WRITE, &file);
   NvDlaFwrite(file, buffer, loadableSize);
   ```

2. Runtime发送的Task其实又分为两个类别，Emulator和DLA，Emulator是完成CPU作业的，当然官方的Runtime也就只支持Softmax和Scale这两个CPU实现而已，这部分的初始化和注销在执行阶段占用了太多时间，因为现在CPU的Fallback工作都交给Tengine来完成，所以笔者把这部分代码去除了。

3. 设计方案的折中，前文提到过了，是直接从CanonicalAST开始对接。

#### 2.1.3 Kernel Driver

即`opendla.ko`，内核驱动代码负责根据Runtime递交的task完成具体的寄存器配置，调用DMA来进行数据搬移，处理中断信号等，相较于官方的文档在这部分代码里的寄存器配置方法更加清晰，例如这部分代码就配置了卷积的输出chw信息，寄存器的配置顺序看起来可比官方的手册要舒服：

```C++
reg = ((conv_surface->dst_data.width - 1)
       << SHIFT(CACC_D_DATAOUT_SIZE_0_0, DATAOUT_WIDTH)) |
  ((conv_surface->dst_data.height - 1)
   << SHIFT(CACC_D_DATAOUT_SIZE_0_0, DATAOUT_HEIGHT));
cacc_reg_write(D_DATAOUT_SIZE_0, reg);

reg = ((conv_surface->dst_data.channel - 1)
       << SHIFT(CACC_D_DATAOUT_SIZE_1_0, DATAOUT_CHANNEL));
cacc_reg_write(D_DATAOUT_SIZE_1, reg);
```

### 2.1.4 需要实现的函数：

Tengine的架构做的很好，对接一个后端的一部分工作就是反复套娃，抄抄其他后端的代码，然后改改名字。自己需要实现的大概是以下三个函数，以及创建一个后端Engine对象，这部分内容分别在`odla_executor.hpp`、`odla_executor.cc`里。

```C++
int odla_dev_prerun(struct device* dev, struct subgraph* subgraph, void* options);
int odla_dev_run(struct device* dev, struct subgraph* subgraph);
int odla_dev_postrun(struct device* dev, struct subgraph* subgraph);
```

从函数名可以看出他们执行的时间，在实际运行的时候顺序如下：prerun用来初始化数据，完成对接，优化等等，执行一次；prerun执行完了之后，run来塞输入数据，拿到输出数据，可以执行多次；在程序的最后，postrun用来free数据。

那么如果有多个子图，怎样来运行呢？一开始我也对此很迷惑（但这个步骤完全不需要后端对接人员care，Tengine会自己创建多个Engine实例，管理他们。

**如何定义支持哪些op？**

在`odla_limit.hpp`里完成注册，将后端能够支持的op取消注释:

```C++
const int odla_supported_ops[] = {
...
		OP_BIAS,
    //    OP_BROADMUL,
    //    OP_CAST,
    //    OP_CEIL,
    //    OP_CLIP,
    //    OP_COMPARISON,
		//    OP_CONCAT,
    OP_CONST,
    ...
   ]
```

然后在`odla_device.cc`里会把支持的op与不支持的op分到两个vector里，由Tengine根据该信息完成切图，而每个具体op的实现在`op`文件夹里可以找到。

### 2.1 拉取代码

#### 2.1.1 拉取 ZYNQ-NVDLA

```bash 
$ git clone https://github.com/LeiWang1999/ZYNQ-NVDLA # clone不下来的话就本地下载用sftp传上去吧:D
```

#### 2.1.2 拉取 Tengine-Lite

```bash
$ git clone https://github.com/OAID/Tengine.git Tengine
```

### 2.2 Tengine-Lite 集成编译 opendla 

Tengine-Lite 目前只支持一种 opendla 的集成编译方法，即编译opendla的软件支持，首先生成.so文件，而在Tengine编译opendla后端的时候进行链接。

其他的方案，例如在Tengine编译的过程中连同opendla的编译器和运行时的源代码一起编译，由于代码肯定是要重构的，所以现在还不支持。

这里不讲解内核驱动程序`opendla.ko`是如何编译的，如何编译看这篇[文章](https://zhuanlan.zhihu.com/p/378202360)。

#### 2.4.0 载入内核驱动程序

```bash
$ insmod /lib/modules/4.19.0-xilinx-v2019.1/extra/opendla.ko
```

使用dmesg查看内核日志:

```bash
$ dmesg | tail
[   12.817877] macb ff0e0000.ethernet eth0: link up (1000/Full)
[   12.817900] IPv6: ADDRCONF(NETDEV_CHANGE): eth0: link becomes ready
[   20.661453] opendla: loading out-of-tree module taints kernel.
[   20.664248] Probe NVDLA config nvidia,nv_small
[   20.669152] 0 . 12 . 5
[   20.669155] reset engine done
[   20.671257] [drm] Initialized nvdla 0.0.0 20171017 for a0000000.NV_nvdla_wrapper on minor 1
```

查看是否注册了nvdla的中断以及nvdla驱动所需的设备`renderD128`是否存在来确定是否真的安装完成驱动了:

```bash
root@arm:~# insmod /lib/modules/4.19.0-xilinx-v2019.1/extra/opendla.ko 
root@arm:~# cat /proc/interrupts | grep nvdla
 45:          0          0     GIC-0  61 Level     40000000.NV_nvdla_wrapper
root@arm:~# ls /dev/dri/
card0  renderD128
```

#### 2.2.1 编译libjpeg6b

如果是aarch64，跳过该步骤即可，直接使用仓库里的libjpeg.a.

``` bash
$ wget http://www.ijg.org/files/jpegsrc.v6b.tar.gz
$ tar -xzvf jpegsrc.v6b.tar.gz
$ cd jpeg-6b/
$ ./configure
$ make -j `nproc`
$ make install
$ cp /usr/local/lib/libjpeg.a ~/ZYNQ-NVDLA/umd/external/ 
```

#### 2.2.2 编译libprotobuf.a

```bash
$ cd ~/ZYNQ-NVDLA/umd/external/protobuf-2.6/
$ apt-get install -y autoconf automake libtool
$ autoscan & aclocal & autoconf
$ automake --add-missing
$ ./configure
$ make -j `nproc`
$ make install
$ cp /usr/local/lib/libprotobuf.a ~/ZYNQ-NVDLA/umd/apps/compiler/
$ cp /usr/local/lib/libprotobuf.a ~/ZYNQ-NVDLA/umd/core/src/compiler/
```

#### 2.2.3 编译 Compiler 与 Runtime

```bash
$ cd ~/ZYNQ-NVDLA/umd/
$ make -j `nproc` TOP=${PWD} TOOLCHAIN_PREFIX=/usr/bin/ compiler
$ make -j `nproc` TOP=${PWD} TOOLCHAIN_PREFIX=/usr/bin/ runtime
```

这样在out目录下就会生成所需的lib，将lib和include拷贝到Tengine目录下：

```bash
$ cp ~/ZYNQ-NVDLA/include -r ~/Tengine/source/device/opendla
$ cp ~/ZYNQ-NVDLA/umd/out/core/src/compiler/libnvdla_compiler/libnvdla_compiler.so -r ~/Tengine/source/device/opendla/lib/
$ cp ~/ZYNQ-NVDLA/umd/out/core/src/runtime/libnvdla_runtime/libnvdla_runtime.so -r ~/Tengine/source/device/opendla/lib/
$ cp /usr/local/lib/libprotobuf.a ~/Tengine/source/device/opendla/lib/
```

#### 2.2.4 编译 Tengine

```bash
$ cd ~/Tengine
$ mkdir build & cd build
$ cmake .. -DTENGINE_ENABLE_OPENDLA=ON
$ cmake --build . --target tm_classification_opendla
```

### 2.4 RESNET18-CIFAR10 推理演示

RESNET18-CIFAR10 需要的五个op DLA都可以支持，所以不需要做切图，完全交给DLA来运行就好，在之前的文章里使用 NVDLA 的工具链进行过评估，DLA运行的部分需要30ms，而Tengine这一套工具链运行一次只需要10ms左右，一部分原因应该是对接的Graph能吃到Tengine的图优化。

```bash
$ cd examples
$ ./tm_classification_opendla -m /root/Tengine/models/resnet18-cifar10-nosoftmax-relu_int8.tmfile -i /root/Tengine/images/cat.jpg -g 32,32 -s 1,1,1
Mean value not specified, use default   104.0, 116.7, 122.7
tengine-lite library version: 1.4-dev
NVDLA time: 0.012502 seconds

model file : /root/Tengine/models/resnet18-cifar10-nosoftmax-relu_int8.tmfile
image file : /root/Tengine/images/cat.jpg
img_h, img_w, scale[3], mean[3] : 32 32 , 1.000 1.000 1.000, 104.0 116.7 122.7
Repeat 1 times, thread 1, avg time 12.62 ms, max_time 12.62 ms, min_time 12.62 ms
--------------------------------------
10.087049, 3
3.833079, 2
3.026115, 5
2.420892, 4
-0.403482, 0
--------------------------------------
```

### 2.5 YOLOV3-Tiny

## 三、有意思的技术细节

### 2.1 捏一个简单的 OP Test

Tengine还有一个很方便的特性是，可以自己通过创建Tensor、Node的方式捏一个Graph，来方便我们测试一个单独的OP是否可以正常工作。

例如，捏一个只有一层relu的Graph：

```C++
node_t test_node = create_graph_node(graph, node_name, "ReLU");

tensor_t input_tensor = get_graph_tensor(graph, input_name);

if (NULL == input_tensor)
{
  fprintf(stderr, "create test node failed.\n");
  return -1;
}

/* input tensors of test node */
set_node_input_tensor(test_node, 0, input_tensor);

/* output tensors of test node */
tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);
set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);
```

然后往里面塞数据，用`create_opendla_test_graph`就可以调度opendla的后端来运行程序、用`create_cpu_test_graph`就可以使用cpu来运行得到golden数据！

### 2.2 量化与反量化 

经过Tengine量化的模型，我们可以直接从中拿到int8的weight以及int32的bias数据。

### 2.3 Average Pooling 的量化透传问题

### 2.4 DLA的数据摆放

## 四、可以优化的方向



## 五、结语

大概一两年前圈圈虫大佬在ncnn社区的交流群里为刚开源的Tengine做宣传，我在那个时候给Tengine点了star之后就没有怎么关注这个框架了，也是有一些奇妙的缘分。
