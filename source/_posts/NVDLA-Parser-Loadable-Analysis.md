---
title: NVDLA Parser | Loadable Analysis
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2021-03-30 09:32:15
---

## 0x 前言

我们知道，NVDLA的软件栈主要分为两个部分即`Compiler`与`Runtime`，由于`Compiler`与硬件无关，所以可以在我们自己的开发机器上编译运行调试，理解起来也较为方便；而`Runtime`与硬件有关，调试非常困难，官方提供的预构建的文件又都是针对64位的操作系统，这对我们在仅搭载了32位处理器的ZYNQ7000系列的开发板上编译Runtime带来了不小的挑战。Loadable文件是两者之间通信的媒介，本文记述一下Loadable文件的组织结构和解析方法，既然不能吃官方给的饭，那么可以试一试自己在SOC上做一份调度的算法，解读Loadable文件就是第一步。

Github Repo：https://github.com/LeiWang1999/nvdla-depicter

<!-- more -->

在官方给出的NVDLA的使用说明里，我们需要将预训练的caffe模型以及其他一系列可选参数作为输入，由Compiler编译后产生Loadable文件，期间可以进行一些硬件无关的优化，例如算子融合、内存重用，这些设置可以在[Profile.cpp](https://github1s.com/nvdla/sw/blob/HEAD/umd/core/src/compiler/Profile.cpp)里找到。

```C++
void Profile::setBasicProfile()
{
    setUseCVSRAMAllocate(false);
    setUseMemPool(false);
    setUseReusePooledMemory(false);
    setUseGreedyEviction(false);
    setCanSDPBustNOPs(false);
    setCanSDPMergeMathOps(false);
    setCanSDPFuseSubEngineOps(false);
}
```

例如最常使用的，也是官方给的默认参数`fast-math`指令编译：

```C++
void Profile::setFastMathProfile()
{
    setPerformanceProfile();
    setCanSDPMergeMathOps(true);
    setCanSDPFuseSubEngineOps(true);
}
```

在aarch64的模拟器里运行runtime、fast-math和default，即全都设置为false两种模式速度差异非常明显。

关于Compiler还做了哪些事情，我们在以后的文章里讨论，这里我们只需要关心的是Compiler最后生成的目标文件，即Loadable。

在sw的仓库里，我们可以发现[loadable.fbs](https://github.com/nvdla/sw/blob/master/umd/core/src/common/loadable.fbs)文件，这将是这篇博客的开始，关于FlatBuffers的介绍。

## 1x FlatBuffers 简介

> FlatBuffers是一个开源的、跨平台的、高效的、提供了C++/Java接口的序列化工具库。它是Google专门为游戏开发或其他性能敏感的应用程序需求而创建。尤其更适用于移动平台，这些平台上内存大小及带宽相比桌面系统都是受限的，而应用程序比如游戏又有更高的性能要求。它将序列化数据存储在缓存中，这些数据既可以存储在文件中，又可以通过网络原样传输，而不需要任何解析开销。

简单来讲，我们需要在流中传输一个对象，比如网络流。一般我们需要把这个对象序列化之后才能在流中传输（例如，可以把对象直接转化为字符串），然后在接收端进行反序列化（例如把字符串解析成对象）。但是显然把对象转成字符串传输的方法效率十分低下，于是有了各种流的转换协议，FlatBuffers也是其中一种。

本文不具体讨论其是如何压缩对象，具体可以参照官方的[文档](https://google.github.io/flatbuffers/index.html#flatbuffers_overview)。这里借用简书中的一个例子：

![FlatBuffers](http://leiblog.wang/static/image/2021/3/HlUQLW.jpg)

1. 每一个存储在FlatBuffer中的对象，被分割成2部分：中心点（Pivot Point）左边的元数据（vtable）和中心点右边的真实数据。
2. 每个字段都对应vtable中的一个槽（slot），它存储了那个字段的真实数据的偏移量。例如，John的vtable的第一个槽的值是1，表明John的名字被存储在John的中心点右边的一个字节的位置上。
3. 如果一个字段是一个对象，那么它在vtable中的偏移量会指向子对象的中心点。比如，John的vtable中第三个槽指向Mary的中心点。
4. 要表明某个字段现在没有数据，可以在vtable对应的槽中使用偏移量0来标注。

[loadable.fbs](https://github.com/nvdla/sw/blob/master/umd/core/src/common/loadable.fbs)是由Schema语言组织的，我们不需要添加定义，只需要考虑理解和读取。

### 1.1x 安装FlatBuffers

```bash
git clone https://github.com/google/flatbuffers.git
git checkout v1.6.0
cmake -G "Unix Makefiles"
make 
make install 
```

确认是否安装成功:

```bash
promote at ~/flatbuffers ±(cebdad4d) ✗ ❯ flatc --version
flatc version 1.6.0 (Mar 30 2021)
```

在笔者写这篇博客的时候，flatc的版本已经是v1.12.0了，在官方仓库里的`flatbuffers.h`文件里可以发现，其使用的flatbuffers的版本是`1.6.0`：

```bash
#define FLATBUFFERS_VERSION_MAJOR 1
#define FLATBUFFERS_VERSION_MINOR 6
#define FLATBUFFERS_VERSION_REVISION 0
```

虽然flatterbuffers声称其有非常好的前后兼容性，但是v1.12.0的程式与官方的代码有些对不上号，为了方便起见，这里把版本回退到v1.6.0.

生成[loadable_generated.h](https://github.com/nvdla/sw/blob/master/umd/core/src/common/include/priv/loadable_generated.h)，在我们的项目里引入这个头文件就可以读取loadable了。

```bash
flatc -c loadable.fbs
```

FlatBuffers是轻依赖的，生成的头文件也只需要依赖`flatbuffers.h`,在我的项目目录下，这样组织文件，flatbuffers文件夹直接拷贝flatbuffers/include目录下的flatbuffers文件夹，但其实只需要`flatbuffers.h`就可以了：

```bash
promote at ~/nvdla-depicter/external ±(master) ✗ ❯ tree
.
├── flatbuffers
│   ├── code_generators.h
│   ├── flatbuffers.h
│   ├── flatc.h
│   ├── flexbuffers.h
│   ├── grpc.h
│   ├── hash.h
│   ├── idl.h
│   ├── reflection.h
│   ├── reflection_generated.h
│   └── util.h
├── half.h
├── loadable.fbs
└── loadable_generated.h
```

## 2x 解析loadble

在这个小节，我们开始正片，讲解

### 2.1x version

```zsh
Loadable Version is 0.7.0
```

### 2.2x task_list

```zsh
Loadable TaskListEntry : 
id : 0
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 1
instance : -1
address_list : 5 1 2 3 4 6 7 8 9 10 
pre_actions : 
post_actions List : 0 

Loadable TaskListEntry : 
id : 1
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 2
instance : -1
address_list : 11 1 2 3 4 12 13 14 15 16 
pre_actions : 1 
post_actions List : 2 
```

### 2.3x memory_list

```zsh
Loadable memory_list : 
id : 0
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 1
size : 4096
alignment : 4096
contents size 0 : 
offsets size 0 : 
bind_id : 0
tensor_desc_id : 0

Loadable memory_list : 
id : 1
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 3
size : 925696
alignment : 4096
contents size 8 : tb-0 tb-2 tb-3 tb-4 tb-5 tb-6 tb-8 tb-9 
offsets size 8 : 0 4096 32768 8192 524288 12288 16384 24576 
bind_id : 0
tensor_desc_id : 0

Loadable memory_list : 
id : 2
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 1
size : 24576
alignment : 4096
contents size 0 : 
offsets size 0 : 
bind_id : 0
tensor_desc_id : 0

Loadable memory_list : 
id : 3
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 5
size : 6272
alignment : 4096
contents size 0 : 
offsets size 0 : 
bind_id : 0
tensor_desc_id : 0

Loadable memory_list : 
id : 4
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 9
size : 16
alignment : 4096
contents size 0 : 
offsets size 0 : 
bind_id : 0
tensor_desc_id : 1

Loadable memory_list : 
id : 5
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 3
size : 40
alignment : 4096
contents size 1 : task-0-addr0 
offsets size 1 : 0 
bind_id : 0
tensor_desc_id : 0

Loadable memory_list : 
id : 6
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 3
size : 360
alignment : 4096
contents size 1 : task-0-dep_graph 
offsets size 1 : 0 
bind_id : 0
tensor_desc_id : 0

Loadable memory_list : 
id : 7
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 3
size : 1160
alignment : 4096
contents size 1 : task-0-op_list 
offsets size 1 : 0 
bind_id : 0
tensor_desc_id : 0

Loadable memory_list : 
id : 8
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 3
size : 6440
alignment : 4096
contents size 1 : task-0-surf_list 
offsets size 1 : 0 
bind_id : 0
tensor_desc_id : 0

Loadable memory_list : 
id : 9
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 3
size : 700
alignment : 4096
contents size 1 : task-0-lut_list 
offsets size 1 : 0 
bind_id : 0
tensor_desc_id : 0

Loadable memory_list : 
id : 10
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 1
size : 4096
alignment : 4096
contents size 0 : 
offsets size 0 : 
bind_id : 0
tensor_desc_id : 0

Loadable memory_list : 
id : 11
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 3
size : 256
alignment : 4096
contents size 1 : task-1-addr0 
offsets size 1 : 0 
bind_id : 0
tensor_desc_id : 0

Loadable memory_list : 
id : 12
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 3
size : 24
alignment : 4096
contents size 1 : task-1-op_list 
offsets size 1 : 0 
bind_id : 0
tensor_desc_id : 0

Loadable memory_list : 
id : 13
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 3
size : 512
alignment : 4096
contents size 1 : task-1-op_buf_list 
offsets size 1 : 0 
bind_id : 0
tensor_desc_id : 0

Loadable memory_list : 
id : 14
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 1
size : 4096
alignment : 4096
contents size 0 : 
offsets size 0 : 
bind_id : 0
tensor_desc_id : 0

Loadable memory_list : 
id : 15
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 1
size : 4096
alignment : 4096
contents size 0 : 
offsets size 0 : 
bind_id : 0
tensor_desc_id : 0

Loadable memory_list : 
id : 16
domain { SYSTEM = 0, SRAM = 1 } : 0
flags { NONE = 0, ALLOC = 1, SET = 2, INPUT = 4, OUTPUT = 8 } : 1
size : 4096
alignment : 4096
contents size 0 : 
offsets size 0 : 
bind_id : 0
tensor_desc_id : 0
```

### 2.4x address_list

```zsh
Loadable AddressListEntry : 
id : 0
mem_id : 0
offset : 0
size : 4096

Loadable AddressListEntry : 
id : 1
mem_id : 1
offset : 0
size : 925696

Loadable AddressListEntry : 
id : 2
mem_id : 2
offset : 0
size : 24576

Loadable AddressListEntry : 
id : 3
mem_id : 3
offset : 0
size : 6272

Loadable AddressListEntry : 
id : 4
mem_id : 4
offset : 0
size : 16

Loadable AddressListEntry : 
id : 5
mem_id : 5
offset : 0
size : 40

Loadable AddressListEntry : 
id : 6
mem_id : 6
offset : 0
size : 360

Loadable AddressListEntry : 
id : 7
mem_id : 7
offset : 0
size : 1160

Loadable AddressListEntry : 
id : 8
mem_id : 8
offset : 0
size : 6440

Loadable AddressListEntry : 
id : 9
mem_id : 9
offset : 0
size : 700

Loadable AddressListEntry : 
id : 10
mem_id : 10
offset : 0
size : 4096

Loadable AddressListEntry : 
id : 11
mem_id : 11
offset : 0
size : 256

Loadable AddressListEntry : 
id : 12
mem_id : 12
offset : 0
size : 24

Loadable AddressListEntry : 
id : 13
mem_id : 13
offset : 0
size : 512

Loadable AddressListEntry : 
id : 14
mem_id : 14
offset : 0
size : 4096

Loadable AddressListEntry : 
id : 15
mem_id : 15
offset : 0
size : 4096

Loadable AddressListEntry : 
id : 16
mem_id : 16
offset : 0
size : 4096
```

### 2.5x event_list

```zsh
Loadable event_list : 
id : 0
type { EVENTTYPE0 = 0, EVENTTYPE1 = 1, EVENTTYPE2 = 2 } : 0
target : 0
val : 1
op { WAIT = 0, SIGNAL = 1 } : 1

Loadable event_list : 
id : 1
type { EVENTTYPE0 = 0, EVENTTYPE1 = 1, EVENTTYPE2 = 2 } : 0
target : 0
val : 1
op { WAIT = 0, SIGNAL = 1 } : 0

Loadable event_list : 
id : 2
type { EVENTTYPE0 = 0, EVENTTYPE1 = 1, EVENTTYPE2 = 2 } : 0
target : 0
val : 2
op { WAIT = 0, SIGNAL = 1 } : 1
```

### 2.6x blobs

```zsh
Loadable Blob : 0
name : task-0-addr0
size : 40
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 1
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 1
version : 0.12.3
data size is 40 : 

Loadable Blob : 1
name : task-0-dep_graph
size : 360
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 1
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 2
version : 0.12.3
data size is 360 : 

Loadable Blob : 2
name : task-0-lut_list
size : 700
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 1
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 5
version : 0.12.3
data size is 700 : 

Loadable Blob : 3
name : task-0-op_list
size : 1160
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 1
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 3
version : 0.12.3
data size is 1160 : 

Loadable Blob : 4
name : task-0-surf_list
size : 6440
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 1
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 4
version : 0.12.3
data size is 6440 : 

Loadable Blob : 5
name : task-1-addr0
size : 256
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 2
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 1
version : 0.0.1
data size is 256 : 

Loadable Blob : 6
name : task-1-op_buf_list
size : 512
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 2
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 4
version : 0.0.1
data size is 512 : 

Loadable Blob : 7
name : task-1-op_list
size : 24
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 2
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 3
version : 0.0.1
data size is 24 : 

Loadable Blob : 8
name : tb-0
size : 504
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 0
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 0
version : 0.0.0
data size is 504 : 

Loadable Blob : 9
name : tb-2
size : 40
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 0
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 0
version : 0.0.0
data size is 40 : 

Loadable Blob : 10
name : tb-3
size : 25000
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 0
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 0
version : 0.0.0
data size is 25000 : 

Loadable Blob : 11
name : tb-4
size : 100
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 0
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 0
version : 0.0.0
data size is 100 : 

Loadable Blob : 12
name : tb-5
size : 400000
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 0
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 0
version : 0.0.0
data size is 400000 : 

Loadable Blob : 13
name : tb-6
size : 1000
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 0
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 0
version : 0.0.0
data size is 1000 : 

Loadable Blob : 14
name : tb-8
size : 5000
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 0
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 0
version : 0.0.0
data size is 5000 : 

Loadable Blob : 15
name : tb-9
size : 20
interface { NONE = 0, DLA1 = 1, EMU1 = 2 } : 0
sub_interface {0:NONE 1:ADDR0 2:DEPS 3:OPS 4:SURFS 5:LUTS} : 0
version : 0.0.0
data size is 20 : 
```

### 2.7x tensor_desc_list

```zsh
Loadable tensor_desc_list : 
name : data
id : 0
size : 6272
offset : 0
data_format { UNKNOWN = 0, NCHW = 1, NHWC = 2 } : 3
data_type { UNKNOWN = 0, FLOAT = 1, HALF = 2, INT16 = 3, INT8 = 4 } : 4
data_category { IMAGE = 0, WEIGHT = 1, FEATURE = 2, PLANAR = 3, BIAS = 4 } : 2
pixel_format {
    R8 = 0,
    R10 = 1,
    R12 = 2,
    R16 = 3,
    R16_I = 4,
    R16_F = 5,
    A16B16G16R16 = 6,
    X16B16G16R16 = 7,
    A16B16G16R16_F = 8,
    A16Y16U16V16 = 9,
    V16U16Y16A16 = 10,
    A16Y16U16V16_F = 11,
    A8B8G8R8 = 12,
    A8R8G8B8 = 13,
    B8G8R8A8 = 14,
    R8G8B8A8 = 15,
    X8B8G8R8 = 16,
    X8R8G8B8 = 17,
    B8G8R8X8 = 18,
    R8G8B8X8 = 19,
    A2B10G10R10 = 20,
    A2R10G10B10 = 21,
    B10G10R10A2 = 22,
    R10G10B10A2 = 23,
    A2Y10U10V10 = 24,
    V10U10Y10A2 = 25,
    A8Y8U8V8    = 26,
    V8U8Y8A8    = 27,
    Y8_U8V8_N444 = 28,
    Y8_V8U8_N444 = 29,
    Y10_U10V10_N444 = 30,
    Y10_V10U10_N444 = 31,
    Y12_U12V12_N444 = 32,
    Y12_V12U12_N444 = 33,
    Y16_U16V16_N444 = 34,
    Y16_V16U16_N444 = 35,
    FEATURE = 36
} : 37
pixel_mapping { PITCH_LINEAR = 0, INVALID_PIXEL_MAP = 1 } : 0
n : 1
c : 1
w : 28
h : 28
stride_0 : 1
stride_1 : 224
stride_2 : 6272
stride_3 : 0
stride_4 : 0
stride_5 : 0
stride_6 : 0
stride_7 : 0

Loadable tensor_desc_list : 
name : prob
id : 1
size : 16
offset : 0
data_format { UNKNOWN = 0, NCHW = 1, NHWC = 2 } : 3
data_type { UNKNOWN = 0, FLOAT = 1, HALF = 2, INT16 = 3, INT8 = 4 } : 4
data_category { IMAGE = 0, WEIGHT = 1, FEATURE = 2, PLANAR = 3, BIAS = 4 } : 2
pixel_format {
    R8 = 0,
    R10 = 1,
    R12 = 2,
    R16 = 3,
    R16_I = 4,
    R16_F = 5,
    A16B16G16R16 = 6,
    X16B16G16R16 = 7,
    A16B16G16R16_F = 8,
    A16Y16U16V16 = 9,
    V16U16Y16A16 = 10,
    A16Y16U16V16_F = 11,
    A8B8G8R8 = 12,
    A8R8G8B8 = 13,
    B8G8R8A8 = 14,
    R8G8B8A8 = 15,
    X8B8G8R8 = 16,
    X8R8G8B8 = 17,
    B8G8R8X8 = 18,
    R8G8B8X8 = 19,
    A2B10G10R10 = 20,
    A2R10G10B10 = 21,
    B10G10R10A2 = 22,
    R10G10B10A2 = 23,
    A2Y10U10V10 = 24,
    V10U10Y10A2 = 25,
    A8Y8U8V8    = 26,
    V8U8Y8A8    = 27,
    Y8_U8V8_N444 = 28,
    Y8_V8U8_N444 = 29,
    Y10_U10V10_N444 = 30,
    Y10_V10U10_N444 = 31,
    Y12_U12V12_N444 = 32,
    Y12_V12U12_N444 = 33,
    Y16_U16V16_N444 = 34,
    Y16_V16U16_N444 = 35,
    FEATURE = 36
} : 37
pixel_mapping { PITCH_LINEAR = 0, INVALID_PIXEL_MAP = 1 } : 0
n : 1
c : 10
w : 1
h : 1
stride_0 : 1
stride_1 : 8
stride_2 : 8
stride_3 : 0
stride_4 : 0
stride_5 : 0
stride_6 : 0
stride_7 : 0
```

### 2.8x reloc_list

```zsh
Loadable reloc_list : 
address_id : 3
write_id : 8
offset : 116
interface : 1
sub_interface : 4
reloc_type : 1

Loadable reloc_list : 
address_id : 3
write_id : 8
offset : 120
interface : 1
sub_interface : 4
reloc_type : 2

Loadable reloc_list : 
address_id : 4
write_id : 13
offset : 274
interface : 2
sub_interface : 4
reloc_type : 1

Loadable reloc_list : 
address_id : 4
write_id : 13
offset : 278
interface : 2
sub_interface : 4
reloc_type : 2
```

### 2.9x submit_list

```zsh
Loadable submit_list : 
id : 0
task_id size is 1 : 
0 

Loadable submit_list : 
id : 1
task_id size is 1 : 
1 
```

### 2.10x Read Net_desc

![Debug_net_desc](http://leiblog.wang/static/image/2021/4/QAmSUM.png)

