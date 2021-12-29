---
title: 论文阅读｜LLHD
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2021-12-29 14:27:05
---

《LLHD: A Multi-level Intermediate Representation for Hardware Description Languages》

因为在CIRCT里有一个叫llhd的dialect，于是很简单的survey了一下这个工作，这篇是苏黎世联邦理工学院发表在PLDI 2020上的，借助MLIR的设计思想，想在EDA领域设计一个统一的IR。

<!-- more -->

## Motivation

1. 单位面积上集成的晶体管数量逐渐变多导致现在数字设计从前端到硅的过程中涉及的工具链非常deep，包括simulators, formal verifiers, linters, synthesizers, logical equivalence checkers, and many other tools.
2. 几乎所有的EDA工具他们都是针对Verilog、VHDL这一些上世纪的语言来做分析的，但是这些HDL非常复杂（几百上千页的Language Reference Manual）
3. 其次，这些EDA工具分别由不同的厂商提供，他们对这些语言的处理也是不一样的，这里应该是指不同的EDA工具分析Verilog代码的时候使用的是不同的IR设计，这会导致很多问题，比如我们为了使我们的代码在综合这个阶段生成的网表文件能够实现和我们使用仿真器仿真预期的功能一致，我们就得惯着这些综合器，小心翼翼的写代码，从而很少会使用到Verilog和Systemverilog内部的一些高级的语法，就好像一些用来仿真的Verilog代码是不可综合的。所以大家都致力于写代码的时候按照约定好的规范，甚至专门做了一些linting tools来尽可能使用能够让这个代码在一整个workflow上都能pass的subset。
4. EDA市场被三大家一直牢牢把控着，闭源发展了几十年，外人想要进军首先就需要解决复杂的前端设计，才可以考虑竞争，**一些现有的开源EDA都会有一些大大小小的问题**（之前在ChinaDA上有调研现有开源EDA的Talk，后悔没好好听了。

于是作者提出了LLHD作为一个可以在整个集成电路设计过程都可以使用的通用的中间表示，但是用一个IR显然是不够的，既要顾及到综合，也要考虑到仿真，LLHD是一个多层的中间表示（MLIR），包含了三个dialect。

Behavioural LLHD： 高层次的表示，行为级的描述，SystemVerilog等高层次的HDL会被先转到这个Dialect，对这个Dialect我们可以来做仿真和设计验证。

Structural LLHD：在给综合器综合之前，我们需要把Behavioural LLHD lower到Structural LLHD。值得注意的是，我们一般把网表的verilog形式称为structural verilog，所以这里的Structural LLHD和下面的Netlist有什么差别的？

Netlist LLHD：Paper里说，经过综合器综合的网表文件可以被转化为Netlist LLHD的形式（Paper里说使用的是第三方的综合器，**所以到底是什么综合器可以接受LLHD的中间表示呢？**）。根据我的理解，现在好像没有综合器可以直接接受LLHD的IR，他的意思是现在的综合器综合生成的netlist可以转换成Netlist LLHD这种中间表示。

这三者是逐级向下包含的关系。

LLHD参考了LLVM IR的基本语法，并且自己增加了一些方便模拟电路的支持并发的特性，还附带开源了一个针对HDL的编译器前端moore.

![4DBA40A1-8CC1-4479-827D-0AB0DAD32E6D](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/4DBA40A1-8CC1-4479-827D-0AB0DAD32E6D.png)作者说，SSA的数据流图和电路图的形式很相似，每个电路图可以近似于数据流图中的一个节点。

其他的可以Synthesis的IR：

Many IRs beside FIRRTL are engineered to interact with hardware synthesizers. LNAST [28] targets the representa- tion of the synthesizable parts of an HDL circuit description in a language-agnostic AST. RTLIL [30] is part of the “Yosys” open-source tool suite and focuses mainly on logic synthesis. It cannot represent higher-level constructs such as aggregate types or conditional assignment. μIR [24] is geared towards HLS design flows and tries to capture entire accelerator archi- tectures. These IRs are very specifically crafted to transport designs into a synthesizer and focus solely on this part of the flow.

关于他们具体的实现，就不细看了，，因为是rust写的么。

怎样实现的LLHD，他们提供了一个前端叫做moore，可以将sv和vhdl的代码转化成LLHD的IR，但是这个moore是用rust写的..一下子打消了我深入看下去的想法；除了moore，他们还提供了一个llhd-sim的针对LLHD IR的前端仿真工具，可以用gtkwave来看波形。

这篇paper的出发点和CIRCT是一致的，是想打破现在的EDA工具都在各做各的局面，但是真的有人会用嘛...，虽然llhd-sim也是rust写的，但是现在已经集合到circt里了，貌似只需要cpp就可以跑了，llhd现在也作为一个dialect在circt里出现。
