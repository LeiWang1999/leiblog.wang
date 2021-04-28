---
title: NVDLA Xilinx FPGA Mapping
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2021-04-28 15:16:34
---

NVDLA 是英伟达于2017年开源出来的深度学习加速器框架。可惜的是，这个项目被开源出来一年后就草草停止维护了。

笔者本科的毕业设计为了与实验室研究的方向贴合，把NVDLA的RTL映射到了 Xilinx FPGA 上，并且上板编译了 Runtime 。这个过程大概花费了两个月的时间，映射成功后，很多伙伴对上板的过程很感兴趣，而这个步骤亦不是使用聊天软件说两句就可以概述的。于是写下这篇文章，记述Mapping 到 FPGA 过程中踩过的一些坑。

开发板：Zynq 7000+ / Zynq MPSOC

软件环境：

- Ubuntu 18.04

- Vivado 2019.1
- Petalinux 2019.1

<!-- more -->

## 硬件系统设计概述

本文采用的硬件版本是[hw仓库](https://github.com/nvdla/sw)的master分支，v1的spec文件仅提供了full版本，应该没有FPGA能够塞得下。master 分支提供了 small 和 large 两个版本的spec 文件，我们使用 small 的配置，当然这个过程对 large 也是适用的，接下来首先讲一下如何生成rtl。

如果你会chisel，还可以弯道超车，参考画面大佬的[soDLA](https://github.com/soDLA-publishment/soDLA)项目，也能生成NVDLA的RTL代码。

如果使用官方的仓库生成RTL，我们需要给把官方的 tmake 搭建出来，这个过程中需要安装很多环境，例如 python、Java、perl、verilator，在这里我们就不用污染自己的环境了，利用docker使用别人安装好的环境：

```bash
docker pull farhanaslam/nvdla 
```

启动docker之后，使用 tmake 生成 RTL

```bash
root@1d0954a2d18b:/usr/local/nvdla/nvdla_hw# ./tools/bin/tmake -build vmod
[TMAKE]: building nv_small in spec/defs 
[TMAKE]: building nv_small in spec/manual 
[TMAKE]: building nv_small in spec/odif 
[TMAKE]: building nv_small in vmod/vlibs 
[TMAKE]: building nv_small in vmod/include 
[TMAKE]: building nv_small in vmod/rams/model 
[TMAKE]: building nv_small in vmod/rams/synth 
[TMAKE]: building nv_small in vmod/rams/fpga/model 
[TMAKE]: building nv_small in vmod/fifos 
[TMAKE]: building nv_small in vmod/nvdla/apb2csb 
[TMAKE]: building nv_small in vmod/nvdla/cdma 
[TMAKE]: building nv_small in vmod/nvdla/cbuf 
[TMAKE]: building nv_small in vmod/nvdla/csc 
[TMAKE]: building nv_small in vmod/nvdla/cmac 
[TMAKE]: building nv_small in vmod/nvdla/cacc 
[TMAKE]: building nv_small in vmod/nvdla/sdp 
[TMAKE]: building nv_small in vmod/nvdla/pdp 
[TMAKE]: building nv_small in vmod/nvdla/cfgrom 
[TMAKE]: building nv_small in vmod/nvdla/cdp 
[TMAKE]: building nv_small in vmod/nvdla/bdma 
[TMAKE]: building nv_small in vmod/nvdla/rubik 
[TMAKE]: building nv_small in vmod/nvdla/car 
[TMAKE]: building nv_small in vmod/nvdla/glb 
[TMAKE]: building nv_small in vmod/nvdla/csb_master 
[TMAKE]: building nv_small in vmod/nvdla/nocif 
[TMAKE]: building nv_small in vmod/nvdla/retiming 
[TMAKE]: building nv_small in vmod/nvdla/top 
[TMAKE]: Done nv_small
[TMAKE]: nv_small: PASS
```

输出的RTL文件会在 `out\nv_small\vmod`里，但是如果直接在Vivado里引入vmod文件夹会存在一些问题，因为其内部的ram是行为级描述，我们需要替换成Bram、将`rams\synth`删除即可。

## 软件系统设计概述

