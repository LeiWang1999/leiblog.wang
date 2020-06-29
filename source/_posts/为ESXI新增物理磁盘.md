---
categories:
  - Technical
tags:
  - ESXI
date: 2020-04-06 15:18:46	
---

在之前的一篇文章里讲述了如何为 ESXI 系统上的虚拟机挂载新的存储，由此我又想到了另外一件事情，如何为 ESXI 系统本身挂载存储呢？

恰好，这件事情我还真的做过，一年前为我们工作室的 MacPro 置备了新的 2TB 机械硬盘，上面运行的也是 ESXI 系统，为此也踩了一些小坑，俾如说尝试了通过 ssh 链接到到 MacPro 上，但是找不到已经插上去的物理磁盘，经过百般折腾后终于找到了正确的姿势：为了给 ESXI 系统挂载磁盘，我们仍然需要使用 VsPhere Client。

<!-- more -->

##### 首先、将机械硬盘插到主机上

##### 然后、在 Vsphere 客户端的主机界面

##### 选择配置界面

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200406151607786.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70#pic_center)

##### 然后在存储器界面点击添加存储器

接着按照提示操作就好了，他会自动扫描已经插上的硬盘。
