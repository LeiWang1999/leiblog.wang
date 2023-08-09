---
title: Parallels Desktop 安装Kali

categories:
  - Technical
tags:
  - Kali
  - Crack
date: 2020-08-16 10:54:16
---

![Banner](http://leiblog.wang/static/image/2020/8/TTJii9.jpg)

最后还是在 Mac 上安装了 Kali 的虚拟机，还好比预期小很多，整个机器的大小也就 10Gb+，Mini but powerful！

（其实主要是馋那几个有趣的图形化工具

记录踩坑过程！！！

<!-- more -->

### 一、软件下载

我使用 Parallels Desktop14 来安装虚拟机，现在最新的是 15 版，但是我安装了破解版之后体验极差，于是还是使用 14 来安装了。

Kali 选取的是安装时候的最新版本，2020.2，可以去 Kali 官网下载最新的镜像。

如果觉得下载速度较慢，可以前往 Material 页面，下载我使用的镜像文件：[Material](http://leiblog.wang/material/)。

[Parallels Desktop 14](http://leiblog.wang/static/2020-08-16/Parallels_Desktop_14.1.2-45485_iMac.hk_.dmg) 解压密码：imac.hk

[Kali 2020.2 ISO](http://leiblog.wang/static/2020-08-16/kali-linux-2020.2-installer-amd64.iso)

### 二、安装 Kali

下面其实是网上嫖的 2018 的安装过程([原博客地址](https://www.cnblogs.com/artwalker/p/13235757.html))，界面可能会有点不一样，不要问我为什么不自己录，纯粹是懒 :/ 。

况且，安装部分完全一致。

首先在 Parallels Desktop 里面新建虚拟机：

![img](http://leiblog.wang/static/image/2020/8/rxFAiy.jpg)

手动选择，应该是识别不了的，因为 kali 的版本太新，而 Parallels 的破解版本旧了。

![img](http://leiblog.wang/static/image/2020/8/2nOXi4.jpg)

![img](http://leiblog.wang/static/image/2020/8/oNjfel.jpg)

![img](http://leiblog.wang/static/image/2020/8/cW2bEh.jpg)

给虚拟机分配硬盘空间，类似 Windows 里面 C 盘 D 盘 E 盘的大小

![img](http://leiblog.wang/static/image/2020/8/P6soFB.jpg)

硬盘可以点开高级设置，进去设置大小，这里选择的是默认 64GB

![img](http://leiblog.wang/static/image/2020/8/LAMdez.jpg)

![img](http://leiblog.wang/static/image/2020/8/OHjE63.jpg)

![img](http://leiblog.wang/static/image/2020/8/TC4fxc.jpg)

![img](http://leiblog.wang/static/image/2020/8/0uq2yz.jpg)

之后等待：

![img](http://leiblog.wang/static/image/2020/8/97LDX8.jpg)

![img](http://leiblog.wang/static/image/2020/8/fmpcY9.jpg)

![img](http://leiblog.wang/static/image/2020/8/QU00L8.jpg)

![img](http://leiblog.wang/static/image/2020/8/dgJXJF.jpg)

设置密码，也是开机密码，一定要记住。

![img](http://leiblog.wang/static/image/2020/8/KfLWFH.jpg)

![img](http://leiblog.wang/static/image/2020/8/z1LcWU.jpg)

![img](http://leiblog.wang/static/image/2020/8/icjvcx.jpg)

![img](http://leiblog.wang/static/image/2020/8/zctsSl.jpg)

![img](http://leiblog.wang/static/image/2020/8/k21gBr.jpg)

![img](http://leiblog.wang/static/image/2020/8/jL5WIi.jpg)

![img](http://leiblog.wang/static/image/2020/8/r7nGgS.jpg)

![img](http://leiblog.wang/static/image/2020/8/JteO75.jpg)

![img](http://leiblog.wang/static/image/2020/8/Xgvezv.jpg)

![img](http://leiblog.wang/static/image/2020/8/OrV3pw.jpg)

![img](http://leiblog.wang/static/image/2020/8/TJwQOS.jpg)

安装到此结束。

### 三、安装 Parallels Tool

不安装 Parallels Tool，就无法和 Mac 共享文件、粘贴板之类的功能，2019.4 版本之后的 Kali 已经无法使用 Parallels 官方提供的方案安装 tool 了。

这里直接提供修改过后的 tool 安装包:http://leiblog.wang/static/2020-08-16/pt_kali2020.2.zip

解压到任意文件夹，然后执行命令:

```bash
./install
```

注意这边要给上 root 权限。

另外，安装完成之后需要重新启动，重启之后需要在 设置->会话和启动 页面关闭 Parallel 的开机自启动，否则每次重启都会有弹窗。
