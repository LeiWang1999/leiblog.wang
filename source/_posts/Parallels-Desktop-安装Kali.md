---
title: Parallels Desktop 安装Kali
top: 3
categories:
  - Technical
tags:
  - Kali
  - Crack
date: 2020-08-16 10:54:16
---

![Banner](http://leiblog.wang/static/image/2020/8/TTJii9.jpg)

最后还是在Mac上安装了Kali的虚拟机，还好比预期小很多，整个机器的大小也就10Gb+，Mini but powerful！

（其实主要是馋那几个有趣的图形化工具

记录踩坑过程！！！

<!-- more -->

### 一、软件下载

我使用Parallels Desktop14来安装虚拟机，现在最新的是15版，但是我安装了破解版之后体验极差，于是还是使用14来安装了。

Kali选取的是安装时候的最新版本，2020.2，可以去Kali官网下载最新的镜像。

如果觉得下载速度较慢，可以前往Material页面，下载我使用的镜像文件：[Material](http://leiblog.wang/material/)。

[Parallels Desktop 14](http://leiblog.wang/static/2020-08-16/Parallels_Desktop_14.1.2-45485_iMac.hk_.dmg)  解压密码：imac.hk

[Kali 2020.2 ISO](http://leiblog.wang/static/2020-08-16/kali-linux-2020.2-installer-amd64.iso)

### 二、安装Kali

下面其实是网上嫖的2018的安装过程([原博客地址](https://www.cnblogs.com/artwalker/p/13235757.html))，界面可能会有点不一样，不要问我为什么不自己录，纯粹是懒 :/ 。

况且，安装部分完全一致。

首先在Parallels Desktop里面新建虚拟机：

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171320901-2113732267.png)

手动选择，应该是识别不了的，因为kali的版本太新，而Parallels的破解版本旧了。

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171330059-418949337.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171339029-671599935.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171347594-1256653510.png)

给虚拟机分配硬盘空间，类似Windows里面C盘D盘E盘的大小

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171357425-342661430.png)

硬盘可以点开高级设置，进去设置大小，这里选择的是默认64GB

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171405803-1298137062.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171412580-1567246561.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171418923-1241681664.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171428124-1892448208.png)

之后等待：

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171436226-1297045846.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171444800-1596551660.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171451958-1876715650.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171459154-2075389277.png)

设置密码，也是开机密码，一定要记住。

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171506589-242016490.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171516464-1100602964.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171525085-713339934.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171533255-1613473902.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171544169-1999842356.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171553391-1853428967.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171600854-1850816467.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171613150-1754523322.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171622401-255391434.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171631490-155942400.png)

![img](https://img2020.cnblogs.com/blog/1994352/202007/1994352-20200704171640260-90378932.png)

安装到此结束。

### 三、安装Parallels Tool

不安装Parallels Tool，就无法和Mac共享文件、粘贴板之类的功能，2019.4版本之后的Kali已经无法使用Parallels官方提供的方案安装tool了。

这里直接提供修改过后的tool安装包:http://leiblog.wang/static/2020-08-16/pt_kali2020.2.zip

解压到任意文件夹，然后执行命令:

```bash
./install
```

注意这边要给上root权限。

另外，安装完成之后需要重新启动，重启之后需要在 设置->会话和启动 页面关闭Parallel的开机自启动，否则每次重启都会有弹窗。

