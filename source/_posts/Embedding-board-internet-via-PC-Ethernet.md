---
title: Embedding board internet via PC Ethernet

categories:
  - Technical
tags:
  - EEEE
date: 2020-10-19 13:50:49
---

最近在使用`PYNQ`,`Jetson Nano`进行一些开发，但是迫于实验室需要外接校园网、使用图形页面登陆所以嵌入式设备无法直接上网。本文给出的解决方案是将嵌入式设备通过以太网连接到个人主机上，然后将以太网桥接到 Wifi，访问因特网。

我分别在`Ubuntu20.04`与`Windows10`上实践了两方案，核心只有两句话，网线接 lan 口，关了 dhcp。

**你需要准备的：**

1. 能通过无线上网的 Windows 主机、Ubuntu 主机一台
2. 以太网线一根
3. 嵌入式开发板一块

<!-- more -->

### Windows 版

1. 打开网络适配器选项，将 WLAN->右击属性->共享。选择你想桥接到的以太网，钩上第一个选项。

![Wifi](http://leiblog.wang/static/image/2020/10/7CCD085B06EDAFD4651B0782D2BA77D5.png)

这样你的以太网就会被默认分配到 192.168.137.1 这个静态 ip。

![](http://leiblog.wang/static/image/2020/10/90BF353847EB0B66EF8F227E9A1BB828.png)

然后我们使用串口助手或者是别的一些手段，登录到 PYNQ/Jetson 板卡上，配置板卡的 ip。

```zsh
sudo vim /etc/network/interfaces
```

修改内容如下：

```zsh
auto eth0
iface eth0 inet static
address 192.168.137.101 # 此处改为192.168.137.x，x为2～255内的任意数
netmask 255.255.255.0
gateway 192.168.137.1

source-directory /etc/network/interfaces.d
```

然后重启板卡，ping 一下百度。

```zsh
ping www.baidu.com
```

意外发现翻车了，ping 不通，但是主机可以 ping 通板卡，板卡不能 ping 通主机，想着可能是路由表坏掉了。

```zsh
sudo route # 查看路由表
```

发现是默认的路由有问题，敲一下这两条命令就好了

```zsh
sudo route del default
sudo route add default gw 192.168.137.1 netmask 0.0.0.0
```

意思是，所有的出口都去找 192.168.137.1

然后，查看一下路由表，大功告成。

### Ubuntu 版

命令行里输入

```zsh
nm-connection-editor
```

在界面里设置以太网->有线连接->IPV4 设置

![](http://leiblog.wang/static/image/2021/1/bCUr9q.jpg)

在 Method 里选择与其他计算机共享，最后在地址栏里增加静态 ip，完成。

这里需要注意，如果有能连接上板卡，但是板卡还是没有办法上网的时候，要看看是不是连错以太网口了，像我这个 Ubuntu 就有两张以太网卡。
