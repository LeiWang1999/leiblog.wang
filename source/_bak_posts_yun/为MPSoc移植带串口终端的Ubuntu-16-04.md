---
title: 为MPSoc移植带串口终端的Ubuntu 16.04
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2021-07-24 20:56:45
---

![](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/20210725151558.png)

为接下来要尝试基于 Tengine 完成 CPU Fallback 的工作，在组里的 ZCU 102 开发板上进行了 NVDLA 的移植。本来以为这个过程会很顺利，没想到还是因为各种问题还是花费了一个星期的时间。

这篇文章记述了部分关于 ZCU102 rev1.1 板卡的坑点，以及利用chroot基于Ubuntu Base Rootfs来订制 aarch64 ubuntu 的文件系统流程。

<!-- more -->

#### ZCU 102 rev 1.1 DDR 配置修改

将 NVDLA 的 RTL 裁剪并打包成 IP，在 Vivado 中完成 BlockDesign 设计，与之前的文章里是一样的，但是笔者使用的 Vivado 版本是 Vivado 2019.1，在软件内自带的板卡文件里 ZCU 102 的 rev 是 1.0，两者的区别在 Xilinx 的[官方文档](https://www.xilinx.com/support/answers/71961.html)里有记录：

![image-20210725125249589](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210725125249589.png) 

要把载入板卡文件之后的 DDR 的参数进行如图所示的修改，否则在 SDK 里连 DDR 的 Memory Test 都过不去，更不提跑 sanity 了。

#### Petalinux 为什么不行

Petalinux 生成 BOOT.BIN 和 image.ub 的流程与之前的文章里讲述的是一样的，值得注意的是原来官方提供的预构件的 runtime 有基于 arm64 编译，可以直接在 petalinux 上运行，不需要我们重新编译。

由于 ZCU102 是官方的开发板，我们在 petalinux 的配置页面：

```bash
petalinux-config --get-hw-description=./
```

选中`DTG settings`–>`(template) MACHINE NAME`, enter进入修改为开发板版本，改成`zcu102-rev1.0`，这个时候会自动的去导入官方写好的设备树，此时以太网、DP显示器之类的外设都会挂载上来。

如果不这样做，除非自己在设备树里加上这些节点，否则在系统启动之后会没有以太网的连接。

本来，在 petalinux 进行 rootfs 订制的时候可以添加一些常用的工具，像是 ssh、gcc、g++，make，都可以提前安装好，但是因为其没有包管理工具，导致在工作的时候仍然会碰到很多工具的缺失，于是还是狠下心来换成了 Ubuntu。

### 和之前文章里的Ubuntu有什么区别？

这次烧写的根文件系统是基于 ubuntu-base 更改的，可以在ubuntu的[官网](http://cdimage.ubuntu.com/ubuntu-base/releases/16.04/release/)下载。而在之前的文章里，我为 ZYNQ 7000 替换的 ubuntu 的根文件系统是 ubuntu-16.04.2-minimal-armhf-2017-06-18。一个是针对 arm64 的、一个是针对 32位操作系统的。

虽然，在64位的处理器上可以运行32位的操作系统，但是运行nvdla的runtime还是出现了一些问题，根据我漫长的debug，好像是因为做虚拟内存映射的时候32位的地址不够用导致mmap错误 :(

于是，我舍弃了这个32位的操作系统，换成了 arm64 的，这个过程是本文的主要内容。

因为如果和原来的教程一样直接解压缩，基于ubuntu-base的操作系统会把默认终端开到DP显示器上，这意味着需要外接一个显示器与键盘才可以工作，这无疑是很麻烦的，所以本文需要把默认的终端替换到串口上。

此外，基于ubuntu-base的操作系统，没有默认的用户和密码，需要我们在替换之前提前添加进来。

**本文都应该在ubuntu的环境下运行。**

### 针对 Aarch64 的 Ubuntu Rootfs 制作

#### 1. 下载、解压 rootfs

```bash
mkdir ~/rootfs && cd ~/rootfs
wget http://cdimage.ubuntu.com/ubuntu-base/releases/16.04.1/release/ubuntu-base-16.04.2-base-arm64.tar.gz
mkdir ubuntu-rootfs
tar -xvf ubuntu-base-16.04.2-base-arm64.tar.gz -C ubuntu-rootfs
cd ubuntu-rootfs
ls
bin   dev  home  media  opt   root  sbin  srv  system  usr
boot  etc  lib   mnt    proc  run   snap  sys  tmp     var
```

#### 2 安装qemu-user-static

```bash
apt-get install qemu-user-static
cp /usr/bin/qemu-aarch64-static  usr/bin
cp -b /etc/resolv.conf etc/
```

#### 3.启动 chroot

```bash
cd ../
vi ch-mount.sh
```

在 ch-mount.sh 里填入如下脚本：

```bash
#!/bin/bash
# 
function mnt() {
    echo "MOUNTING"
    sudo mount -t proc /proc ${2}proc
    sudo mount -t sysfs /sys ${2}sys
    sudo mount -o bind /dev ${2}dev
    sudo mount -o bind /dev/pts ${2}dev/pts		
    sudo chroot ${2}
}
function umnt() {
    echo "UNMOUNTING"
    sudo umount ${2}proc
    sudo umount ${2}sys
    sudo umount ${2}dev/pts
    sudo umount ${2}dev
}
if [ "$1" == "-m" ] && [ -n "$2" ] ;
then
    mnt $1 $2
elif [ "$1" == "-u" ] && [ -n "$2" ];
then
    umnt $1 $2
else
    echo ""
    echo "Either 1'st, 2'nd or both parameters were missing"
    echo ""
    echo "1'st parameter can be one of these: -m(mount) OR -u(umount)"
    echo "2'nd parameter is the full path of rootfs directory(with trailing '/')"
    echo ""
    echo "For example: ch-mount -m /media/sdcard/"
    echo ""
    echo 1st parameter : ${1}
    echo 2nd parameter : ${2}
fi
```

通过ch-mount.sh脚本chroot到arm64的文件系统下

```shell
./ch-mount.sh -m ubuntu-rootfs/
```

**这里注意，因为脚本里写的是字符串拼接，所以命令最后的/不能少**

这样，就可以随便操作rootfs了。

####  4. Rootfs 定制

首先，可以安装一些软件

```bash
apt update -y && apt upgrade -y && apt-get -y install \
  language-pack-en-base \
  sudo \
  ssh \
  net-tools \
  network-manager \
  iputils-ping \
  rsyslog \
  bash-completion
```

添加用户，账户名为 ubuntu、密码也为 ubuntu

```bash
useradd -s '/bin/bash' -m -G adm,sudo ubuntu
echo ubuntu:ubuntu | chpasswd
```

设置主机名

```bash
echo 'ubuntu.zcu102' > /etc/hostname
```

其次，还要更新host，否则本地主机名无法解析会导致sudo等一些命令出现问题

```bash
echo '127.0.0.1 ubuntu.zcu102' >> /etc/hosts
```

配置默认的shell为串口：

在/etc/init/下，添加一个新文件，名叫ttyS0.conf

```bash
start on stopped rc or RUNLEVEL=[12345]
stop on RUNLEVEL [!12345]
respawn
exec /sbin/getty -L 115200 ttyPS0 vt102
```

其中，ttyPS0根据机器不同而不同，在ZCU102上，UART0对应的就是 ttyPS0、UART1对应的是 ttyPS1。

制作完成之后，退出，取消挂载，并打包：

```shell
exit
./ch-mount.sh -u ubuntu-rootfs/
tar zcvf ubuntu-rootfs.tar.gz -C ubuntu-rootfs/ .
```

这样打包可以取消顶层文件夹，否则打包出来最外层会有一个ubuntu-rootfs的文件夹。

最后，解压缩到 rootfs 中：

```bash
sudp tar xvfp ubuntu-rootfs.tar.gz -C /media/ubuntu/rootfs/
```

大功告成啦！

