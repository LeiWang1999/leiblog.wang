---
top: 10
title: 千兆以太网视频传输
categories:
  - Technical
tags:
  - FPGA
  - Ethernet
date: 2020-07-19 17:15:34
---

![Banner](http://leiblog.wang/static/image/2020/7/swedishfishing.jpg)

我接触到的每一块 FPGA Evaluation Board 上都配有以太网口，于是我总觉得以太网的协议一定要学。

加上疫情，学校的实验室居然不开！现在迫切的想验证一些图像算法，我可不想在狭隘的宿舍空间里给 FPGA 再接一块 VGA/HDMI 接口的显示器。于是，这是一篇探索利用以太网传输视频讯号，通过 PC 机来显示的文章。

本文使用的是 ZYNQ 器件，事实上，使用纯 Verilog 实现的以太网传输视频网上也找到了一些 demo，这应该是因为 UDP 传输使用 Verilog 更高效（不一定，但 Verilog 写 TCP 协议的至今都没有能普遍商用的解决方案），因为我使用的是 ZYNQ 器件，准确的说是`PYNQ-Z2`,以太网口在 PS 端，没法套用那些工程，于是就有了挖坑、踩坑、填坑的过程，就有了这篇博客，就有了新的 Github 的仓库。

Github Page: https://github.com/LeiWang1999/EthernetVideo

**PS：求 Follow、Star、PR！QAQ**

<!-- more -->

## 1. Tips Before Experiment

推荐阅读[UG 585](https://www.xilinx.com/support/documentation/user_guides/ug585-Zynq-7000-TRM.pdf)，这是官方提供的，最为详实的 ZYNQ 器件教学文档，其中有关以太网的驱动以及 Vivado SDK 提供的几个有关 lwip 的 Template 说明都可以在**Chapter 16 Gigabit Ethernet Controller**找到。

![Chapter16](http://leiblog.wang/static/image/2020/7/fc6aDJ.png)

其实国内的相关教程，基本都是翻译的 Xilinx 的文档，毕竟如果不看这个文档，编写 Arm 端程序的时候根本无从下手，比如我应该导入哪个库，如何初始化等等。

另外，因为我曾经自学过计算机网络，所以对于 UDP、TCP、IPV4、IPV6 这些关键词并不陌生，如果你不清楚或许你应该去补一下计算机网络的知识。

### What is UDP/TCP

TCP/IP 协议集包括应用层,传输层，网络层，网络访问层。

##### 其中应用层包括:

1、超文本传输协议（HTTP）:万维网的基本协议；
2、文件传输（TFTP 简单文件传输协议）；
3、远程登录（Telnet），提供远程访问其它主机功能, 它允许用户登录 internet 主机，并在这台主机上执行命令；
4、网络管理（SNMP 简单网络管理协议），该协议提供了监控网络设备的方法， 以及配置管理,统计信息收集,性能管理及安全管理等；
5、域名系统（DNS），该系统用于在 internet 中将域名及其公共广播的网络节点转换成 IP 地址。

##### 其次网络层包括:

1、Internet 协议（IP）；
2、Internet 控制信息协议（ICMP）；
3、地址解析协议（ARP）；
4、反向地址解析协议（RARP）。

##### 网络访问层:

网络访问层又称作主机到网络层（host-to-network），网络访问层的功能包括 IP 地址与物理地址硬件的映射， 以及将 IP 封装成帧.基于不同硬件类型的网络接口，网络访问层定义了和物理介质的连接. 当然我这里说得不够完善，TCP/IP 协议本来就是一门学问，每一个分支都是一个很复杂的流程， 但我相信每位学习软件开发的同学都有必要去仔细了解一番。

#### 下面着重讲解一下 TCP 协议和 UDP 协议的区别

TCP（Transmission Control Protocol，传输控制协议）是面向连接的协议，也就是说，在收发数据前，必须和对方建立可靠的连接。 一个 TCP 连接必须要经过三次“对话”才能建立起来，其中的过程非常复杂， 只简单的描述下这三次对话的简单过程：

1）主机 A 向主机 B 发出连接请求数据包：“我想给你发数据，可以吗？”，这是第一次对话；

2）主机 B 向主机 A 发送同意连接和要求同步 （同步就是两台主机一个在发送，一个在接收，协调工作）的数据包 ：“可以，你什么时候发？”，这是第二次对话；

3）主机 A 再发出一个数据包确认主机 B 的要求同步：“我现在就发，你接着吧！”， 这是第三次对话。

三次“对话”的目的是使数据包的发送和接收同步， 经过三次“对话”之后，主机 A 才向主机 B 正式发送数据。

#### TCP 三次握手过程

第一次握手：主机 A 通过向主机 B 发送一个含有同步序列号的标志位的数据段给主机 B，向主机 B 请求建立连接，通过这个数据段， 主机 A 告诉主机 B 两件事：我想要和你通信；你可以用哪个序列号作为起始数据段来回应我。

第二次握手：主机 B 收到主机 A 的请求后，用一个带有确认应答（ACK）和同步序列号（SYN）标志位的数据段响应主机 A，也告诉主机 A 两件事：我已经收到你的请求了，你可以传输数据了；你要用那个序列号作为起始数据段来回应我

第三次握手：主机 A 收到这个数据段后，再发送一个确认应答，确认已收到主机 B 的数据段："我已收到回复，我现在要开始传输实际数据了，这样 3 次握手就完成了，主机 A 和主机 B 就可以传输数据了。

**3 次握手的特点**：没有应用层的数据 ,SYN 这个标志位只有在 TCP 建立连接时才会被置 1 ,握手完成后 SYN 标志位被置 0。

TCP 建立连接要进行 3 次握手，而断开连接要进行 4 次

第一次： 当主机 A 完成数据传输后,将控制位 FIN 置 1，提出停止 TCP 连接的请求 ；

第二次： 主机 B 收到 FIN 后对其作出响应，确认这一方向上的 TCP 连接将关闭,将 ACK 置 1；

第三次： 由 B 端再提出反方向的关闭请求,将 FIN 置 1 ；

第四次： 主机 A 对主机 B 的请求进行确认，将 ACK 置 1，双方向的关闭结束.。

由 TCP 的三次握手和四次断开可以看出，TCP 使用面向连接的通信方式， 大大提高了数据通信的可靠性，使发送数据端和接收端在数据正式传输前就有了交互， 为数据正式传输打下了可靠的基础。

#### 名词解释

1、ACK 是 TCP 报头的控制位之一，对数据进行确认。确认由目的端发出， 用它来告诉发送端这个序列号之前的数据段都收到了。 比如确认号为 X，则表示前 X-1 个数据段都收到了，只有当 ACK=1 时,确认号才有效，当 ACK=0 时，确认号无效，这时会要求重传数据，保证数据的完整性。

2、SYN 同步序列号，TCP 建立连接时将这个位置 1。

3、FIN 发送端完成发送任务位，当 TCP 完成数据传输需要断开时,，提出断开连接的一方将这位置 1。

#### TCP 的包头结构：

源端口 16 位；目标端口 16 位；序列号 32 位；回应序号 32 位；TCP 头长度 4 位；reserved 6 位；控制代码 6 位；窗口大小 16 位；偏移量 16 位；校验和 16 位；选项 32 位(可选)；

这样我们得出了 TCP 包头的最小长度，为 20 字节。

#### UDP（User Data Protocol，用户数据报协议）

1、UDP 是一个非连接的协议，传输数据之前源端和终端不建立连接， 当它想传送时就简单地去抓取来自应用程序的数据，并尽可能快地把它扔到网络上。 在发送端，UDP 传送数据的速度仅仅是受应用程序生成数据的速度、 计算机的能力和传输带宽的限制； 在接收端，UDP 把每个消息段放在队列中，应用程序每次从队列中读一个消息段。

2、 由于传输数据不建立连接，因此也就不需要维护连接状态，包括收发状态等， 因此一台服务机可同时向多个客户机传输相同的消息。

3、UDP 信息包的标题很短，只有 8 个字节，相对于 TCP 的 20 个字节信息包的额外开销很小。

4、吞吐量不受拥挤控制算法的调节，只受应用软件生成数据的速率、传输带宽、 源端和终端主机性能的限制。

5、UDP 使用尽最大努力交付，即不保证可靠交付， 因此主机不需要维持复杂的链接状态表（这里面有许多参数）。

6、UDP 是面向报文的。发送方的 UDP 对应用程序交下来的报文， 在添加首部后就向下交付给 IP 层。既不拆分，也不合并，而是保留这些报文的边界， 因此，应用程序需要选择合适的报文大小。

我们经常使用“ping”命令来测试两台主机之间 TCP/IP 通信是否正常， 其实“ping”命令的原理就是向对方主机发送 UDP 数据包，然后对方主机确认收到数据包， 如果数据包是否到达的消息及时反馈回来，那么网络就是通的。

**ping 命令**是用来探测主机到主机之间是否可通信，如果不能**ping**到某台主机，表明不能和这台主机建立连接。**ping 命令**是使用 IP 和网络控制信息协议 (ICMP)，因而没有涉及到任何传输协议(UDP/TCP) 和应用程序。它发送 icmp 回送请求消息给目的主机。

ICMP 协议规定：目的主机必须返回 ICMP 回送应答消息给源主机。如果源主机在一定时间内收到应答，则认为主机可达。

#### UDP 的包头结构

源端口 16 位；目的端口 16 位；长度 16 位；校验和 16 位

#### 小结 TCP 与 UDP 的区别

1、基于连接与无连接；

2、对系统资源的要求（TCP 较多，UDP 少）；

3、UDP 程序结构较简单；

4、流模式与数据报模式 ；

5、TCP 保证数据正确性，UDP 可能丢包；

6、TCP 保证数据顺序，UDP 不保证。

### What is IPV4/IPV6

IPv6 和 IPv4，都是给网络中的主机编址的一种方式。

网络数据要从一个主机传送到另外一个主机，就像把快递从北京送到石家庄，你得有地址。而 IPv4，或者 IPv6，解决的就是这个网络空间的编织问题。

**那么 IPv6 是什么？**

是接入网络的一台主机的网络地址。准确地说，是给主机编址的一种方式。

**IPv4 和 IPv6 有什么不一样？**

IPv4 用 32 位的二进制位来表示一台主机的网络地址；而 IPv6 用 128 位二进制位来表示一台主机的网络地址。

我们日常看到的 8.8.8.8 这样的地址，只是一种便于人类记忆和观察的表示方式而已，8.8.8.8 的实际地址表示成二进制应该是 00000100000001000000010000000100

IPv6 也有更便于人类记忆和观察的表示方式，比如 2001:fecd:ba23:cd1f:dcb1:1010:9234:4088，但在计算机网络中，它的实际地址仍然是一个二进制。

显而易见，无论是 32 位的网络地址，还是 128 位的网络地址，其能够提供的地址个数都是可数的，有上限的。32 位的 IPv4 地址，大约能提供 43 亿个设备接入互联网，当初设计 IP 协议的人觉得，43 亿还不够你们玩？结果我们看到了，确实不够玩。所以 IPv6 作为一个能提供更多可用地址的协议被设计出来了，那么 128 位的 IPv6，能提供多少地址呢？

太多了。超过了宇宙中原子数量的总和（作为参考，宇宙中原子数的总和大概是 10^80 这个数量级，而 IPv6 提供的是 10^124 这个数量级），因此，IPv6 从理论上讲，是不可能用尽的。

关于计算机位数，我一直很喜欢一段超级浪漫的话，大概所有的指数级增长都适用这句话：

在用 32 位二进制数表示时间戳的计算机上，2038 年这个时间戳将会溢出。但在用 64 位二进制数表示时间戳的计算机上，当这个二进制数溢出时，那大概是 2920 亿年。到那时，位于猎户座旋臂的太阳，已经是黑矮星或暗黑物质，猎户座旋臂已经被重力波震断，银河系大概则已经变成小型似星体了。

数学的魔力，指数的魔力。

**从「IPv4」过渡到「IPv6」需要哪些准备工作？**

其实我们一直在做各种准备工作了。

1. 在 IPv6 全面铺开之前，通过一些方式延缓 IPv4 耗尽的速度。

按照互联网发展的实际速度，接入互联网的主机越来越多，早已超过了 IPv4 的容纳极限。为了延缓 IPv4 耗尽的速度，为 IPv6 的建设争取时间，互联网早都应用了 NAT 技术，通过多层接入的方式，把一个公网 IP 地址共享给好几台设备使用。这样一个家庭或者一个组织，内部可能接入了很多主机，但其在公网上的地址是一样的。公网地址就类似于：北京市海淀区清华大学，紫荆公寓 6 号楼。而楼里还有 125、225 宿舍，宿舍还有床位。这样很多主机接入了互联网，但共享一个 IP 地址

2. 通过隧道技术将 IPv6 和 IPv4 结合起来。

现在网络寻址的方式建立在 IPv4 之上，IPv6 还没有完全普及。那么在这个普及的过程中，显然会有长期的 IPv4 和 IPv6 共存的阶段。那可能会出现这样的一种情况：现阶段，使用 IPv6 的两个区域网络之间，并没有能支持 IPv6 的路由来连接，等 IPv6 逐渐普及，IPv4 网络萎缩，这时候使用 IPv4 的两个区域网络之间，并没有能支持 IPv4 的路由来连接。在这种情况下，科学家又创造性地提出了隧道技术，也就是说，在第一种情况下，用 IPv4 的格式把 IPv6 格式的数据包再包装一层，使用 IPv4 的协议寻址，让他能通过 IPv4 路由网络，从一个 IPv6 区域网络到另外一个 IPv6 区域网络，到达之后再把外面那层包装拆掉，在区域网络中通过 IPv6 寻址；在第二种情况下反过来，用 IPv6 的格式把 IPv4 格式的数据包再包装一层，使用 IPv6 的协议寻址，让他能通过 IPv6 路由网络，从一个 IPv4 区域网络到另外一个 IPv4 区域网络，到达之后再把外面那层包装拆掉，在区域网络中通过 IPv4 寻址。

3. 逐步更新网络路由节点，使其可以使用 IPv6 的方式寻址。

直到彻底替代 IPv4，这个需要多少年呢？需要很多很多年吧。等我们都老去了，离去了，这个过程也未必能完全结束。

### What is lwip

不能再讲计算机网络的原理了，让我们的文章回归正轨，我们要开始讲一些工程相关的知识。

互联网技术非常灵活，能够适应过去几十年不断变化的网络环境。互联网技术虽然最初是为阿帕网（ARPANET）等低速网络开发的，但现在可以在一个很大的链路技术频谱上运行，在带宽和误码率方面由截然不同的特性。由于现在已经开发了大量使用互联网技术的应用程序，能在未来的无线网络中使用现有的互联网技术是非常有利的。诸如传感器之类的小型设备通常需要体积小且价格便宜，因此不得不在有限的计算资源和内存上实现互联网协议。

lwIP 最早由 Adam Dunkels 编写，目前由 Kieran Mansley 带领的团队开发（开发者主页http://savannah.nongnu.org/projects/lwip ）。lwIP 是一个小型 TCP/IP 栈，可以在嵌入式系统中使用。lwIP 采取模块化设计，核心栈是 IP 协议的实现，用户可以在其上选择添加 TCP、UDP、DHCP 等其它协议，包括这些协议的各种特性。当然这样会导致代码量增加、复杂性提高，需要根据用户的需求进行调整。此外，lwIP 在有无操作系统、支持或不支持线程的情况下都可以运行，适用于 8 位或 32 位微处理器，支持小端和大端系统。

**总而言之，lwip 是一个库，支持了多种网络协议，包括 ARP、TCP、UDP**。

而操作 lwip 库，可以使用**RAW API**和**socket API**两种 API。

顾名思义，RAW API 的操作更原始，效率高，socket API 的封装层次更高，更好用。

## 2. Ethernet Toturial

首先设计硬件环境，打开 Vivado，新建一个 BlockDesign，把 ZYNQ 添加进工程。

在 ZYNQ 配置界面，配置以太网接口，我们使用 MIO 来驱动以太网。

![ZYNQ_CONFIG](http://leiblog.wang/static/image/2020/7/ZYNQ_config.png)

{% colorquote info %}

其实你可以下载我预先保存好的 tcl 配置文件，在 Preset 选项里选择导入，下载地址：http://leiblog.wang/static/FPGA/Easy_Simple_pynq_z2.tcl

{% endcolorquote %}

然后点击自动布线，连接上 Clock，如下图

![BlockDesign](http://leiblog.wang/static/image/2020/7/BlockDesign.png)

配置完成之后、综合、布线、生成 bit、Export Hardware、Launch SDK。

### Example 00 udp_helloworld

这里我们写以太网的工程有很多选择，一种是从头到尾自己配置 UDP 等，第二种是选择 SDK 内置的几个 Template，主要包括 TCP 回传，TCP/UDP 的 Server/Client 端。我们使用第一种，因为第二种，实在是不利于后期的封装，代码龙飞凤舞，因为 main 函数里塞了太多的东西了，并且我们一般只会用到 ipv4，感兴趣的网友可以自己去体验一下:)，验证一下以太网回传等。

于是，我们以默认的`Helloword`模版新建一个工程，第一个实验名是用 UDP 协议向 PC 发送"Hello World!"，所以取名叫 udp_helloworld。

但是默认的 bsp 里是没有加入 lwip 库的，我们在 system.mss 文件里选择 Modify this BSP's Settings，添加 lwip 的依赖。

![Add LWIP Lib](http://leiblog.wang/static/image/2020/7/add_lwip_lib.png)

然后，在把以下代码拖入工程的 src 里，https://github.com/LeiWang1999/EthernetVideo/tree/master/example/00_udp_helloworld

然后 File->Refresh. 确保没有 Error。

**如果碰到了找不到 lwip_init 的定义错误，尝试右击 udp_helloworld,在 ARM_GCC_LINKER 里加入 lwip4**

![Add Lwip 4](http://leiblog.wang/static/image/2020/7/add_lib.png)

如果没有错误，把 ZYNQ 的以太网口用网线和 PC 连接到一起，就可以烧录程序了。

在我提供的程序里，ZYNQ 的配置的网络信息如下：

IP:192.168.1.100, netmask:255.255.255.0, gateway:192.168.1.1

UDP 连接的 PC 的配置如下：

IP:192.168.1.200,netmask:255.255.255.0, gateway:192.168.1.1

所以，我们需要给我们的以太网设置一下静态 ip

![config ip](http://leiblog.wang/static/image/2020/7/config_ip.png)

然后，烧录程序，打开网络调试助手(网上随便下载的)，注意这里不要挂 vpn，不然可能会翻车（不要问我怎么知道的

![HelloWorld Result](http://leiblog.wang/static/image/2020/7/hello_world_result.png)

每隔一段时间，ZYNQ 给 PC 发送“Hello World”，**well done!**

#### 分析程序

其实都是套路，搬砖就完事了，但凡使用 lwIP 的程序，无论 TCP 还是 UDP，在进入 while(1)循环前，都会有这样一个配置流程：

- 设置开发板 MAC 地址
- 开启中断系统
- 设置本地 IP 地址
- 初始化 lwIP
- 添加网络接口
- 设置默认网络接口
- 启动网络
- 初始化 TCP 或 UDP 连接（自定义函数）

在 while(1)循环中，第一件事必然是使用 xemacif_input 函数将 MAC 队列中的包传输到 lwIP 栈中，这是 Xilinx 适配器提供的函数。再之后才是用户代码。我们继续看 UDP 相关文件中是如何进行连接初始化和“Hello World”字符输出的。

lwip 库中的函数很多都会返回执行状态码，也就是错误码，如果没有错误会返回 ERR_OK,在"lwip/err.h"中定义了各种错误码。

```c++
err = udp_send(tpcb, pbuf_to_be_sent);
	if (err != ERR_OK) {
		xil_printf("Error on udp send : %d\r\n", err);
		return;
	}
```

memset 和 memcpy 这两个函数来自 string.h。这是两个经典的 C 语言中的内存操作函数。memset 函数是将某一块内存中的内容全部设置为指定的值，通常为新申请的内存做初始化操作。其原型如下，将 s 中当前位置后面的 n 个字节用 ch 替换，memcpy 函数用于内存拷贝，其原型如下，在 src 所指的内存地址的起始位置开始，拷贝 n 个字节到目标 dest 所指内存地址的起始位置中。

剩下的配置请仔细阅读代码与 Xilinx 的文档！

https://github.com/LeiWang1999/EthernetVideo/tree/master/example/01_udp_sendto_helloworld 的代码里提供了 sendto 发送数据的方案，相较于使用 send 函数，不用每次发送的时候都要申请和释放内存。

## 3. Video Transfer Toturial

我们传输的图像从何而来？在 Vivado 里用 VDMA 搭建视频传输通道是非常常见的解决方案，关于如何使用 Vivado 搭建视频传输通道，可以在 Xilinx Forums 里找到非常详细的教程[4].

我用 ZYNQ 器件搭建的视频传输通道可以在这里下载到：

https://github.com/LeiWang1999/EthernetVideo/tree/master/video_pynq

![BD](http://leiblog.wang/static/image/2020/7/btjKvY.jpg)

摄像头使用的是 ov5640、并且编写了 hdmi 输出视频的通路，用的也是 digilent-library 里提供的 rgb2dvi 的 IP。

在 vivado 里使用 tcl 命令行，cd 到项目目录，然后

```tcl
source ./ZYNQ_VIDEO.tcl
```

就可以创建工程了。

由于我使用的软件环境是是 vivado2020 和 vitis，所以这里就不把 sdk 的 tcl 放出来了，只放上对应的程序文件。

https://github.com/LeiWang1999/EthernetVideo/tree/master/software

效果如下：

![Performence](http://leiblog.wang/static/image/2020/7/2SX255.jpg)

因为 UDP 一次最多只能发送 65536 个字节的数据，而一副 640 \* 720 的 RGB 图像需要占用 900K 字节的数据，所以我使用 UDP 一次发送一行的数据，然后再上位机那边拼起来，但总的来说还有一些小问题，嘛～毕竟做这个是为了兴趣，懒得解决了，如果有有心人可以帮忙解决一下、美化一下界面 hh ～

### Reference

1. https://www.zhihu.com/question/358209123
2. https://zhuanlan.zhihu.com/p/24860273
3. https://blog.csdn.net/FPGADesigner
4. https://forums.xilinx.com/t5/Video-and-Audio/Xilinx-Video-Series/td-p/849583