---
title: Nmap Notes
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2020-08-17 19:22:41
---

本文是使用Nmap的一些小笔记，省的用的时候再去网络上查找。

<!-- more -->

## 基本使用

刚开始使用的时候可能会因为信息量太大无从下手，最简单的使用就是`nmap your-ip（域名）` 就可以扫描出其对外开放的服务。

```bash
root@kali:~# nmap 192.168.31.13
Starting Nmap 7.70 ( https://nmap.org ) at 2018-08-12 23:02 CST
Nmap scan report for 192.168.31.13
Host is up (0.00038s latency).
Not shown: 998 closed ports
PORT      STATE SERVICE
8080/tcp  open  http-proxy
10010/tcp open  rxapi
MAC Address: 00:0C:29:99:D3:E6 (VMware)
Nmap done: 1 IP address (1 host up) scanned in 1.85 seconds
```

可以看出只开放了8080端口和10010端口

`nmap -p 端口 IP(域名)`，判断ip是否开放指定端口

```bash
root@kali:~# nmap -p 8080 192.168.31.13
Starting Nmap 7.70 ( https://nmap.org ) at 2018-08-12 23:05 CST
Nmap scan report for 192.168.31.13
Host is up (0.00045s latency).

PORT     STATE SERVICE
8080/tcp open  http-proxy
MAC Address: 00:0C:29:99:D3:E6 (VMware)

Nmap done: 1 IP address (1 host up) scanned in 0.36 seconds
root@kali:~# nmap -p 80 192.168.31.13
Starting Nmap 7.70 ( https://nmap.org ) at 2018-08-12 23:05 CST
Nmap scan report for 192.168.31.13
Host is up (0.00049s latency).

PORT   STATE  SERVICE
80/tcp closed http
MAC Address: 00:0C:29:99:D3:E6 (VMware)

Nmap done: 1 IP address (1 host up) scanned in 0.42 seconds
```

可以看出8080端口开放，80端口没有开放

也可以增加端口和网段 ：

```zsh
nmap  -p 22,21,80 192.168.31.13
nmap  -p 22,21,80 192.168.31.1-253
```

nmap 192.168.31.1/24 扫描整个子网的端口 ，这个过程可能会比较久

## 进阶

在继续讲之前，先介绍一下Nmap可以识别出的6种端口状态

**开放**：工作于开放端口的服务器端的应用程序可以受理TCP	连接、接收UDP数据包或者响 应SCTP（流控制传输协议）请求。

**关闭**：虽然我们确实可以访问有关的端口，但是没有应用程序工作于该端口上。

**过滤**：Nmap	不能确定该端口是否开放。包过滤设备屏蔽了我们向目标发送的探测包。

**未过滤**：虽然可以访问到指定端口，但Nmap不能确定该端口是否处于开放状态。 

**打开｜过滤**：Nmap认为指定端口处于开放状态或过滤状态，但是不能确定处于两者之中的 哪种状态。在遇到没有响应的开放端口时，Nmap会作出这种判断。这可以是由于防火墙丢 弃数据包造成的。

**关闭｜过滤**：Nmap	认为指定端口处于关闭状态或过滤状态，但是不能确定处于两者之中的 哪种状态。

### 常用选项

 1. 服务版本识别（-sV），Nmap可以在进行端口扫描的时候检测服务端软件的版本信息。版本信息将使后续的漏 洞识别工作更有针对性。

```zsh
root@kali:~# nmap -sV 192.168.31.13 -p 8080
Starting Nmap 7.70 ( https://nmap.org ) at 2018-08-13 00:02 CST
Nmap scan report for 192.168.31.13
Host is up (0.00076s latency).

PORT     STATE SERVICE VERSION
8080/tcp open  http    Apache Tomcat 8.5.14
MAC Address: 00:0C:29:99:D3:E6 (VMware)

Service detection performed. Please report any incorrect results at https://nmap.org/submit/ .
Nmap done: 1 IP address (1 host up) scanned in 12.75 seconds
```

 2. 操作系统检测（-O），Nmap还能识别目标主机的操作系统。

```zsh
root@kali:~# nmap -O 192.168.31.13 
Starting Nmap 7.70 ( https://nmap.org ) at 2018-08-13 00:03 CST
Nmap scan report for 192.168.31.13
Host is up (0.00072s latency).
Not shown: 998 closed ports
PORT      STATE SERVICE
8080/tcp  open  http-proxy
10010/tcp open  rxapi
MAC Address: 00:0C:29:99:D3:E6 (VMware)
Device type: general purpose
Running: Linux 3.X|4.X
OS CPE: cpe:/o:linux:linux_kernel:3 cpe:/o:linux:linux_kernel:4
OS details: Linux 3.2 - 4.9
Network Distance: 1 hop

OS detection performed. Please report any incorrect results at https://nmap.org/submit/ .
Nmap done: 1 IP address (1 host up) scanned in 3.49 seconds
```

 3. 禁用主机检测（-Pn），如果主机屏蔽了ping请求，Nmap可能会认为该主机没有开机。这将使得Nmap无法进行进一 步检测，比如端口扫描、服务版本识别和操作系统识别等探测工作。为了克服这一问题，就 需要禁用Nmap的主机检测功能。在指定这个选项之后，Nmap会认为目标主机已经开机并会 进行全套的检测工作

 4. 强力检测选项（-A），启用-A选项之后，Nmap将检测目标主机的下述信息
服务版本识别（-sV）；
操作系统识别（-O）；
脚本扫描（-sC）；
Traceroute（–traceroute）。

### TCP扫描选项

 1. TCP连接扫描（-sT）：指定这个选项后，程序将和目标主机的每个端口都进行完整的三次 握手。如果成功建立连接，则判定该端口是开放端口。由于在检测每个端口时都需要进行三 次握手，所以这种扫描方式比较慢，而且扫描行为很可能被目标主机记录下来。如果启动 Nmap的用户的权限不足，那么默认情况下Nmap程序将以这种模式进行扫描。

 2. SYN扫描（-sS）：该选项也称为半开连接或者SYN stealth。采用该选项后，Nmap将使用 含有SYN标志位的数据包进行端口探测。如果目标主机回复了SYN/ACK包，则说明该端口处 于开放状态：如果回复的是RST/ACK包，则说明这个端口处于关闭状态；如果没有任何响应 或者发送了ICMP unreachable信息，则可认为这个端口被屏蔽了。SYN模式的扫描速度非常 好。而且由于这种模式不会进行三次握手，所以是一种十分隐蔽的扫描方式。如果启动Nmap 的用户有高级别权限，那么在默认情况下Nmap程序将以这种模式进行扫描。

 3. TCP NULL（-sN）、FIN（-sF）及XMAS（-sX）扫描：NULL 扫描不设置任何控制位； FIN扫描仅设置FIN标志位：XMAS扫描设置FIN、PSH和URG的标识位。如果目标主机返回 了含有RST标识位的响应数据，则说明该端口处于关闭状态；如果目标主机没有任何回应， 则该端口处于打开｜过滤状态。

 4. TCP Maimon扫描（-sM）：Uriel Maimon 首先发现了TCP Maimom扫描方式。这种模式的 探测数据包含有FIN/ACK标识。对于BSD衍生出来的各种操作系统来说，如果被测端口处于 开放状态，主机将会丢弃这种探测数据包；如果被测端口处于关闭状态，那么主机将会回复 RST。

 5. TCPACK扫描（-sA）：这种扫描模式可以检测目标系统是否采用了数据包状态监测技术 （stateful）防火墙，并能确定哪些端口被防火墙屏蔽。这种类型的数据包只有一个ACK标识 位。如果目标主机的回复中含有RST标识，则说明目标主机没有被过滤。

 6. TCP窗口扫描（-sW）：这种扫描方式检测目标返回的RST数据包的TCP窗口字段。如果目 标端口处于开放状态，这个字段的值将是正值；否则它的值应当是0。

 7. TCP Idle扫描（-sI）：采用这种技术后，您将通过指定的僵尸主机发送扫描数据包。本机 并不与目标主机直接通信。如果对方网络里有IDS，IDS将认为发起扫描的主机是僵尸主机。

### UDP扫描选项

Nmap有多种TCP扫描方式，而UDP扫描仅有一种扫描方式（-sU）。虽然UDP扫描结果没有 TCP扫描结果的可靠度高，但渗透测试人员不能因此而轻视UDP扫描，毕竟UDP端口代表着 可能会有价值的服务端程序。但是UDP扫描的最大问题是性能问题。由干Linux内核限制1秒内最多发送一次ICMP Port Unreachable信息。按照这个速度，对一台主机的65536个UDP端口进行完整扫描，总耗时必 定会超过18个小时。

优化方法主要是：
1. 进行并发的UDP扫描；
2. 优先扫描常用端口；
3. 在防火墙后面扫描；
4. 启用--host-timeout选项以跳过响应过慢的主机。

假如我们需要找到目标主机开放了哪些 UDP端口。为提高扫描速度，我们仅扫描 53端口 （DNS）和161端口（SNMP）。

可以使用命令`nmap -sU 192.168.56.103 -p 53,161`

### 目标端口选项

默认情况下，Nmap将从每个协议的常用端口中随机选择1000个端口进行扫描。其nmapservices文件对端口的命中率进行了排名。

可以自定义端口参数：

-p端口范围：只扫描指定的端口。扫描1〜1024号端口，可设定该选项为–p	1-1024。扫描1 〜65535端口时，可使用-p-选项。

-F（快速扫描）：将仅扫描100	个常用端口。

-r（顺序扫描）：指定这个选项后，程序将从按照从小到大的顺序扫描端口。 ●	-

-top-ports <1 or=""	greater="">：扫描nmap-services	里排名前N的端口。

### 输出选项

Nmap可以把扫描结果保存为外部文件。在需要使用其他工具处理Nmap的扫描结果时，这一 功能十分有用。即使您设定程序把扫描结果保存为文件，Nmap还是会在屏幕上显示扫描结果。

Nmap支持以下几种输出形式。

- 正常输出（-oN）：不显示runtime信息和警告信息。

- XML	文件（-oX）：生成的	XML	格式文件可以转换成	HTML	格式文件，还可被Nmap	的图 形用户界面解析，也便于导入数据库。本文建议您尽量将扫描结果输出为XML文件。

- 生成便于Grep使用的文件（-oG）：虽然这种文件格式已经过时，但仍然很受欢迎。这种格 式的文件，其内容由注释（由#开始）和信息行组成。信息行包含6个字段，每个字段的字段 名称和字段值以冒号分割，字段之间使用制表符隔开。这些字段的名称分别为Host、Ports、Protocols、Ignored State、OS、Seq Index、IP ID	Seq	和Status。这种格式的文件便于 grep或awk之类的UNIX指令整理扫描结果。

- 输出至所有格式(-oA)
为使用方便，利用-oA选项 可将扫描结果以标准格式、XML格式和Grep格式一次性输出。分别存放在.nmap，.xml和.gnmap文件中。


### 时间排程控制选项

Nmap可通过-T选项指定时间排程控制的模式。它有6种扫描模式。

- paranoid（0）：每5分钟发送一次数据包，且不会以并行方式同时发送多组数据。这种模式 的扫描不会被IDS检测到。

- sneaky（1）：每隔15秒发送一个数据包，且不会以并行方式同时发送多组数据。

- polite（2）：每0.4	秒发送一个数据包，且不会以并行方式同时发送多组数据。

- normal（3）：此模式同时向多个目标发送多个数据包，为	Nmap	默认的模式，该模式能自 动在扫描时间和网络负载之间进行平衡。

- aggressive（4）：在这种模式下，Nmap	对每个既定的主机只扫描5	分钟，然后扫描下一 台主机。它等待响应的时间不超过1.25秒。

- insane（5）：在这种模式下，Nmap	对每个既定的主机仅扫描75	秒，然后扫描下一台主 机。它等待响应的时间不超过0.3秒。

默认的扫描模式通常都没有问题。除非您想要进行更隐匿或更快速的扫 描，否则没有必要调整这一选项。

### 扫描IPv6主机

启用Nmap的-6选项即可扫描IPv6的目标主机。当前，只能逐个指定目标主机的IPv6地址。

```zsh
nmap	-6	fe80::a00:27ff:fe43:1518
```

同一台主机在IPv6网络里开放的端口比它在IPv4网络里开放的端口数量要 少。这是因为部分服务程序尚未支持IPv6网络。

### 脚本引擎功能（Nmap Scripting Engine，NSE）

最后但是同样重要的，Nmap本身已经很强大了，但是加上它的脚本引擎更加开挂了，NSE 可使用户的各种网络检査工作更为自动化，有助于识别应 用程序中新发现的漏洞、检测程序版本等Nmap原本不具有的功能。虽然Nmap软件包具有各 种功能的脚本，但是为了满足用户的特定需求，它还支持用户撰写自定义脚本。

- auth：此类脚本使用暴力破解等技术找出目标系统上的认证信息。

- default：启用--sC	或者-A	选项时运行此类脚本。这类脚本同时具有下述特点：执行速度快；输出的信息有指导下一步操作的价值；输出信息内容丰富、形式简洁；必须可靠；不会侵入目标系统；能泄露信息给第三方。

- discovery：该类脚本用于探索网络。

- dos：该类脚本可能使目标系统拒绝服务，请谨慎使用。

- exploit：该类脚本利用目标系统的安全漏洞。

- external：该类脚本可能泄露信息给第三方。

- fuzzer：该类脚本用于对目标系统进行模糊测试。

- instrusive：该类脚本可能导致目标系统崩溃，或耗尽目标系统的所有资源。

- malware：该类脚本检査目标系统上是否存在恶意软件或后门。

- safe：该类脚本不会导致目标服务崩溃、拒绝服务且不利用漏洞。

- version：配合版本检测选项（-sV），这类脚本对目标系统的服务程序进行深入的版本检测。

- vuln：该类脚本可检测检査目标系统上的安全漏洞。
在Kali	Linux系统中，Nmap脚本位于目录/usr/share/nmap/scripts。

- -sC	或--script=default：启动默认类NSE脚本。

- --script	<filename>|<category>|<directories>：根据指定的文件名、类别名、目录名，执行 相应的脚本。

- --script-args	<args>：这个选项用于给脚本指定参数。例如，在使用认证类脚本时，可通过 这个选项指定用户名和密码
```zsh
nmap --script http-enum,http-headers,http-methods,http-php-version	-p	80 192.168.56.103
```

### 规避检测的选项

在渗透测试的工作中，目标主机通常处于防火墙或 IDS 系统的保护之中。在这种环境中使用 Nmap 的默认选项进行扫描，不仅会被发现，而且往往一无所获。此时，我们就要使用Nmap 规避检测的有关选项。

```zsh
-f（使用小数据包）：这个选项可避免对方识别出我们探测的数据包。指定这个选项之后， Nmap将使用8字节甚至更小数据体的数据包。

--mtu：这个选项用来调整数据包的包大小。MTU（Maximum	Transmission	Unit，最大传输 单元）必须是8的整数倍，否则Nmap将报错。

-D（诱饵）：这个选项应指定假	IP，即诱饵的	IP。启用这个选项之后，Nmap	在发送侦测 数据包的时候会掺杂一些源地址是假IP（诱饵）的数据包。这种功能意在以藏木于林的方法 掩盖本机的真实	IP。也就是说，对方的log还会记录下本机的真实IP。您可使用RND生成随机 的假IP地址，或者用RND：number的参数生成<number>个假IP地址。您所指定的诱饵主机 应当在线，否则很容易击溃目标主机。另外，使用了过多的诱饵可能造成网络拥堵。尤其是 在扫描客户的网络的时候，您应当极力避免上述情况。
Kali	Linux	渗透测试的艺术（中文版）
151第 6章	服务枚举
--source-port	<portnumber>或-g（模拟源端口）：如果防火墙只允许某些源端口的入站流 量，这个选项就非常有用。

--data-length：这个选项用于改变Nmap	发送数据包的默认数据长度，以避免被识别出来是 Nmap的扫描数据。

--max-parallelism：这个选项可限制Nmap	并发扫描的最大连接数。

--scan-delay	<time>：这个选项用于控制发送探测数据的时间间隔，以避免达到IDS/IPS端 口扫描规则的阈值。
```

