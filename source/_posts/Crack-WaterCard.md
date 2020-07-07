---
title: IC Card Crack 傻瓜教程
categories:
  - Technical
tags:
  - Njtech
  - Crack
date: 2020-07-07 17:29:41
---

![Banner](http://leiblog.wang/static/image/2020/7/u9ZKM2.jpg)

如何破解不联网的 IC 卡呢？本文以我就读的学校开水卡为例，只需要有一台装有`Windows XP`以上版本的电脑，再加上一些硬件和软件，你就可以使用该解决方案。

{% colorquote danger %}
本教程仅供学习，请勿将本教程用于非法用途，因为 Crack 产生非法利益继而被学校处分的事情互联网上比比皆是。
{% endcolorquote %}

<!-- more -->

### 一、准备硬件和软件

1. 可以进行 NFC 读写的模块，我用的是`PN 532`，价格相对便宜，推荐购买赠送了UART-USB转接线的店铺，比如下图中的第一家。

![PN 532](http://leiblog.wang/static/image/2020/7/6a8Lum.png)

2. 对应的软件可以去我的 Github 下载全家桶: [iCARDCrack](https://github.com/LeiWang1999/iCARDCrack)，如果你不会使用 Github，那我强烈建议你在空闲时间去学习，但是对于本文，你只需要进入该 Github 页面，选择`Download Zip`并且解压到本地就好。

### 二、顺利读写 IC 卡

硬件设备和软件都已经准备好了，接下来只要把你电脑和NFC读写模块一起扔到锅里煮，小火微炖，让各种调料随着时间慢慢入味，这样既保证了肉质的鲜美，又避免了残忍和血腥，两全其美。。

并不是，希望你不要真的这么做了。

#### PN532焊接

其实写文章的时候我才意识到有这一步，刚买回来的PN532是一块裸板。

好在需要焊接的地方很好，只要焊一排引脚就好了，那如果你不会焊接，随便拉一个电类专业的同学给他练手好了。

#### 正确连接UART-USB转接线

我们使用PN-532默认的UART接口，当然可以看到他提供了SPI/IIC等接口，但为了和PC通讯还是用默认的UART比较好。也许这些因为你不熟悉基本的几种通讯协议听不懂，但没关系，骚扰你的店家，把线接对了就行！

{% colorquote warning %}

其实也就四根线，但新人往往容易犯的错误是，主机的RX需要和从机的TX接到一起，主机的TX需要和从机的RX接到一起，这样才能正常通讯！

{% endcolorquote %}

#### 打上PN532的驱动

[iCARDCrack](https://github.com/LeiWang1999/iCARDCrack)下，有[PL-2303驱动](https://github.com/LeiWang1999/iCARDCrack/tree/master/PL-2303驱动)这一子目录

