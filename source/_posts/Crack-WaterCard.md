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

可以进行 NFC 读写的模块，我用的是`PN 532`，价格相对便宜，推荐购买赠送了UART-USB转接线的店铺，比如下图中的第一家。

![PN 532](http://leiblog.wang/static/image/2020/7/6a8Lum.png)

对应的软件可以去我的 Github 下载全家桶: [iCARDCrack](https://github.com/LeiWang1999/iCARDCrack)，如果你不会使用 Github，那我强烈建议你在空闲时间去学习，但是对于本文，你只需要进入该 Github 页面，选择`Download Zip`并且解压到本地就好。

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

[iCARDCrack](https://github.com/LeiWang1999/iCARDCrack)下，有[PL-2303驱动](https://github.com/LeiWang1999/iCARDCrack/tree/master/PL-2303驱动)这一子目录，安装对应的驱动就可以。

如果你没有像我希望的那样，看到该目录下的txt文件，那我好心的提醒你，关于 Windows8到Windows10，安装驱动可能需要参考到下面的百度经验。

[WIN10下如何解决PL2303驱动不可用](https://jingyan.baidu.com/article/c85b7a646f1db5003bac95be.html)

如果安装完了驱动，并且UART的接线方式也没有出现错误，那么你就可以打开[PN532工具XP版](https://github.com/LeiWang1999/iCARDCrack/tree/master/PN532工具XP版)，把USB头插到电脑上。

在软件界面，点击发现设备：

![发现设备](http://leiblog.wang/static/image/2020/7/发现设备.png)

准备好想Crack的IC卡，贴近PN532,如下图

![Water Card](http://leiblog.wang/static/image/2020/7/hg2Doq.png)

#### 读取和保存数据

点击读整卡按钮，开始暴力破解水卡的Key，整个过程看运气，我最慢最慢没有超过一个小时。

![破解完成](http://leiblog.wang/static/image/2020/7/破解完成.png)

首先，备份你的原始数据，万一你错误修改了你的卡的数据，这张卡也就报废了。点击下图中的三角形，可以保存当前卡片的信息到dump文件中。

![保存dump](http://leiblog.wang/static/image/2020/7/保存dump文件.png)

#### 修改余额

下跳到第六扇区的第0块号，N4-N5这两列分别对应的是水卡的余额，比如N4:24:N5:40, 十六进制2440转换到10进制为9280，则水卡的余额为92.80元，所以理论上水卡的最大余额为FFFF->655.35元。

但N6是随着水卡余额变化的，这是八位的加密位，但我和朋友经过一下午的研究还是捣鼓出加密算法。

$$
N6 = N0 \oplus N1 \oplus N2 \oplus N3 \oplus N4 \oplus N5
$$
如果你想充值的余额为92.80,转换为16进制为2440,用上述公式算出，加密为65，则只要把N4N5N6改为244065即可。

更改完成之后，再次保存dump文件，然后点击写整卡，就可以完成写卡了！

### Tips for this blog

1. 希望你千万不要做出卖水卡这种傻事，虽然经过我的观察在我们学校确实会有这种人。人际关系的研究里有一条定律叫做「六度分离理论 Six Degrees of Separation」，这条理论最初是1929年由匈牙利作家Frigyes Karinthy所提出的，你平均每认识六个人就会有两个人联系在一起！
2. 我们应该使用这种技术嘛？说实话，我平时打开水都没有自己充过钱了，一是因为冲开水余额的时候阿姨态度不好，二是我真的蛮穷的😂。但是谁让他的加密算法搞得这么简单呢，如果我们都顺着他的意思，不去使用这项能力，那这家公司的加密技术就不会进步，我们不能让他恰烂钱。

3. 在Github上还有一个文件夹，是DumpViewer，该工具旨在把dump文件转换为txt文件，方便我使用diff等工具观察不同水卡，或者水卡充值前后之间有哪些数据发生了变化，从而找出加密的算法。

4. 最后需要注意的是，每张水卡都有不同的ID，请不要直接把别的水卡的dump文件写入到其他水卡上，那样这张卡就报废了！

