---
categories:
  - Technical
tags:
  - Bilibili
  - MacOS
date: 2020-03-19 16:38:05	
---

今天剪辑视频的时候，在 Bilibili 看到了想要的素材。想着白嫖是多么的舒服，于是研究了几个小时碰瓷 Bilibili 视频的方法。
网上是有挺多的，什么在 bilibili 前面加 i，加 jj，一些乱七八糟的，奈何我是尊贵的 Mac 用户，以上方法统统免疫。
最后，我研究出了自己的一套方案。

<!-- more -->

### 安装 you-get

首先你得有 python，Mac 应该是自带的。我的是 Python3.7
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200228230846653.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
然后，用 pip 命令安装 you-get

```bash
pip install you-get
```

you-get 是我一年前写爬虫的时候接触到的工具，简单的使用方法：

```bash
you-get 视频链接
```

就可以下载视频了。
但是 Bilibili 的视频，都是 FLV 格式的，我们还需要转码的软件。

### 安装 VideoConverter

这是一款 Mac 下的转码软件，特殊的是，他支持 FLV 转 MP4。
安装包可以去我的个人主页：
[点击前往](http://www.leiblog.wang/material)
下载
