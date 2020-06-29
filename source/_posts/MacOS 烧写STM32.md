---
categories:
  - Technical
tags:
  - Bilibili
  - STM32
  - MacOS
date: 2020-03-19 16:41:24	
---

2019 年在实验室的主机上安装了黑苹果，迫于体验极佳，在 19 年的最后一个月入了 MacBookPro 替代原来的游戏本提高生产力，这一篇文章也因此而来。

不得不说，Mac 剪辑视频、写 Web 都是极佳的利器，但我作为一名电子专业的学生，平时需要使用的一些 EDA 软件。他们有很多仅支持 Windows，或者仅在 Windows 上能够破解，这成为了令很多电子类专业用 MacBook 的同学一个脑阔疼的问题，最简单直接的办法是安装双系统，大部分人都是如此。但我在买笔记本之前就已经想好退路了，要我在 Mac 上装双系统是不可能的，接下来要介绍的是 **How to Develop Stm32 with MacOS？**

<!-- more -->

在 MacOS 上实现 STM32 烧写的方案网络上有多种：

1. ST 官方推出的 STM32CubeIDE

   基于 Eclipse 改过来的 IDE，集成了 STM32CubeMX、STM32CubeProgrammer 和烧写程序的功能。

   But，用 Vivado 的 SDK 时带来的不好的体验，使我对 Eclipse 魔改过来的 IDE 一直没有好感，这个方案 PASS

2. VsCode 插件 PlatformIO IDE

   也是可以通过 Jtag 等方式直接烧写 stm32，但用的库文件不够官方，结构有点诡异。

   适合写 Arduino 或小规模的工程。而且新建工程的速度慢，这个方案 PASS。

3. stm32Cube 加上 openocd

   我的好朋友用的是这种方案，我没有尝试，似乎效果还不错？感兴趣的朋友可以百度研究一下。

而我们使用的工具链是：VSCode、ArmGCC、STM32CubeMX、STM32CubeProgrammer。

类似第三种方案与第一种方案的结合。

> Keil 这些年来一直停滞不前，而这些工具一直在进步。

**注：本文下载的软件较多，且下载速度较慢，可以去我的个人博客下载镜像**[立即前往](http://leiblog.wang)

### 一、安装 VsCode

[前往官网下载](https://code.visualstudio.com)

安装完成之后，在左侧的扩展栏目里，先搜索插件：Chinese，安装，汉化。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020022619425146.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
然后按住**command+shift+p**，选择“Install ‘code’ command in PATH。接下来我们就能够使用 code file 这个命令来打开文件了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200226194310241.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
安装 C/C++插件
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227200402488.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
安装 Cortex-Debug 插件
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227184502960.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

### 二、安装 ArmGCC

ArmGCC 是 Arm 官方用于编译 ARM 架构的裸机系统的编译器，就和他的名字一下，用于 Arm 的 GCC 编译器。我们需要使用 ArmGCC 将我们编写的 C 语言程序、Arm 的汇编指令，编译成烧写用的 hex、bin 文件、调试用的 elf 文件。

[点击前往下载](https://launchpad.net/gcc-arm-embedded/+download)

下载 MacOS 版本，解压后得到如下的文件夹。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020022619474650.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

之后，我将这个文件夹拷贝到了 home 下的 opt 目录中，毕竟不能把他随便丢在下载文件夹里面，太难看了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200226194834843.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
然后，我们需要把 gcc-arm 的 bin 文件夹放到环境变量 PATH 里去。

> 为什么需要放到环境变量 PATH 中去？怎么放到环境变了里去？
>
> 我相信有实力折腾本文操作的读者，应该不会被这种问题困扰，但还是为了以防万一，这里对我的环境做简单的介绍。
>
> 如果不清楚，可以参考相关文章。

打开命令行，输入 code ~/.zshrc 由于我用的是 zsh，所以是 zshrc，如果没有经过配置，那么你输入的应该是：code ~/.bash_profile

在文件的末尾加入：

```zsh
export PATH="/Users/wanglei/opt/gcc-arm-none-eabi/bin:$PATH"
```

把中间的 “/Users/wanglei/opt/gcc-arm-none-eabi/bin”替换成你的目录，然后重启命令行，就可以使用 ArmGCC 的命令了。

打开命令行，输入以下命令查看是否配置成功：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227184422582.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

### 三、安装 STM32CubeMX 和 STM32CubeProgrammer

STM32CubeMX 是意法半导体官方，在前几年推出的图形化底层代码生成工具，简单的说，它能够芯片/开发版，进行一些底层代码的配置，例如要用哪些资源，控制这些资源的库代码，这样你就可以不用依赖正点给你写的那些功能库了。

而对于 STM32CubeProgrammer，我们需要了解将程序下载到单片机的两种方法：

- 普通串口下载

  只要开发板上有 CH340 这个芯片，我们就能通过 USB 直接往芯片里烧写 Hex 文件。但 Mac 上没找到烧写的软件，所以这个方案对我们不适用。

- 下载器下载

  用 SWD 等方式烧写 Bin 文件，STM32CubeProgrammer 是帮助我们实现该功能的工具。

**但是，这个软件居然要依靠 JAVA 来运行，我们还得先去官网下载一个 JDK 安装一下**

[点击前往下载 JDK](https://www.oracle.com/java/technologies/javase-downloads.html)

**推荐安装 JDK8，我先前安装的是 JDK13，烧写正常，但是 Programmer 的图形界面打不开**

然后分别下载 Mac 版本的 STM32CubeMX 和 STM32CubeProgrammer

[点击前往下载 STM32CubeMX](https://www.st.com/content/st_com/en/products/development-tools/software-development-tools/stm32-software-development-tools/stm32-configurators-and-code-generators/stm32cubemx.html)

[点击前往下载 STM32CubeProgrammer](https://www.st.com/content/st_com/en/products/development-tools/software-development-tools/stm32-software-development-tools/stm32-programmers/stm32cubeprog.html#get-software)

MAC 安装这俩软件的方式有些不同，下载完成，解压之后。需要右击，显示包内容。然后前往 Content->MacOS 里执行安装（如果权限不够，需要给一下执行权限）安装过程还是很无脑的，一路 Next 就好了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200226194854654.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
安装完成 STM32CubeMX 之后，打开软件。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200226194916291.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

安装成功。

接着一样安装完 STM32CubeProgrammer，为了能够方便的烧写单片机，我们还需要将 STM32_Programmer_CLI 这个文件所在目录放在系统的环境变量里。

```zsh
export PATH="/Applications/STMicroelectronics/STM32Cube/STM32CubeProgrammer/STM32CubeProgrammer.app/Contents/MacOs/bin:$PATH"
```

然后我们就能用

```zshz
STM32_Programmer_CLI -c port=SWD -d build/ProgramDemo.bin 0x8000000 -s
```

来烧写单片机了。

### 四、实战演示

看到这里，我相信你还有很多疑惑 🤔，似乎环境已经安装完成了，但如何使用？

流程大概是这样：

1. 使用 STM32CubeMX 生成工程模板
2. 配置 VsCode 工程环境
3. 编写代码
4. make 编译出 hex、bin、elf 文件
5. 用 STM32_Programmer_CLI 命令烧写/或者用 Cortex-Debugger 插件调试

用语言描述有些复杂，录制了一段视频放在 bilibili。
[点击前往](https://www.bilibili.com/video/av92270190/)

<iframe src="//player.bilibili.com/player.html?aid=92270190&bvid=BV1C7411N7PV&cid=157541123&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>

### 五、一些问题的解答

1. 报错信息：Error: Old ST-LINK firmware version. Upgrade ST-LINK firmware

   下载 ST-Link 升级固件：[点击下载](https://www.st.com/zh/development-tools/stsw-link007.html)

   也可以去我的个人博客下载哦：[点击前往](http://www.leiblog.wang/home)

   给自己的 ST-Link 升级一下.

2. SVD 文件在哪儿下载
   百度搜索 stm32xxxxx svd
   在这里插入图片描述
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227192258737.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
   进入官网的，全英文的这个网站，然后点击
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/2020022719254336.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
   下载下图的压缩包即可。
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200227192645235.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
