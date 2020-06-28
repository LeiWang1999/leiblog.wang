---
categories:
  - Technical
tags:
  - FPGA
  - VsCode
  - Vivado
---

## 引言

大二的时候，接触硬件描述语言。Vivado 自带的编辑器实在很难用，为了有更加舒适的体验，我寻找着取而代之的方法。
网络上的答案大同小异，基本上都是用的 Sublime Text.（我的学长们也都是）
平时写工程的过程中，我习惯使用 VsCode,尤其是在微软收购 GitHub 之后，虽然越来越卡了。但是我还是尽可能的选择了 Code,现在用了一年多了，感觉还是很不错的，接下来分享一下如何配置使用，让您的生活更美好。

## 站在前人的肩膀上

是找了一些年代相对比较久远的文章，关于 VsCode 替换掉 Vivado 原本的编辑器的。我稍作尝试，其实也很简单。

#### 步骤一、更换 Vivado 自带文本编辑器

##### 第一步 打开 Vivado 再 Tool 菜单中 打开 Settings

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190730172400496.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

##### 第二步 在 Settings 里更换默认的文本编辑器

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190730172425179.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
这里需要键入的表达式是： C:/Program Files/Microsoft VS Code/Code.exe [file name] -[line number]
前面是 VsCode 应用程序的绝对路径。Linux 下如果是在环境变量中，可以直接写 Code 但是 Windows 下好像不可以。
这样双击工程下面的文件，Vivado 会自动使用 Code 打开文件。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190730172648606.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
这样，我们就能用 VsCode 取代原本的编辑器了。

#### 步骤二、用 VsCode 舒适的编写 Verilog

##### 第一步 安装 Verilog 扩展

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190730172831152.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
在 VsCode 扩展商店搜索 verilog。
我们安装使用人数最多的扩展。可以看见其是支持 Verilog 和 SystemVerilog 的，如果你使用的是 VHDL 则下载另外的插件即可。
他能帮你实现的功能：

- 语法高亮
- 自动补齐
- 列出端口。
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190730173249308.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
  可以看到，我们的 verilog 文本，被渲染的很漂亮。并且能够实现自动补齐。

##### 第二步 实现自动纠错

使用一个 IDE（文本编辑器），我们最关心的问题往往是，他能否实现**自动语法纠错**？
当然是可以的，实现这一功能的前提是：
vivado 安装目录下的 xvlog（这个是 vivado 自带的语法纠错工具）。
**你需要将这个工具所在的目录放置在系统的环境变量**，以便 VsCode 能够方便的调用他。
具体的目录就是 Vivado 的 bin 文件夹。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190730173554915.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
如果你不知道上述界面如何调出，请移步:www.google.cn
添加完成之后，在命令行输入 xvlog -- version 检测是否生效
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190730173907830.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
如果没有打印出未找到该命令，那么你可能需要重启您的电脑。

接下来我们在设置里，找到刚才安装的 verilog 扩展，将 verilog 的 Linter 更换成 xvlog。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019073017403368.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
同理，如果你想使用的语法纠错插件来自 modelsim,quatus，选择他们对应的 linter 即可。
就我个人的使用经验，各个软件的语法排错机制还是有一点细微的不同的，建议选择正确的解析器。
设置完成之后，就能实现语法的纠错，在平常的工程中已经可以很给力的帮助你了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190730174127342.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
需要注意的是，编译器需要您手动保存，才会开启 xvlog 解析，也就是说观看最新错误之前，需要保存一下。

#### 步骤三、自动生成 Testbench

有时候在工程中要例化一个模块，这个模块有几十个输入几十个输出，如果没有一个好的脚本帮助你，不仅人为出错的可能比较大，例化的过程想必也是痛苦的。
还好有人已经在 VsCode 编写过自动生成 Testbench 的脚本了，感谢。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190730174338671.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
扩展商店搜索 Verilog_TestBench,安装过后，任意编写一段 verilog 程序。按下 ctrl+shift+p,选择 testbench 即可生成 testbench 对应的 tb 文本。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190730174455337.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
效果如上图所示。执行脚本之后，其出结果导向 powershell 的输出结果之中。帮我们自动生成了了时钟信号，复位信号，以及进行了模块的例化。如果你只需要例化模块，复制这一部分进你的代码中就可以了。到这里，VsCode 已经能够给你的工程带来及其舒适的体验了。

#### 步骤四、进一步优化

可以看到，美中不足的是，生成的文本你还需复制粘贴到新建的 testbench 文件中去，真是有些麻烦了。
但是从命令行执行的命令可以看到，这个脚本是用 python 编写的。顺着文件目录找到原本的 python 文件，即可修改输出内容。
这里我为了能让输出的 testbench 自动生成 tb 文件，上了一段 powershell 的脚本。

理清一下我们脚本的思路：脚本需要将命令执行，输入的第一个参数为文件名 a.v,输出的文件名为 tb_a.v.
可以将整个脚本的初始化条件写入 powershell 的 profile 文件中（就和 bash 里的.bashrc 一样，ps 在启动时会自动加载此配置文件的内容）。

那么 profile 文件在哪儿呢？打开你的 powershell。输入 echo $profile 即可。
想编辑文件，直接在命令行输入 code $profile 。
前提是你的 vscode 添加进系统环境变量了，关于怎么添加环境变量，请看上文。

最后写的脚本如下，只需更改 TestBenchPath 的值就行了，你完全可以写的比我好，不如自己试一下？

```powershell
function createtb_function{
    param(
        [Parameter(ValueFromPipeline=$true)]
        $InputObject
    )
    $FileName = $InputObject
    $tbFileName = "tb_" + $FileName.split("\")[-1]
    echo $tbFileName
    python $env:TestBenchPath $FileName >> $tbFileName
}

set-alias ll Get-ChildItemColor

$env:TestBenchPath="C:\Users\22306\.vscode\extensions\truecrab.verilog-testbench-instance-0.0.5\out\vTbgenerator.py"

set-alias createtb createtb_function
```

修改过后，重启 vscode 的 powershell 命令行。输入命令 createtb xxx.v，即可输出生成文件。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019073017491592.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
最后 testbench 文件就自动生成了。

#### 步骤五 VsCode 使用小技能

这部分用来总结一下 VsCode 使用过程中的一些小经验，可能会持续更新

- 按住鼠标中键，可以连续操作多行，这个在例化模块，以及一些无脑操作的时候很管用。
- ctrl + f 搜索 替换
- Code 支持文件对比功能，在左边的资源栏中右击比较即可（抄代码很方便）
- ctrl+r 可以搜索过去曾经使用 code 打开的文件，真的方便，不需要鼠标操作，够极客

## 如果你的 vivado 在 ubuntu 系统下

在学院的电脑里装了一个 ubuntu18.04 系统，发现 vivado 也有 ubuntu 的支持，那么为了能够有更好的编程体验，我又对上述过程进行了 linux 的移植.

首先，安装 vscode

其次,把更换 vivado 中文本编辑器的命令换成 code [filename] , 这样你的 vivado 文本编辑器就换成 vscode 了。

然后,在您的系统里安装 powershell。

再然后在设置里搜索 terminal，把终端在 linux 上使用的路径换成 pwsh 所在路径。

最后修改 powershell 的 profile 文件，不过与 windows 的略有不同，这里贴上代码。

```powershell
#以后要 使用 ll 而不是 ls了。

function createtb_function{
    param(
        [Parameter(ValueFromPipeline=$true)]
        $InputObject
    )
    $FileName = $InputObject
    $tbFileName = "tb_" + $FileName.split("/")[-1]
    echo $tbFileName
    python $env:TestBenchPath $FileName >> $tbFileName
}

set-alias ll Get-ChildItemColor


$env:TestBenchPath="/home/princeling/.vscode/extensions/truecrab.verilog-testbench-instance-0.0.5/out/vTbgenerator.py"

set-alias createtb createtb_function
```

    其他就能和原来一样喽～

## 答疑总结

无法正确进行语法纠错的原因：

1. 打开的 Verilog 文件目录有中文路径。
2. 打开的 Verilog 文件目录有空格。

## 写在最后的

    至此，我深深体会到了作为一名verilog编程者的辛酸。
    2020年寒假，在家给自己写了一个博客。欢迎访问:
    [**点击进入**](http://www.leiblog.wang)
    学习FPGA的朋友们，欢迎关注我的项目：https://github.com/LeiWang1999/FPGA
