---
title: ModelSim加Matlab图像仿真
categories:
  - Technical
tags:
  - FPGA
  - ModelSim
  - CV
date: 2020-02-04 22:15:21	
---

# 仿真系统搭建

### 1 系统框图

![img](https://img-blog.csdnimg.cn/20200209163836796.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

<!-- more -->

### 2 VGA 时序发生器

由于图像处理电路都是基于 VGA 时序，所以在仿真平台中，需要有一个源 VGA 信号发生器，模仿 VGA 的时序。其模块基本属性如下所示:

`define pix_800_600

_module_ vga_ctl

(

input pix_clk,

input reset_n,

input [23:0] VGA_RGB,

output VGA_CLK,

output [11:0] hcount,

output [11:0] vcount,

output [7:0] VGA_R,

output [7:0] VGA_G,

output [7:0] VGA_B,

output VGA_HS,

output VGA_VS,

output VGA_DE,

output BLK

);

`include "vga_parameter.vh"

对应不同分辨率，我们需要不同的 VGA 标准参数，这些参数定义在 vga_parameter.vh 文件中，例如本仿真平台处理的图像数据为 800\*600:则该文件中应有如下参数的定义:

`ifdef pix_800_600

​ //---------------------------------//

​ // 800\*600 60HZ pixel clock 40.00MHZ

​ //--------------------------------//

​ parameter H_Total = 1056;

​ parameter H_Sync = 128;

​ parameter H_Back = 88;

​ parameter H_Active = 800;

​ parameter H_Front = 40;

​ parameter H_Start = 216;//H_Sync+H_Back

​ parameter H_End = 1016;//H_Sync+H_Back+H_Active

​ //-------------------------------//

​ // 800\*600 60HZ

​ //-------------------------------//

​ parameter V_Total = 628;

​ parameter V_Sync = 4;

​ parameter V_Back = 23;

​ parameter V_Active = 600;

​ parameter V_Front = 1;

​ parameter V_Start = 27;//V_Sync+V_Back

​ parameter V_End = 627;//V_Sync+V_Back+V_Active

`endif

## **3** **图像数据转换**

![img](https://img-blog.csdnimg.cn/20200209163911447.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

我们需要将待处理的图像，转换成为 VGA 能够解析的数据流，这部分利用 Matlab 实现数据格式的转换。在工作目录执行 img_txt.m 脚本，脚本执行结束之后，有如上图所示效果。其中 img 为拍摄的原图，R、G、B 为三个通道的灰度图。并在工作目录下生成如下图所示的文件:

![img](https://img-blog.csdnimg.cn/20200209163939541.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

为了读取这些文件，我们需要编写 verilog 软件程序，把它们转换成 RGB 格式输出到 VGA 的 RGB 输出端。其模块基本属性如下:

`timescale 1ns/1ps

`define NULL 0

_module_ imread_frame1(

input pixel_clk,

input reset_n,

input de,

input [11:0] frame_cnt,

output [23:0] rgb

);

经过上述几个步骤，我们已经能够把一幅图像以 VGA 时序进行输出，至此已经搭建出了图像仿真系统的雏形，并且为了将经过 verilog 处理的图像以可视化的形式展现，还需要有 verilog 写入图像数据的软件程序，其模块基本属性如下:

`timescale 1ns/1ps

`define NULL 0

_module_ imwrite_frame4(

input pixel_clk,

input reset_n,

input de,

input [11:0] frame_cnt,

input [23:0] rgb

);

该程序会在图像处理的过程中，将数据缓存到工作目录，并且例化多个写图像的模块，即可观察到图像处理的过程，其缓存结果如下:

![img](https://img-blog.csdnimg.cn/2020020916395324.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)![img](https://img-blog.csdnimg.cn/20200209163952989.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

最后，我们需要将 verilog 保存的文件利用 matlab 脚本 txt_img.m 转换成可视化的图片，观察 verilog 处理过程是否正确。最后显示处理结果如下

![img](https://img-blog.csdnimg.cn/20200209164000278.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

# 图像处理仿真

## 1 RGB 转 YCbCr

​ 为了方便进行图像阈值的提取，利用 verilog 实现了 RGB 到 YCbCr 的色域转换，计算式可以百度:

其效果如下图所示

![img](https://img-blog.csdnimg.cn/20200209164057522.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

rgb2ycbcr U_rgb2ycbcr(

.pixelclk(pixel_clk),

.i_rgb(VGA_RGB),

.i_hsync(VGA_HS),

.i_vsync(VGA_VS),

.i_de(VGA_DE),

.o_rgb(y_rgb),

.o_ycbcr(y_ycbcr),

.o_gray(y_gray),

.o_hsync(y_o_hsync),

.o_vsync(y_o_vsync),

.o_de(y_o_de));

### 2 阈值确定与二值化

​ 首先将车牌从复杂的环境中提取出来。确定车牌的阈值，二值化。

![img](https://img-blog.csdnimg.cn/20200209164121605.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

ycbcr_location#( //Paramter Define such as threshold

​ )U_ycbcr_location(

​ .pixelclk(pixel_clk),

​ .reset_n(reset_n),

​ .i_rgb(y_rgb),

​ .i_ycbcr(y_ycbcr),

​ .i_gray(y_gray),

​ .i_hsync(y_o_hsync),

​ .i_vsync(y_o_vsync),

​ .i_de(y_o_de),

​ .binary_image(yl_binary),

​ .rgb_image(yl_rgb),

​ .gray_image(yl_gray),

​ .o_hsync(yl_o_hsync),

​ .o_vsync(yl_o_vsync),

​ .o_de(yl_o_de)

​ );

## 3 锁定车牌位置

​ 通过水平垂直投影，可以将车牌框选出来，并且为了便于识别数字，我们将车牌以外的颜色转换成近似车牌的底色，便于进行第二次的垂直投影与阈值提取。此外，框选车牌的同时，削掉了一部分上下边界，去除了定孔。

![img](https://img-blog.csdnimg.cn/20200209164143241.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

Vertical_Projection#( //Paramter Define such as threshold

)U_Vertical_Projection(

.pixelclk(pixel_clk),

.reset_n(reset_n),

.i_binary(HV_dout),

.i_hs(HV_o_hsync),

.i_vs(HV_o_vsync),

.i_de(HV_o_de),

.i_hcount(hcount),

.i_vcount(vcount),

.hcount_l(hcount_l),

.hcount_r(hcount_r),

.vcount_l(vcount_l),

.vcount_r(vcount_r));

可在 modelsim 中观察仿真波形图如下:

![img](https://img-blog.csdnimg.cn/20200209164156882.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

可以看到，车牌的水平边界和垂直边界都被取出，分别对应为 hcount_l, hcount_r, vcount_l, vcount_r 这几个信号。

## 4 提取车牌数字

这次，我们重复进行一次色域的转换，在 YCbCr 色域下提取出车牌内的字符，效果如下图所示，至此我们已经提取出了复杂场景下的车牌字符。

![img](https://img-blog.csdnimg.cn/20200209164221634.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

## 5 水平垂直投影

对已经经过二值化的车牌字符图像，我们例化一个能够检测八个边界的水平垂直投影模块。其垂直投影后的效果如下图所示

![img](https://img-blog.csdnimg.cn/20200209164243717.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

至此，我们可以将八个字符一个一个的分割，进行单独的数字识别处理，极大的简化了处理的难度。接下来介绍本工程中采用的两种数字识别的方案。

## 6 方案一:数字特征识别

## ![img](https://img-blog.csdnimg.cn/20200209164323594.png)

本方案已经上板实现，但是仍然存在缺陷，比如说仅仅能识别数字，且识别条件较为苛刻，需要现场调试一段时间。

## 7 方案二:5X8 维矩阵检测

对单个数字进行画框操作，并且统计出每个小格子的像素点数目，超过该格子的一半则认为是 1，少于则认为是 0。下图为 5 的画框仿真。

![img](https://img-blog.csdnimg.cn/20200209164351984.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

在 modelsim 中观察仿真波形，则 5 的数字模型被正确识别，并且对于同一字体其数字特征值应相同。![img](https://img-blog.csdnimg.cn/20200209164404463.png)

## Video

{% dplayer "url=http://leiblog.wang/static/2020-03-25/car.mp4"  "theme=#FADFA3" "autoplay=false" %}

相关工程文件可以去 Material 标签下载。
