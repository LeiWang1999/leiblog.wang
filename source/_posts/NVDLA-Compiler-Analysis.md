---
title: NVDLA Compiler Analysis
categories:
  - Technical
tags:
  - NVDLA
date: 2021-06-24 16:42:15
---

这篇文章记录一下笔者剖析 NVDLA Compiler 工作机制的一些经过，在 NVDLA 的软件仓库中，Compiler与Runtime的源代码被放置在umd目录下，由同一个 Makefile 组织。

对于sw部分文档十分缺乏，在 NVDLA的页面里只有零星的介绍，而关于软件设计的细节、以及如何拓展设计都没有介绍。并且，Compiler这部分的代码充满了智慧，例如：

![image-20210624203707954](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210624203707954.png)

还有一些突然停止维护了而没开发完的feature。并且由于其前端只有 Caffe 的 Parser，导致其端到端的推理仅可以支持一些比较弱的分类网络，但阅读代码理解其设计与工作的思路，学习其抽象语法树的设计还是一个比较有意义的工作。

<!-- more -->

### 1. Clion 结合 Docker 调试 Makefile Project

由于 Compiler 需要一些 Linux 的头文件，所以在 Mac 和 Windows 上不方便直接编译（笔者是 MacOS），为了不污染环境，我推荐使用 docker 新建一个容器，当然为了方便验证，我还是推荐直接使用之前文章里提到过的 image ：

```bash
docker pull farhanaslam/nvdla 
# 做一个端口映射，毕竟22端口还是很容易冲突的
docker run -it -v /Users/wanglei/:/home -p 2222:22 --name "nvdlakit" wanglei/nvdla-kit
```

然后为其添加sshd服务，具体可以参考这篇[博客](https://www.cnblogs.com/mengw/p/11413461.html)：

```bash
apt-get upgrade
apt-get install openssh-client
apt-get install openssh-server
/etc/init.d/ssh start
```

开启 root 的访问权限：

```bash
vim /etc/ssh/sshd_config
```

找到 `PermitRootLogin`将其设置为 yes 。

再用`passwd `命令设置一下Root的密码就好了。

大概一年之前，Clion 并不支持 Makefile 项目，在这之后其在插件里提供了部分对 Makefile 的支持，但是不能够进行远程调试，而在笔者尝试使用 Clion 远程调试 Makefile 项目的时候，JB的官方刚好发布了这个问题的解决方案，时机非常巧妙，官方的post title叫做 [Full remote mode](https://www.jetbrains.com/help/clion/remote-projects-support.html)，根据配置就可以远程调试了，但是对于 Compiler，其在编译的过程中依赖环境变量`TOP`、其在编译之后的执行阶段又依赖环境变量`LD_LIBRARY_PATH`，所以我们需要对配置做一点小改动。

在`Preference|Build,Execution,Deployment｜Makefile`页面，为 options 加上 TOP 的路径。

![image-20210624181927692](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210624181927692.png)

在`Configurations`页面，加上 Environment variables 。

![image-20210624182055447](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210624182055447.png)

再来，apps 的 Makefile 里没有编译出带调试信息的可执行文件，为`MODULE_COMPILEFLAGS`加上`-g`选项。这样，就可以单步调试了。

### 2. Compiler

首先，umd目录下面一共有若干个文件夹：

- apps：compiler和runtime的外层逻辑，一般这里放的都是解析命令行参数之类的程序。
- core：runtime和compiler的主要实现逻辑放在这里，是需要着重阅读的部分
- externel：这个目录存放了umd用到的第三方库，需要说明的是protobuf主要用于caffe模型的解析，因为caffe的blob格式原始存储方式是使用了google开发的protobuf、用来压缩生成 Loadable 文件的 FlatBuffers等等。
- make：若干通用的makefile规则。
- port：主要是runtime的底层访问API，为了实现OS无关，做了这个隔离层，目前只实现了Linux下的。这层API包括内存访问，用户态和内核态的交互IOCTL，内存分配等。需要注意的是NVDLA的runtime部分用户态和内核态交互使用了Linux用于显卡抽线的DRM接口
- utils：这个文件夹放了几个通用的模块，包括BitBinaryTree以及伙伴内存管理等。其中伙伴内存管理模块在compiler的tensor内存分配推导部分有用到

#### 2.1 NVDLA Compiler 工作流程

在阅读代码的时候，可以打开官方已经写了的部分调试信息输出开关，在源代码里可以找到类似这样的声明：

```cpp
inline bool debugGraphDump() const { return false; }
inline bool debugClone() const { return false; }
inline bool debugOps() const { return false; }
inline bool debugGroupOps() const { return false; }
inline bool debugMathOptz() const { return false; }
inline bool debugWeights() const { return false; }
inline bool debugQuantization() const { return false; }
inline bool debugFuseSubEngineOps() const { return false; }
inline bool debugSurfaces() const { return false; }
inline bool debugBuffers() const { return false; }
inline bool debugCopyOutDebug() const { return false; }
inline bool debugMemoryLayout() const { return false; }
inline bool debugBinding() const { return false; }
inline bool debugDepGraph() const { return false; }
inline bool debugMemHazards() const { return false; }
inline bool debugRelocs() const { return false; }
```

将`false`调成`true`，就可以输出一些信息，例如 debugGraphs 开关可以把几个关键的抽象语法树保存为 Json 格式的文件。

这里列出一下把所有的 inline bool debugxxx 开关都打开的log，有三千多行：

<details>
  <summary>
  Console Output of Lenet5.
  </summary> 

  creating new wisdom context...
  opening wisdom context...
  parsing caffe network...
  mark prob
  Marking total 1 outputs
  parsing calibration table...
  attaching parsed network to the wisdom...
  compiling profile "fast-math"... config "opendla_small"...
  Invalid target config
  Compiler profile: fast-math
  compute precision NVDLA_PRECISION_INT8
  network input data format NHWC
  network input surface format NVDLA_IMG_A8B8G8R8
  network output data format NCxHWx
  network output surface format NVDLA_FEATURE_DATA_INT8
  batch size 0
  tensor scaling mode PER_TENSOR
  quantization mode PER_KERNEL
  raw weights of bias-0: -0.189939, -0.0122971, -0.127561, -0.0370565, 0.0295314, 0.00792045, -0.212149, -0.159978, -0.100598, -0.0833697, -0.234531, -0.257588, 0.211461, -0.181789, -0.289102, -0.106559, -0.24326, 0.238864, -0.0477089, 0.183563, 
  n-0:->n-0:dc-conv-0 n-1:bias-0 
  n-1:->n-2:pdp-0 
  raw weights of bias-1: 0.0266923, -0.0210181, -0.0988805, 0.0149911, -0.0189492, -0.0394126, -0.00789565, -0.0799653, -0.0767248, -0.0982602, -0.0398029, -0.0778087, -0.041269, 0.0100064, -0.0130392, 0.0253316, 0.0122957, -0.0409093, -0.00850784, 0.00141418, -0.0388875, -0.0170651, -0.0327771, -0.0181817, -0.086323, -0.0200653, -0.0210894, -0.0629572, -0.0672779, 0.0461048, -0.0657149, -0.0425482, 0.0145769, -0.0452918, -0.0231088, -0.106806, -0.0240538, -0.0203086, -0.0370518, -0.0241791, -0.0967711, 0.0139108, -0.0661912, -0.0684465, -0.0150072, 0.0310905, -0.0493356, -0.0828543, 0.007227, -0.0534207, 
  n-2:->n-3:dc-conv-1 n-4:bias-1 
  n-3:->n-5:pdp-1 
  raw weights of bias-2: -0.0160908, -0.00596529, 0.00571031, 0.0141845, 0.00457521, 0.000918757, 0.00182481, 0.00487062, -0.010953, 0.00405698, 0.00241628, 0.00215307, -0.00274916, 0.000693179, 0.00434156, -0.00366797, -0.000154131, 0.0017697, -0.00575291, 0.00485401, 0.0049358, 0.00053109, 0.0035042, 0.00115322, -0.00513705, 0.00220661, 0.0130855, -0.0040593, 0.00742773, -0.00207535, 0.00708569, 0.0130998, 0.00320843, 0.00807043, -0.00725761, 0.00397733, 0.000158773, -0.00426002, -0.00715113, 0.00737358, 0.000183469, 0.0121654, -0.00185334, -0.00478108, -0.00362065, 0.00572813, -0.000719333, 0.0062139, -0.0042254, 0.004558, 0.00368481, -0.00638158, -0.000542821, 0.0056826, -0.0025529, -0.000308287, 0.00540015, 0.0128624, -0.00784788, -0.0110209, 0.00165233, -0.00979184, 0.00627872, 0.00449704, 0.00551083, 0.0103216, -8.54422e-05, 0.0100825, -0.00381044, 0.00709125, -0.00162287, 0.0100035, 0.00153186, -0.00192266, 0.0177549, 0.0028109, -0.000116088, 0.00857696, 0.00336946, -0.00577133, 0.00734607, 0.00367282, 0.00290999, -0.00764302, -0.00435251, 0.0112651, -0.00474309, -0.00293007, -0.00337636, 0.00350733, 0.00413807, 0.00648092, -0.00520403, -0.00946726, -0.00299138, -0.0129122, 0.00296569, -0.00357285, 0.00962356, 0.00297598, -0.00135108, 0.00128038, -0.00179181, 0.010566, 0.0180151, -0.00460528, -0.000536081, -0.00138855, 0.00448689, 0.00400495, 0.00287102, 0.00166039, -0.00473274, 0.000981164, -0.00258576, -0.00459088, -0.00192138, 0.00659034, -0.00184609, 0.000764893, -0.00330961, -0.00467795, 0.00309991, 0.00156627, 0.00300166, -0.00358254, -0.00415914, 0.0100725, -0.0018276, 0.00587635, 0.00282938, 0.00114757, 0.00344898, -0.00504183, 0.0038491, -0.000436917, 0.0107106, 0.0101657, 0.000640607, 0.000200644, -0.00187112, -0.000905704, -0.00430365, -0.0114306, -0.00254204, 0.00999499, -0.00361041, -0.000814386, -0.00166425, 0.0036898, 0.00910051, -0.008081, -0.00621499, -0.00476039, 0.00352937, -0.00883984, 0.01267, 0.00553043, 0.00413786, 0.00378476, -0.000376468, -0.00935109, 0.00111507, 0.00449815, 0.00232352, -0.00526044, -0.00657905, 0.00202523, 0.00783183, 0.00331134, 0.0113301, 0.000284586, 0.00555074, -0.00561669, -0.00304549, 0.00232793, 0.00480359, 0.00417361, 0.00917487, -0.0106033, -0.00935452, 0.000472582, -0.00565795, 0.00265887, 0.00553717, 0.00830267, 0.00499382, -0.00830142, 0.00125424, 0.017598, 0.0110582, -0.00110636, 0.00206649, 0.00600223, 0.00580203, 0.0095547, -0.00347894, -0.000947946, -0.00558407, 0.00457049, -0.00695887, 0.00425796, 0.00692133, 0.0110334, 0.00080584, 0.00967977, -0.00152148, 0.000685625, 0.0111667, -0.0122356, -0.00163696, 0.00874573, -0.00747964, 9.53092e-05, 0.00125609, -0.00484556, 0.00964425, -0.0049331, -0.00674903, 0.00155267, 0.00779544, -0.00475451, -0.00446209, 0.00994591, -0.00069266, 0.0028227, -0.00315651, -0.00449883, 0.0124552, 0.00346543, -0.00341063, -0.00535677, -0.00169873, -0.000445362, -0.0113099, -0.000209029, -0.000552506, 0.00152527, 0.00539479, 0.01207, 0.00102972, 0.00209548, -0.000864239, 0.0142878, -0.00662105, -0.000244511, 0.0163849, 0.0115843, -0.000624234, 0.00149396, -0.00786256, 0.00504049, -0.00208822, -0.0053671, 0.00282316, 0.0123619, 0.00466023, -0.00132708, -0.00287518, 0.00934964, 0.0016545, -0.00141152, -0.00762542, 0.00133011, 0.00354479, -0.000554709, 0.00377187, -0.00410209, -0.0126422, 0.00330497, 0.00150753, 0.00671193, -0.000671467, -0.0020892, -0.00214926, 0.0089044, -0.00174577, -0.00241748, 0.00212694, 0.00360922, 0.000103615, -0.00459254, 0.0130479, 0.00234114, 0.00618787, 0.00378565, 0.00384775, -0.00576199, 0.00337918, -0.00326975, -0.00731954, -0.00448419, -0.00779377, -0.00449993, -0.00854937, 0.00736093, 0.00287959, 0.0059524, 0.000887351, -0.00268741, 0.00199675, -0.00819081, -0.00205734, 0.00159371, -0.00959275, -0.00528138, 0.00541374, -0.00262693, -0.00522113, 0.00781175, 0.0028028, 0.00183135, 0.00101005, 0.00178905, 0.00425291, 0.00325326, -8.29429e-05, -0.00223922, -0.0016051, -0.000555084, -0.00882739, 0.00275148, -0.0073338, -0.0130495, -0.00503639, -0.00335259, -0.00762874, -0.000773354, -0.0105282, -0.00604246, -0.0148792, 0.00509254, -0.00343132, 0.000884271, -0.00659211, -0.00150819, -0.00066668, -0.0123335, 0.00527709, 0.0030958, -0.000243152, 0.00163575, 0.00305789, -0.0032451, 0.00757511, -0.00441922, -0.00156038, -0.00333063, -0.00142264, -0.0023367, -0.000629858, -0.00742183, -0.00525475, 0.00106295, 0.00557893, -0.00735296, 0.0104845, 0.00501339, -8.00025e-05, -0.00386855, 0.0118741, 0.00171545, -0.00241817, -0.00286085, 0.0102796, 0.00680886, 0.00343526, -0.00503297, 0.00212876, 0.00683794, -0.003925, 0.000772909, 0.00182556, -0.0014057, 0.00537661, -0.00120791, 0.00606066, -0.00678337, -0.00536369, 0.00152798, 0.00113391, -0.00381443, 0.00313435, 0.00663192, -0.00171548, -0.00413377, -0.00378595, 0.00658897, 0.0115695, -0.00290412, 0.00122328, -0.00573779, -0.00666367, 0.00148788, -9.36032e-05, -0.00523711, 0.00100092, -0.00102205, -0.00262461, -0.00398723, -0.0089795, -0.00639421, -0.00406703, -0.00919813, -0.00157012, -0.00926236, 0.000345377, 0.0134298, 0.00564769, 0.0109816, 0.00467368, -0.00677339, -0.00471123, 0.00475499, -0.00101872, 0.00660738, 0.00300622, -0.00118807, 0.00205993, -0.00326071, 0.00502191, -0.010096, -0.0014445, 0.00273264, 0.00262441, -0.000504633, 0.0147181, -0.00540654, -0.00326369, -0.0115329, 0.002148, 0.00194639, -0.0101629, -0.00186599, 0.0116932, -0.000642768, -0.00464428, -0.00737856, 0.0024578, 0.00681272, 0.00416895, 0.00396947, -0.0062771, 0.000580598, -0.00101734, -0.00331908, -0.00147223, 0.0143072, 0.00945186, 0.00212467, 0.00702214, 0.00546126, 0.0012048, 0.00901155, -0.00802692, 0.0110263, -0.00302939, -0.000107664, 0.00674109, -0.00715094, -0.00416368, -0.00109759, -0.00181006, 0.00966231, -0.00492121, 0.00562308, 0.00988432, 0.00530393, -0.000590022, 0.00142536, -0.00840012, -0.00277301, -0.00480831, 0.00468205, 0.00824822, 0.0036084, 0.00174475, 0.00406262, 0.00370747, -0.00549466, 0.00743413, 0.0100927, 0.0071719, -0.00537823, 0.00443552, 0.00363082, 0.00574112, -0.0126982, -0.00116731, -0.00286545, 0.0194753, 0.0103246, 0.00858375, 0.000202983, -0.00622996, -0.00117932, -0.0201595, 0.0038894, 0.0100873, 0.00150142, 
  n-4:->n-6:fc-0 n-7:bias-2 
  n-5:->n-8:sdp-scale-0 n-9:act-0 
  raw weights of bias-3: -0.0285562, 0.102577, -0.0604402, -0.0525417, -0.00433865, 0.028257, -0.0173933, 0.0503867, 0.0181165, -0.0360676, 
  n-6:->n-10:fc-1 n-11:bias-3 
  n-7:->n-12:cpu-sm-0 
  Prototxt #chnls (C = 1) != Profile #chnls for input (NVDLA_IMG_A8B8G8R8: C = 4). Preferring #chnls from Profile for compiling.
  ::Edge setBindId edge=e-0 domain=0 id=0
  EngineAST graph level input edge[0] is e-0
  ::Edge edge=e-0 bindid=0
  input bind id: 0
  ::Edge setBindId edge=e-8 domain=1 id=0
  EngineAST graph level output edge[0] is e-8
  ::Edge edge=e-8 bindid=0
  output bind id: 0
  dc-conv-0/n-0/conv1:
  in e-0
  out e-11
  aux e-9
  bias-0/n-1/conv1:
  in e-11
  out e-1
  aux e-10
  pdp-0/n-2/pool1:
  in e-1
  out e-2
  dc-conv-1/n-3/conv2:
  in e-2
  out e-14
  aux e-12
  bias-1/n-4/conv2:
  in e-14
  out e-3
  aux e-13
  pdp-1/n-5/pool2:
  in e-3
  out e-4
  fc-0/n-6/ip1:
  in e-4
  out e-17
  aux e-15
  bias-2/n-7/ip1:
  in e-17
  out e-5
  aux e-16
  sdp-scale-0/n-8/(No canonical node):
  in e-5
  out e-19
  aux e-18
  act-0/n-9/relu1:
  in e-19
  out e-6
  fc-1/n-10/ip2:
  in e-6
  out e-22
  aux e-20
  bias-3/n-11/ip2:
  in e-22
  out e-7
  aux e-21
  cpu-sm-0/n-12/prob:
  in e-7
  out e-8
  ::Edge edge=e-9 bindable=0
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-10 bindable=0
  ::Edge edge=e-12 bindable=0
  ::Edge edge=e-13 bindable=0
  ::Edge edge=e-15 bindable=0
  ::Edge edge=e-16 bindable=0
  ::Edge edge=e-18 bindable=0
  ::Edge edge=e-20 bindable=0
  ::Edge edge=e-21 bindable=0
  ::Edge edge=e-11 bindable=0
  ::Edge edge=e-1 bindable=0
  ::Edge edge=e-2 bindable=0
  ::Edge edge=e-14 bindable=0
  ::Edge edge=e-3 bindable=0
  ::Edge edge=e-4 bindable=0
  ::Edge edge=e-17 bindable=0
  ::Edge edge=e-5 bindable=0
  ::Edge edge=e-19 bindable=0
  ::Edge edge=e-6 bindable=0
  ::Edge edge=e-22 bindable=0
  ::Edge edge=e-7 bindable=0
  ::Edge edge=e-8 bindable=1
  printGraph: engine_ast::generateGraph
  n-0:dc-conv-0/conv1:    (in)e-0[tt-1],  (Aux)e-9[tt-4],         (out)e-11[tt-8], 
  n-1:bias-0/conv1:       (Aux)e-10[tt-5],        (in)e-11[tt-8],         (out)e-1[tt-3], 
  n-2:pdp-0/pool1:        (in)e-1[tt-3],  (out)e-2[tt-3], 
  n-3:dc-conv-1/conv2:    (in)e-2[tt-3],  (Aux)e-12[tt-4],        (out)e-14[tt-8], 
  n-4:bias-1/conv2:       (Aux)e-13[tt-5],        (in)e-14[tt-8],         (out)e-3[tt-3], 
  n-5:pdp-1/pool2:        (in)e-3[tt-3],  (out)e-4[tt-3], 
  n-6:fc-0/ip1:   (in)e-4[tt-3],  (Aux)e-15[tt-4],        (out)e-17[tt-8], 
  n-7:bias-2/ip1: (Aux)e-16[tt-5],        (in)e-17[tt-8],         (out)e-5[tt-3], 
  n-8:sdp-scale-0/:       (in)e-5[tt-3],  (Aux)e-18[tt-7],        (out)e-19[tt-3], 
  n-9:act-0/relu1:        (in)e-19[tt-3],         (out)e-6[tt-3], 
  n-10:fc-1/ip2:  (in)e-6[tt-3],  (Aux)e-20[tt-4],        (out)e-22[tt-8], 
  n-11:bias-3/ip2:        (Aux)e-21[tt-5],        (in)e-22[tt-8],         (out)e-7[tt-3], 
  n-12:cpu-sm-0/prob:     (in)e-7[tt-3],  (out)e-8[tt-2], 
  ::Edge edge=e-9 bindable=0
  edge: e-9 tsd: tsd-0 registered
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-0 domain=0 bind_id=0
  ::Surface bindId(tsd-1, 0) -> 0
  set bind id 0 for e-0 tsd-1
  edge: e-0 tsd: tsd-1 registered
  ::Edge edge=e-10 bindable=0
  edge: e-10 tsd: tsd-2 registered
  ::Edge edge=e-12 bindable=0
  edge: e-12 tsd: tsd-3 registered
  ::Edge edge=e-13 bindable=0
  edge: e-13 tsd: tsd-4 registered
  ::Edge edge=e-15 bindable=0
  edge: e-15 tsd: tsd-5 registered
  ::Edge edge=e-16 bindable=0
  edge: e-16 tsd: tsd-6 registered
  ::Edge edge=e-18 bindable=0
  edge: e-18 tsd: tsd-7 registered
  ::Edge edge=e-20 bindable=0
  edge: e-20 tsd: tsd-8 registered
  ::Edge edge=e-21 bindable=0
  edge: e-21 tsd: tsd-9 registered
  ::Edge edge=e-11 bindable=0
  edge: e-11 tsd: tsd-10 registered
  ::Edge edge=e-1 bindable=0
  edge: e-1 tsd: tsd-11 registered
  ::Edge edge=e-2 bindable=0
  edge: e-2 tsd: tsd-12 registered
  ::Edge edge=e-14 bindable=0
  edge: e-14 tsd: tsd-13 registered
  ::Edge edge=e-3 bindable=0
  edge: e-3 tsd: tsd-14 registered
  ::Edge edge=e-4 bindable=0
  edge: e-4 tsd: tsd-15 registered
  ::Edge edge=e-17 bindable=0
  edge: e-17 tsd: tsd-16 registered
  ::Edge edge=e-5 bindable=0
  edge: e-5 tsd: tsd-17 registered
  ::Edge edge=e-19 bindable=0
  edge: e-19 tsd: tsd-18 registered
  ::Edge edge=e-6 bindable=0
  edge: e-6 tsd: tsd-19 registered
  ::Edge edge=e-22 bindable=0
  edge: e-22 tsd: tsd-20 registered
  ::Edge edge=e-7 bindable=0
  edge: e-7 tsd: tsd-21 registered
  ::Edge edge=e-8 bindable=1
  ::Edge edge=e-8 domain=1 bind_id=0
  ::Surface bindId(tsd-22, 1) -> 0
  set bind id 0 for e-8 tsd-22
  edge: e-8 tsd: tsd-22 registered
  e-9 edge setting new surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge setting new surface format NVDLA_IMG_A8B8G8R8
  e-10 edge setting new surface format NVDLA_BIAS_DATA_INT16
  e-12 edge setting new surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge setting new surface format NVDLA_BIAS_DATA_INT16
  e-15 edge setting new surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge setting new surface format NVDLA_BIAS_DATA_INT16
  e-18 edge setting new surface format NVDLA_SCALE_DATA_INT16
  e-20 edge setting new surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge setting new surface format NVDLA_BIAS_DATA_INT16
  e-11 edge setting new surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge setting new surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge setting new surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge setting new surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge setting new surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge setting new surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge setting new surface format NVDLA_FEATURE_DATA_INT8
  e-5 edge setting new surface format NVDLA_FEATURE_DATA_INT8
  e-19 edge setting new surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge setting new surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge setting new surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge setting new surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge setting new surface format NVDLA_FEATURE_DATA_INT8
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-7 for tsd-7 for e-18 with NVDLA_SCALE_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-17 for tsd-17 for e-5 with NVDLA_FEATURE_DATA_INT8
  tb-18 for tsd-18 for e-19 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-18 edge already has set surface format NVDLA_SCALE_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-5 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-19 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-7 for tsd-7 for e-18 with NVDLA_SCALE_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-17 for tsd-17 for e-5 with NVDLA_FEATURE_DATA_INT8
  tb-18 for tsd-18 for e-19 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0

  Try Merging: bias-2 & sdp-scale-0
  Merging: Sucess
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-19 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-18 for tsd-18 for e-19 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-19 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-18 for tsd-18 for e-19 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0

  Try Merging: dc-conv-0 & bias-0
  Merging: Not Feasible

  Try Merging: dc-conv-1 & bias-1
  Merging: Not Feasible

  Try Merging: fc-0 & bias-2
  Merging: Not Feasible

  Try Merging: bias-2 & act-0
  Merging: Sucess
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0

  Try Merging: fc-0 & bias-2
  Merging: Not Feasible

  Try Merging: fc-1 & bias-3
  Merging: Not Feasible
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  rawBias/Si*Sw -0.189939 /  ( 0.00784518 * 0.00345303 ) = -7011.49 -> -7011 -> -7011*2^-0
  rawBias/Si*Sw -0.0122971 /  ( 0.00784518 * 0.00345303 ) = -453.941 -> -454 -> -454*2^-0
  rawBias/Si*Sw -0.127561 /  ( 0.00784518 * 0.00345303 ) = -4708.85 -> -4709 -> -4709*2^-0
  rawBias/Si*Sw -0.0370565 /  ( 0.00784518 * 0.00345303 ) = -1367.92 -> -1368 -> -1368*2^-0
  rawBias/Si*Sw 0.0295314 /  ( 0.00784518 * 0.00345303 ) = 1090.14 -> 1090 -> 1090*2^-0
  rawBias/Si*Sw 0.00792045 /  ( 0.00784518 * 0.00345303 ) = 292.379 -> 292 -> 292*2^-0
  rawBias/Si*Sw -0.212149 /  ( 0.00784518 * 0.00345303 ) = -7831.37 -> -7831 -> -7831*2^-0
  rawBias/Si*Sw -0.159978 /  ( 0.00784518 * 0.00345303 ) = -5905.52 -> -5906 -> -5906*2^-0
  rawBias/Si*Sw -0.100598 /  ( 0.00784518 * 0.00345303 ) = -3713.51 -> -3714 -> -3714*2^-0
  rawBias/Si*Sw -0.0833697 /  ( 0.00784518 * 0.00345303 ) = -3077.55 -> -3078 -> -3078*2^-0
  rawBias/Si*Sw -0.234531 /  ( 0.00784518 * 0.00345303 ) = -8657.59 -> -8658 -> -8658*2^-0
  rawBias/Si*Sw -0.257588 /  ( 0.00784518 * 0.00345303 ) = -9508.72 -> -9509 -> -9509*2^-0
  rawBias/Si*Sw 0.211461 /  ( 0.00784518 * 0.00345303 ) = 7805.96 -> 7806 -> 7806*2^-0
  rawBias/Si*Sw -0.181789 /  ( 0.00784518 * 0.00345303 ) = -6710.65 -> -6711 -> -6711*2^-0
  rawBias/Si*Sw -0.289102 /  ( 0.00784518 * 0.00345303 ) = -10672.1 -> -10672 -> -10672*2^-0
  rawBias/Si*Sw -0.106559 /  ( 0.00784518 * 0.00345303 ) = -3933.57 -> -3934 -> -3934*2^-0
  rawBias/Si*Sw -0.24326 /  ( 0.00784518 * 0.00345303 ) = -8979.83 -> -8980 -> -8980*2^-0
  rawBias/Si*Sw 0.238864 /  ( 0.00784518 * 0.00345303 ) = 8817.55 -> 8818 -> 8818*2^-0
  rawBias/Si*Sw -0.0477089 /  ( 0.00784518 * 0.00345303 ) = -1761.15 -> -1761 -> -1761*2^-0
  rawBias/Si*Sw 0.183563 /  ( 0.00784518 * 0.00345303 ) = 6776.12 -> 6776 -> 6776*2^-0
  rawBias/Si*Sw 0.0266923 /  ( 0.0137246 * 0.00122683 ) = 1585.26 -> 1585 -> 1585*2^-0
  rawBias/Si*Sw -0.0210181 /  ( 0.0137246 * 0.00122683 ) = -1248.27 -> -1248 -> -1248*2^-0
  rawBias/Si*Sw -0.0988805 /  ( 0.0137246 * 0.00122683 ) = -5872.53 -> -5873 -> -5873*2^-0
  rawBias/Si*Sw 0.0149911 /  ( 0.0137246 * 0.00122683 ) = 890.325 -> 890 -> 890*2^-0
  rawBias/Si*Sw -0.0189492 /  ( 0.0137246 * 0.00122683 ) = -1125.4 -> -1125 -> -1125*2^-0
  rawBias/Si*Sw -0.0394126 /  ( 0.0137246 * 0.00122683 ) = -2340.72 -> -2341 -> -2341*2^-0
  rawBias/Si*Sw -0.00789565 /  ( 0.0137246 * 0.00122683 ) = -468.924 -> -469 -> -469*2^-0
  rawBias/Si*Sw -0.0799653 /  ( 0.0137246 * 0.00122683 ) = -4749.15 -> -4749 -> -4749*2^-0
  rawBias/Si*Sw -0.0767248 /  ( 0.0137246 * 0.00122683 ) = -4556.7 -> -4557 -> -4557*2^-0
  rawBias/Si*Sw -0.0982602 /  ( 0.0137246 * 0.00122683 ) = -5835.69 -> -5836 -> -5836*2^-0
  rawBias/Si*Sw -0.0398029 /  ( 0.0137246 * 0.00122683 ) = -2363.9 -> -2364 -> -2364*2^-0
  rawBias/Si*Sw -0.0778087 /  ( 0.0137246 * 0.00122683 ) = -4621.07 -> -4621 -> -4621*2^-0
  rawBias/Si*Sw -0.041269 /  ( 0.0137246 * 0.00122683 ) = -2450.97 -> -2451 -> -2451*2^-0
  rawBias/Si*Sw 0.0100064 /  ( 0.0137246 * 0.00122683 ) = 594.283 -> 594 -> 594*2^-0
  rawBias/Si*Sw -0.0130392 /  ( 0.0137246 * 0.00122683 ) = -774.398 -> -774 -> -774*2^-0
  rawBias/Si*Sw 0.0253316 /  ( 0.0137246 * 0.00122683 ) = 1504.45 -> 1504 -> 1504*2^-0
  rawBias/Si*Sw 0.0122957 /  ( 0.0137246 * 0.00122683 ) = 730.241 -> 730 -> 730*2^-0
  rawBias/Si*Sw -0.0409093 /  ( 0.0137246 * 0.00122683 ) = -2429.61 -> -2430 -> -2430*2^-0
  rawBias/Si*Sw -0.00850784 /  ( 0.0137246 * 0.00122683 ) = -505.282 -> -505 -> -505*2^-0
  rawBias/Si*Sw 0.00141418 /  ( 0.0137246 * 0.00122683 ) = 83.9884 -> 84 -> 84*2^-0
  rawBias/Si*Sw -0.0388875 /  ( 0.0137246 * 0.00122683 ) = -2309.53 -> -2310 -> -2310*2^-0
  rawBias/Si*Sw -0.0170651 /  ( 0.0137246 * 0.00122683 ) = -1013.5 -> -1013 -> -1013*2^-0
  rawBias/Si*Sw -0.0327771 /  ( 0.0137246 * 0.00122683 ) = -1946.64 -> -1947 -> -1947*2^-0
  rawBias/Si*Sw -0.0181817 /  ( 0.0137246 * 0.00122683 ) = -1079.82 -> -1080 -> -1080*2^-0
  rawBias/Si*Sw -0.086323 /  ( 0.0137246 * 0.00122683 ) = -5126.74 -> -5127 -> -5127*2^-0
  rawBias/Si*Sw -0.0200653 /  ( 0.0137246 * 0.00122683 ) = -1191.68 -> -1192 -> -1192*2^-0
  rawBias/Si*Sw -0.0210894 /  ( 0.0137246 * 0.00122683 ) = -1252.5 -> -1253 -> -1253*2^-0
  rawBias/Si*Sw -0.0629572 /  ( 0.0137246 * 0.00122683 ) = -3739.04 -> -3739 -> -3739*2^-0
  rawBias/Si*Sw -0.0672779 /  ( 0.0137246 * 0.00122683 ) = -3995.64 -> -3996 -> -3996*2^-0
  rawBias/Si*Sw 0.0461048 /  ( 0.0137246 * 0.00122683 ) = 2738.17 -> 2738 -> 2738*2^-0
  rawBias/Si*Sw -0.0657149 /  ( 0.0137246 * 0.00122683 ) = -3902.82 -> -3903 -> -3903*2^-0
  rawBias/Si*Sw -0.0425482 /  ( 0.0137246 * 0.00122683 ) = -2526.94 -> -2527 -> -2527*2^-0
  rawBias/Si*Sw 0.0145769 /  ( 0.0137246 * 0.00122683 ) = 865.723 -> 866 -> 866*2^-0
  rawBias/Si*Sw -0.0452918 /  ( 0.0137246 * 0.00122683 ) = -2689.88 -> -2690 -> -2690*2^-0
  rawBias/Si*Sw -0.0231088 /  ( 0.0137246 * 0.00122683 ) = -1372.44 -> -1372 -> -1372*2^-0
  rawBias/Si*Sw -0.106806 /  ( 0.0137246 * 0.00122683 ) = -6343.22 -> -6343 -> -6343*2^-0
  rawBias/Si*Sw -0.0240538 /  ( 0.0137246 * 0.00122683 ) = -1428.56 -> -1429 -> -1429*2^-0
  rawBias/Si*Sw -0.0203086 /  ( 0.0137246 * 0.00122683 ) = -1206.13 -> -1206 -> -1206*2^-0
  rawBias/Si*Sw -0.0370518 /  ( 0.0137246 * 0.00122683 ) = -2200.51 -> -2201 -> -2201*2^-0
  rawBias/Si*Sw -0.0241791 /  ( 0.0137246 * 0.00122683 ) = -1436 -> -1436 -> -1436*2^-0
  rawBias/Si*Sw -0.0967711 /  ( 0.0137246 * 0.00122683 ) = -5747.25 -> -5747 -> -5747*2^-0
  rawBias/Si*Sw 0.0139108 /  ( 0.0137246 * 0.00122683 ) = 826.167 -> 826 -> 826*2^-0
  rawBias/Si*Sw -0.0661912 /  ( 0.0137246 * 0.00122683 ) = -3931.1 -> -3931 -> -3931*2^-0
  rawBias/Si*Sw -0.0684465 /  ( 0.0137246 * 0.00122683 ) = -4065.05 -> -4065 -> -4065*2^-0
  rawBias/Si*Sw -0.0150072 /  ( 0.0137246 * 0.00122683 ) = -891.278 -> -891 -> -891*2^-0
  rawBias/Si*Sw 0.0310905 /  ( 0.0137246 * 0.00122683 ) = 1846.47 -> 1846 -> 1846*2^-0
  rawBias/Si*Sw -0.0493356 /  ( 0.0137246 * 0.00122683 ) = -2930.05 -> -2930 -> -2930*2^-0
  rawBias/Si*Sw -0.0828543 /  ( 0.0137246 * 0.00122683 ) = -4920.73 -> -4921 -> -4921*2^-0
  rawBias/Si*Sw 0.007227 /  ( 0.0137246 * 0.00122683 ) = 429.213 -> 429 -> 429*2^-0
  rawBias/Si*Sw -0.0534207 /  ( 0.0137246 * 0.00122683 ) = -3172.66 -> -3173 -> -3173*2^-0
  rawBias/Si*Sw -0.0160908 /  ( 0.0670743 * 0.000756184 ) = -317.243 -> -317 -> -317*2^-0
  rawBias/Si*Sw -0.00596529 /  ( 0.0670743 * 0.000756184 ) = -117.611 -> -118 -> -118*2^-0
  rawBias/Si*Sw 0.00571031 /  ( 0.0670743 * 0.000756184 ) = 112.584 -> 113 -> 113*2^-0
  rawBias/Si*Sw 0.0141845 /  ( 0.0670743 * 0.000756184 ) = 279.66 -> 280 -> 280*2^-0
  rawBias/Si*Sw 0.00457521 /  ( 0.0670743 * 0.000756184 ) = 90.2043 -> 90 -> 90*2^-0
  rawBias/Si*Sw 0.000918757 /  ( 0.0670743 * 0.000756184 ) = 18.1141 -> 18 -> 18*2^-0
  rawBias/Si*Sw 0.00182481 /  ( 0.0670743 * 0.000756184 ) = 35.9777 -> 36 -> 36*2^-0
  rawBias/Si*Sw 0.00487062 /  ( 0.0670743 * 0.000756184 ) = 96.0286 -> 96 -> 96*2^-0
  rawBias/Si*Sw -0.010953 /  ( 0.0670743 * 0.000756184 ) = -215.949 -> -216 -> -216*2^-0
  rawBias/Si*Sw 0.00405698 /  ( 0.0670743 * 0.000756184 ) = 79.9869 -> 80 -> 80*2^-0
  rawBias/Si*Sw 0.00241628 /  ( 0.0670743 * 0.000756184 ) = 47.6391 -> 48 -> 48*2^-0
  rawBias/Si*Sw 0.00215307 /  ( 0.0670743 * 0.000756184 ) = 42.4497 -> 42 -> 42*2^-0
  rawBias/Si*Sw -0.00274916 /  ( 0.0670743 * 0.000756184 ) = -54.2021 -> -54 -> -54*2^-0
  rawBias/Si*Sw 0.000693179 /  ( 0.0670743 * 0.000756184 ) = 13.6666 -> 14 -> 14*2^-0
  rawBias/Si*Sw 0.00434156 /  ( 0.0670743 * 0.000756184 ) = 85.5977 -> 86 -> 86*2^-0
  rawBias/Si*Sw -0.00366797 /  ( 0.0670743 * 0.000756184 ) = -72.3172 -> -72 -> -72*2^-0
  rawBias/Si*Sw -0.000154131 /  ( 0.0670743 * 0.000756184 ) = -3.03883 -> -3 -> -3*2^-0
  rawBias/Si*Sw 0.0017697 /  ( 0.0670743 * 0.000756184 ) = 34.8912 -> 35 -> 35*2^-0
  rawBias/Si*Sw -0.00575291 /  ( 0.0670743 * 0.000756184 ) = -113.424 -> -113 -> -113*2^-0
  rawBias/Si*Sw 0.00485401 /  ( 0.0670743 * 0.000756184 ) = 95.701 -> 96 -> 96*2^-0
  rawBias/Si*Sw 0.0049358 /  ( 0.0670743 * 0.000756184 ) = 97.3137 -> 97 -> 97*2^-0
  rawBias/Si*Sw 0.00053109 /  ( 0.0670743 * 0.000756184 ) = 10.4709 -> 10 -> 10*2^-0
  rawBias/Si*Sw 0.0035042 /  ( 0.0670743 * 0.000756184 ) = 69.0884 -> 69 -> 69*2^-0
  rawBias/Si*Sw 0.00115322 /  ( 0.0670743 * 0.000756184 ) = 22.7367 -> 23 -> 23*2^-0
  rawBias/Si*Sw -0.00513705 /  ( 0.0670743 * 0.000756184 ) = -101.281 -> -101 -> -101*2^-0
  rawBias/Si*Sw 0.00220661 /  ( 0.0670743 * 0.000756184 ) = 43.5053 -> 44 -> 44*2^-0
  rawBias/Si*Sw 0.0130855 /  ( 0.0670743 * 0.000756184 ) = 257.992 -> 258 -> 258*2^-0
  rawBias/Si*Sw -0.0040593 /  ( 0.0670743 * 0.000756184 ) = -80.0326 -> -80 -> -80*2^-0
  rawBias/Si*Sw 0.00742773 /  ( 0.0670743 * 0.000756184 ) = 146.444 -> 146 -> 146*2^-0
  rawBias/Si*Sw -0.00207535 /  ( 0.0670743 * 0.000756184 ) = -40.9173 -> -41 -> -41*2^-0
  rawBias/Si*Sw 0.00708569 /  ( 0.0670743 * 0.000756184 ) = 139.701 -> 140 -> 140*2^-0
  rawBias/Si*Sw 0.0130998 /  ( 0.0670743 * 0.000756184 ) = 258.273 -> 258 -> 258*2^-0
  rawBias/Si*Sw 0.00320843 /  ( 0.0670743 * 0.000756184 ) = 63.2571 -> 63 -> 63*2^-0
  rawBias/Si*Sw 0.00807043 /  ( 0.0670743 * 0.000756184 ) = 159.116 -> 159 -> 159*2^-0
  rawBias/Si*Sw -0.00725761 /  ( 0.0670743 * 0.000756184 ) = -143.09 -> -143 -> -143*2^-0
  rawBias/Si*Sw 0.00397733 /  ( 0.0670743 * 0.000756184 ) = 78.4166 -> 78 -> 78*2^-0
  rawBias/Si*Sw 0.000158773 /  ( 0.0670743 * 0.000756184 ) = 3.13036 -> 3 -> 3*2^-0
  rawBias/Si*Sw -0.00426002 /  ( 0.0670743 * 0.000756184 ) = -83.9899 -> -84 -> -84*2^-0
  rawBias/Si*Sw -0.00715113 /  ( 0.0670743 * 0.000756184 ) = -140.991 -> -141 -> -141*2^-0
  rawBias/Si*Sw 0.00737358 /  ( 0.0670743 * 0.000756184 ) = 145.377 -> 145 -> 145*2^-0
  rawBias/Si*Sw 0.000183469 /  ( 0.0670743 * 0.000756184 ) = 3.61726 -> 4 -> 4*2^-0
  rawBias/Si*Sw 0.0121654 /  ( 0.0670743 * 0.000756184 ) = 239.851 -> 240 -> 240*2^-0
  rawBias/Si*Sw -0.00185334 /  ( 0.0670743 * 0.000756184 ) = -36.5403 -> -37 -> -37*2^-0
  rawBias/Si*Sw -0.00478108 /  ( 0.0670743 * 0.000756184 ) = -94.2632 -> -94 -> -94*2^-0
  rawBias/Si*Sw -0.00362065 /  ( 0.0670743 * 0.000756184 ) = -71.3842 -> -71 -> -71*2^-0
  rawBias/Si*Sw 0.00572813 /  ( 0.0670743 * 0.000756184 ) = 112.935 -> 113 -> 113*2^-0
  rawBias/Si*Sw -0.000719333 /  ( 0.0670743 * 0.000756184 ) = -14.1823 -> -14 -> -14*2^-0
  rawBias/Si*Sw 0.0062139 /  ( 0.0670743 * 0.000756184 ) = 122.513 -> 123 -> 123*2^-0
  rawBias/Si*Sw -0.0042254 /  ( 0.0670743 * 0.000756184 ) = -83.3075 -> -83 -> -83*2^-0
  rawBias/Si*Sw 0.004558 /  ( 0.0670743 * 0.000756184 ) = 89.8649 -> 90 -> 90*2^-0
  rawBias/Si*Sw 0.00368481 /  ( 0.0670743 * 0.000756184 ) = 72.6492 -> 73 -> 73*2^-0
  rawBias/Si*Sw -0.00638158 /  ( 0.0670743 * 0.000756184 ) = -125.819 -> -126 -> -126*2^-0
  rawBias/Si*Sw -0.000542821 /  ( 0.0670743 * 0.000756184 ) = -10.7022 -> -11 -> -11*2^-0
  rawBias/Si*Sw 0.0056826 /  ( 0.0670743 * 0.000756184 ) = 112.037 -> 112 -> 112*2^-0
  rawBias/Si*Sw -0.0025529 /  ( 0.0670743 * 0.000756184 ) = -50.3327 -> -50 -> -50*2^-0
  rawBias/Si*Sw -0.000308287 /  ( 0.0670743 * 0.000756184 ) = -6.07815 -> -6 -> -6*2^-0
  rawBias/Si*Sw 0.00540015 /  ( 0.0670743 * 0.000756184 ) = 106.469 -> 106 -> 106*2^-0
  rawBias/Si*Sw 0.0128624 /  ( 0.0670743 * 0.000756184 ) = 253.594 -> 254 -> 254*2^-0
  rawBias/Si*Sw -0.00784788 /  ( 0.0670743 * 0.000756184 ) = -154.728 -> -155 -> -155*2^-0
  rawBias/Si*Sw -0.0110209 /  ( 0.0670743 * 0.000756184 ) = -217.286 -> -217 -> -217*2^-0
  rawBias/Si*Sw 0.00165233 /  ( 0.0670743 * 0.000756184 ) = 32.5771 -> 33 -> 33*2^-0
  rawBias/Si*Sw -0.00979184 /  ( 0.0670743 * 0.000756184 ) = -193.055 -> -193 -> -193*2^-0
  rawBias/Si*Sw 0.00627872 /  ( 0.0670743 * 0.000756184 ) = 123.79 -> 124 -> 124*2^-0
  rawBias/Si*Sw 0.00449704 /  ( 0.0670743 * 0.000756184 ) = 88.6631 -> 89 -> 89*2^-0
  rawBias/Si*Sw 0.00551083 /  ( 0.0670743 * 0.000756184 ) = 108.651 -> 109 -> 109*2^-0
  rawBias/Si*Sw 0.0103216 /  ( 0.0670743 * 0.000756184 ) = 203.499 -> 203 -> 203*2^-0
  rawBias/Si*Sw -8.54422e-05 /  ( 0.0670743 * 0.000756184 ) = -1.68457 -> -2 -> -2*2^-0
  rawBias/Si*Sw 0.0100825 /  ( 0.0670743 * 0.000756184 ) = 198.786 -> 199 -> 199*2^-0
  rawBias/Si*Sw -0.00381044 /  ( 0.0670743 * 0.000756184 ) = -75.1261 -> -75 -> -75*2^-0
  rawBias/Si*Sw 0.00709125 /  ( 0.0670743 * 0.000756184 ) = 139.81 -> 140 -> 140*2^-0
  rawBias/Si*Sw -0.00162287 /  ( 0.0670743 * 0.000756184 ) = -31.9964 -> -32 -> -32*2^-0
  rawBias/Si*Sw 0.0100035 /  ( 0.0670743 * 0.000756184 ) = 197.228 -> 197 -> 197*2^-0
  rawBias/Si*Sw 0.00153186 /  ( 0.0670743 * 0.000756184 ) = 30.2019 -> 30 -> 30*2^-0
  rawBias/Si*Sw -0.00192266 /  ( 0.0670743 * 0.000756184 ) = -37.9069 -> -38 -> -38*2^-0
  rawBias/Si*Sw 0.0177549 /  ( 0.0670743 * 0.000756184 ) = 350.053 -> 350 -> 350*2^-0
  rawBias/Si*Sw 0.0028109 /  ( 0.0670743 * 0.000756184 ) = 55.4193 -> 55 -> 55*2^-0
  rawBias/Si*Sw -0.000116088 /  ( 0.0670743 * 0.000756184 ) = -2.28878 -> -2 -> -2*2^-0
  rawBias/Si*Sw 0.00857696 /  ( 0.0670743 * 0.000756184 ) = 169.102 -> 169 -> 169*2^-0
  rawBias/Si*Sw 0.00336946 /  ( 0.0670743 * 0.000756184 ) = 66.4318 -> 66 -> 66*2^-0
  rawBias/Si*Sw -0.00577133 /  ( 0.0670743 * 0.000756184 ) = -113.787 -> -114 -> -114*2^-0
  rawBias/Si*Sw 0.00734607 /  ( 0.0670743 * 0.000756184 ) = 144.834 -> 145 -> 145*2^-0
  rawBias/Si*Sw 0.00367282 /  ( 0.0670743 * 0.000756184 ) = 72.4128 -> 72 -> 72*2^-0
  rawBias/Si*Sw 0.00290999 /  ( 0.0670743 * 0.000756184 ) = 57.3731 -> 57 -> 57*2^-0
  rawBias/Si*Sw -0.00764302 /  ( 0.0670743 * 0.000756184 ) = -150.689 -> -151 -> -151*2^-0
  rawBias/Si*Sw -0.00435251 /  ( 0.0670743 * 0.000756184 ) = -85.8135 -> -86 -> -86*2^-0
  rawBias/Si*Sw 0.0112651 /  ( 0.0670743 * 0.000756184 ) = 222.102 -> 222 -> 222*2^-0
  rawBias/Si*Sw -0.00474309 /  ( 0.0670743 * 0.000756184 ) = -93.5142 -> -94 -> -94*2^-0
  rawBias/Si*Sw -0.00293007 /  ( 0.0670743 * 0.000756184 ) = -57.769 -> -58 -> -58*2^-0
  rawBias/Si*Sw -0.00337636 /  ( 0.0670743 * 0.000756184 ) = -66.568 -> -67 -> -67*2^-0
  rawBias/Si*Sw 0.00350733 /  ( 0.0670743 * 0.000756184 ) = 69.1502 -> 69 -> 69*2^-0
  rawBias/Si*Sw 0.00413807 /  ( 0.0670743 * 0.000756184 ) = 81.5858 -> 82 -> 82*2^-0
  rawBias/Si*Sw 0.00648092 /  ( 0.0670743 * 0.000756184 ) = 127.777 -> 128 -> 128*2^-0
  rawBias/Si*Sw -0.00520403 /  ( 0.0670743 * 0.000756184 ) = -102.602 -> -103 -> -103*2^-0
  rawBias/Si*Sw -0.00946726 /  ( 0.0670743 * 0.000756184 ) = -186.655 -> -187 -> -187*2^-0
  rawBias/Si*Sw -0.00299138 /  ( 0.0670743 * 0.000756184 ) = -58.9778 -> -59 -> -59*2^-0
  rawBias/Si*Sw -0.0129122 /  ( 0.0670743 * 0.000756184 ) = -254.575 -> -255 -> -255*2^-0
  rawBias/Si*Sw 0.00296569 /  ( 0.0670743 * 0.000756184 ) = 58.4711 -> 58 -> 58*2^-0
  rawBias/Si*Sw -0.00357285 /  ( 0.0670743 * 0.000756184 ) = -70.4418 -> -70 -> -70*2^-0
  rawBias/Si*Sw 0.00962356 /  ( 0.0670743 * 0.000756184 ) = 189.737 -> 190 -> 190*2^-0
  rawBias/Si*Sw 0.00297598 /  ( 0.0670743 * 0.000756184 ) = 58.6741 -> 59 -> 59*2^-0
  rawBias/Si*Sw -0.00135108 /  ( 0.0670743 * 0.000756184 ) = -26.6377 -> -27 -> -27*2^-0
  rawBias/Si*Sw 0.00128038 /  ( 0.0670743 * 0.000756184 ) = 25.2438 -> 25 -> 25*2^-0
  rawBias/Si*Sw -0.00179181 /  ( 0.0670743 * 0.000756184 ) = -35.3272 -> -35 -> -35*2^-0
  rawBias/Si*Sw 0.010566 /  ( 0.0670743 * 0.000756184 ) = 208.318 -> 208 -> 208*2^-0
  rawBias/Si*Sw 0.0180151 /  ( 0.0670743 * 0.000756184 ) = 355.183 -> 355 -> 355*2^-0
  rawBias/Si*Sw -0.00460528 /  ( 0.0670743 * 0.000756184 ) = -90.797 -> -91 -> -91*2^-0
  rawBias/Si*Sw -0.000536081 /  ( 0.0670743 * 0.000756184 ) = -10.5693 -> -11 -> -11*2^-0
  rawBias/Si*Sw -0.00138855 /  ( 0.0670743 * 0.000756184 ) = -27.3764 -> -27 -> -27*2^-0
  rawBias/Si*Sw 0.00448689 /  ( 0.0670743 * 0.000756184 ) = 88.463 -> 88 -> 88*2^-0
  rawBias/Si*Sw 0.00400495 /  ( 0.0670743 * 0.000756184 ) = 78.9611 -> 79 -> 79*2^-0
  rawBias/Si*Sw 0.00287102 /  ( 0.0670743 * 0.000756184 ) = 56.6046 -> 57 -> 57*2^-0
  rawBias/Si*Sw 0.00166039 /  ( 0.0670743 * 0.000756184 ) = 32.7361 -> 33 -> 33*2^-0
  rawBias/Si*Sw -0.00473274 /  ( 0.0670743 * 0.000756184 ) = -93.3102 -> -93 -> -93*2^-0
  rawBias/Si*Sw 0.000981164 /  ( 0.0670743 * 0.000756184 ) = 19.3445 -> 19 -> 19*2^-0
  rawBias/Si*Sw -0.00258576 /  ( 0.0670743 * 0.000756184 ) = -50.9806 -> -51 -> -51*2^-0
  rawBias/Si*Sw -0.00459088 /  ( 0.0670743 * 0.000756184 ) = -90.5133 -> -91 -> -91*2^-0
  rawBias/Si*Sw -0.00192138 /  ( 0.0670743 * 0.000756184 ) = -37.8817 -> -38 -> -38*2^-0
  rawBias/Si*Sw 0.00659034 /  ( 0.0670743 * 0.000756184 ) = 129.934 -> 130 -> 130*2^-0
  rawBias/Si*Sw -0.00184609 /  ( 0.0670743 * 0.000756184 ) = -36.3974 -> -36 -> -36*2^-0
  rawBias/Si*Sw 0.000764893 /  ( 0.0670743 * 0.000756184 ) = 15.0805 -> 15 -> 15*2^-0
  rawBias/Si*Sw -0.00330961 /  ( 0.0670743 * 0.000756184 ) = -65.2518 -> -65 -> -65*2^-0
  rawBias/Si*Sw -0.00467795 /  ( 0.0670743 * 0.000756184 ) = -92.2298 -> -92 -> -92*2^-0
  rawBias/Si*Sw 0.00309991 /  ( 0.0670743 * 0.000756184 ) = 61.1174 -> 61 -> 61*2^-0
  rawBias/Si*Sw 0.00156627 /  ( 0.0670743 * 0.000756184 ) = 30.8804 -> 31 -> 31*2^-0
  rawBias/Si*Sw 0.00300166 /  ( 0.0670743 * 0.000756184 ) = 59.1803 -> 59 -> 59*2^-0
  rawBias/Si*Sw -0.00358254 /  ( 0.0670743 * 0.000756184 ) = -70.633 -> -71 -> -71*2^-0
  rawBias/Si*Sw -0.00415914 /  ( 0.0670743 * 0.000756184 ) = -82.0011 -> -82 -> -82*2^-0
  rawBias/Si*Sw 0.0100725 /  ( 0.0670743 * 0.000756184 ) = 198.589 -> 199 -> 199*2^-0
  rawBias/Si*Sw -0.0018276 /  ( 0.0670743 * 0.000756184 ) = -36.0328 -> -36 -> -36*2^-0
  rawBias/Si*Sw 0.00587635 /  ( 0.0670743 * 0.000756184 ) = 115.857 -> 116 -> 116*2^-0
  rawBias/Si*Sw 0.00282938 /  ( 0.0670743 * 0.000756184 ) = 55.7838 -> 56 -> 56*2^-0
  rawBias/Si*Sw 0.00114757 /  ( 0.0670743 * 0.000756184 ) = 22.6254 -> 23 -> 23*2^-0
  rawBias/Si*Sw 0.00344898 /  ( 0.0670743 * 0.000756184 ) = 67.9997 -> 68 -> 68*2^-0
  rawBias/Si*Sw -0.00504183 /  ( 0.0670743 * 0.000756184 ) = -99.404 -> -99 -> -99*2^-0
  rawBias/Si*Sw 0.0038491 /  ( 0.0670743 * 0.000756184 ) = 75.8883 -> 76 -> 76*2^-0
  rawBias/Si*Sw -0.000436917 /  ( 0.0670743 * 0.000756184 ) = -8.6142 -> -9 -> -9*2^-0
  rawBias/Si*Sw 0.0107106 /  ( 0.0670743 * 0.000756184 ) = 211.168 -> 211 -> 211*2^-0
  rawBias/Si*Sw 0.0101657 /  ( 0.0670743 * 0.000756184 ) = 200.426 -> 200 -> 200*2^-0
  rawBias/Si*Sw 0.000640607 /  ( 0.0670743 * 0.000756184 ) = 12.6301 -> 13 -> 13*2^-0
  rawBias/Si*Sw 0.000200644 /  ( 0.0670743 * 0.000756184 ) = 3.95588 -> 4 -> 4*2^-0
  rawBias/Si*Sw -0.00187112 /  ( 0.0670743 * 0.000756184 ) = -36.8908 -> -37 -> -37*2^-0
  rawBias/Si*Sw -0.000905704 /  ( 0.0670743 * 0.000756184 ) = -17.8567 -> -18 -> -18*2^-0
  rawBias/Si*Sw -0.00430365 /  ( 0.0670743 * 0.000756184 ) = -84.8502 -> -85 -> -85*2^-0
  rawBias/Si*Sw -0.0114306 /  ( 0.0670743 * 0.000756184 ) = -225.364 -> -225 -> -225*2^-0
  rawBias/Si*Sw -0.00254204 /  ( 0.0670743 * 0.000756184 ) = -50.1186 -> -50 -> -50*2^-0
  rawBias/Si*Sw 0.00999499 /  ( 0.0670743 * 0.000756184 ) = 197.06 -> 197 -> 197*2^-0
  rawBias/Si*Sw -0.00361041 /  ( 0.0670743 * 0.000756184 ) = -71.1824 -> -71 -> -71*2^-0
  rawBias/Si*Sw -0.000814386 /  ( 0.0670743 * 0.000756184 ) = -16.0563 -> -16 -> -16*2^-0
  rawBias/Si*Sw -0.00166425 /  ( 0.0670743 * 0.000756184 ) = -32.8121 -> -33 -> -33*2^-0
  rawBias/Si*Sw 0.0036898 /  ( 0.0670743 * 0.000756184 ) = 72.7476 -> 73 -> 73*2^-0
  rawBias/Si*Sw 0.00910051 /  ( 0.0670743 * 0.000756184 ) = 179.425 -> 179 -> 179*2^-0
  rawBias/Si*Sw -0.008081 /  ( 0.0670743 * 0.000756184 ) = -159.324 -> -159 -> -159*2^-0
  rawBias/Si*Sw -0.00621499 /  ( 0.0670743 * 0.000756184 ) = -122.534 -> -123 -> -123*2^-0
  rawBias/Si*Sw -0.00476039 /  ( 0.0670743 * 0.000756184 ) = -93.8552 -> -94 -> -94*2^-0
  rawBias/Si*Sw 0.00352937 /  ( 0.0670743 * 0.000756184 ) = 69.5847 -> 70 -> 70*2^-0
  rawBias/Si*Sw -0.00883984 /  ( 0.0670743 * 0.000756184 ) = -174.285 -> -174 -> -174*2^-0
  rawBias/Si*Sw 0.01267 /  ( 0.0670743 * 0.000756184 ) = 249.801 -> 250 -> 250*2^-0
  rawBias/Si*Sw 0.00553043 /  ( 0.0670743 * 0.000756184 ) = 109.037 -> 109 -> 109*2^-0
  rawBias/Si*Sw 0.00413786 /  ( 0.0670743 * 0.000756184 ) = 81.5815 -> 82 -> 82*2^-0
  rawBias/Si*Sw 0.00378476 /  ( 0.0670743 * 0.000756184 ) = 74.6199 -> 75 -> 75*2^-0
  rawBias/Si*Sw -0.000376468 /  ( 0.0670743 * 0.000756184 ) = -7.4224 -> -7 -> -7*2^-0
  rawBias/Si*Sw -0.00935109 /  ( 0.0670743 * 0.000756184 ) = -184.365 -> -184 -> -184*2^-0
  rawBias/Si*Sw 0.00111507 /  ( 0.0670743 * 0.000756184 ) = 21.9846 -> 22 -> 22*2^-0
  rawBias/Si*Sw 0.00449815 /  ( 0.0670743 * 0.000756184 ) = 88.6849 -> 89 -> 89*2^-0
  rawBias/Si*Sw 0.00232352 /  ( 0.0670743 * 0.000756184 ) = 45.8102 -> 46 -> 46*2^-0
  rawBias/Si*Sw -0.00526044 /  ( 0.0670743 * 0.000756184 ) = -103.714 -> -104 -> -104*2^-0
  rawBias/Si*Sw -0.00657905 /  ( 0.0670743 * 0.000756184 ) = -129.712 -> -130 -> -130*2^-0
  rawBias/Si*Sw 0.00202523 /  ( 0.0670743 * 0.000756184 ) = 39.9293 -> 40 -> 40*2^-0
  rawBias/Si*Sw 0.00783183 /  ( 0.0670743 * 0.000756184 ) = 154.411 -> 154 -> 154*2^-0
  rawBias/Si*Sw 0.00331134 /  ( 0.0670743 * 0.000756184 ) = 65.286 -> 65 -> 65*2^-0
  rawBias/Si*Sw 0.0113301 /  ( 0.0670743 * 0.000756184 ) = 223.384 -> 223 -> 223*2^-0
  rawBias/Si*Sw 0.000284586 /  ( 0.0670743 * 0.000756184 ) = 5.61086 -> 6 -> 6*2^-0
  rawBias/Si*Sw 0.00555074 /  ( 0.0670743 * 0.000756184 ) = 109.438 -> 109 -> 109*2^-0
  rawBias/Si*Sw -0.00561669 /  ( 0.0670743 * 0.000756184 ) = -110.738 -> -111 -> -111*2^-0
  rawBias/Si*Sw -0.00304549 /  ( 0.0670743 * 0.000756184 ) = -60.0444 -> -60 -> -60*2^-0
  rawBias/Si*Sw 0.00232793 /  ( 0.0670743 * 0.000756184 ) = 45.8973 -> 46 -> 46*2^-0
  rawBias/Si*Sw 0.00480359 /  ( 0.0670743 * 0.000756184 ) = 94.7071 -> 95 -> 95*2^-0
  rawBias/Si*Sw 0.00417361 /  ( 0.0670743 * 0.000756184 ) = 82.2863 -> 82 -> 82*2^-0
  rawBias/Si*Sw 0.00917487 /  ( 0.0670743 * 0.000756184 ) = 180.891 -> 181 -> 181*2^-0
  rawBias/Si*Sw -0.0106033 /  ( 0.0670743 * 0.000756184 ) = -209.054 -> -209 -> -209*2^-0
  rawBias/Si*Sw -0.00935452 /  ( 0.0670743 * 0.000756184 ) = -184.433 -> -184 -> -184*2^-0
  rawBias/Si*Sw 0.000472582 /  ( 0.0670743 * 0.000756184 ) = 9.31736 -> 9 -> 9*2^-0
  rawBias/Si*Sw -0.00565795 /  ( 0.0670743 * 0.000756184 ) = -111.551 -> -112 -> -112*2^-0
  rawBias/Si*Sw 0.00265887 /  ( 0.0670743 * 0.000756184 ) = 52.4219 -> 52 -> 52*2^-0
  rawBias/Si*Sw 0.00553717 /  ( 0.0670743 * 0.000756184 ) = 109.17 -> 109 -> 109*2^-0
  rawBias/Si*Sw 0.00830267 /  ( 0.0670743 * 0.000756184 ) = 163.694 -> 164 -> 164*2^-0
  rawBias/Si*Sw 0.00499382 /  ( 0.0670743 * 0.000756184 ) = 98.4575 -> 98 -> 98*2^-0
  rawBias/Si*Sw -0.00830142 /  ( 0.0670743 * 0.000756184 ) = -163.67 -> -164 -> -164*2^-0
  rawBias/Si*Sw 0.00125424 /  ( 0.0670743 * 0.000756184 ) = 24.7284 -> 25 -> 25*2^-0
  rawBias/Si*Sw 0.017598 /  ( 0.0670743 * 0.000756184 ) = 346.96 -> 347 -> 347*2^-0
  rawBias/Si*Sw 0.0110582 /  ( 0.0670743 * 0.000756184 ) = 218.023 -> 218 -> 218*2^-0
  rawBias/Si*Sw -0.00110636 /  ( 0.0670743 * 0.000756184 ) = -21.8128 -> -22 -> -22*2^-0
  rawBias/Si*Sw 0.00206649 /  ( 0.0670743 * 0.000756184 ) = 40.7427 -> 41 -> 41*2^-0
  rawBias/Si*Sw 0.00600223 /  ( 0.0670743 * 0.000756184 ) = 118.339 -> 118 -> 118*2^-0
  rawBias/Si*Sw 0.00580203 /  ( 0.0670743 * 0.000756184 ) = 114.392 -> 114 -> 114*2^-0
  rawBias/Si*Sw 0.0095547 /  ( 0.0670743 * 0.000756184 ) = 188.379 -> 188 -> 188*2^-0
  rawBias/Si*Sw -0.00347894 /  ( 0.0670743 * 0.000756184 ) = -68.5904 -> -69 -> -69*2^-0
  rawBias/Si*Sw -0.000947946 /  ( 0.0670743 * 0.000756184 ) = -18.6896 -> -19 -> -19*2^-0
  rawBias/Si*Sw -0.00558407 /  ( 0.0670743 * 0.000756184 ) = -110.095 -> -110 -> -110*2^-0
  rawBias/Si*Sw 0.00457049 /  ( 0.0670743 * 0.000756184 ) = 90.1113 -> 90 -> 90*2^-0
  rawBias/Si*Sw -0.00695887 /  ( 0.0670743 * 0.000756184 ) = -137.2 -> -137 -> -137*2^-0
  rawBias/Si*Sw 0.00425796 /  ( 0.0670743 * 0.000756184 ) = 83.9493 -> 84 -> 84*2^-0
  rawBias/Si*Sw 0.00692133 /  ( 0.0670743 * 0.000756184 ) = 136.46 -> 136 -> 136*2^-0
  rawBias/Si*Sw 0.0110334 /  ( 0.0670743 * 0.000756184 ) = 217.533 -> 218 -> 218*2^-0
  rawBias/Si*Sw 0.00080584 /  ( 0.0670743 * 0.000756184 ) = 15.8878 -> 16 -> 16*2^-0
  rawBias/Si*Sw 0.00967977 /  ( 0.0670743 * 0.000756184 ) = 190.845 -> 191 -> 191*2^-0
  rawBias/Si*Sw -0.00152148 /  ( 0.0670743 * 0.000756184 ) = -29.9974 -> -30 -> -30*2^-0
  rawBias/Si*Sw 0.000685625 /  ( 0.0670743 * 0.000756184 ) = 13.5177 -> 14 -> 14*2^-0
  rawBias/Si*Sw 0.0111667 /  ( 0.0670743 * 0.000756184 ) = 220.161 -> 220 -> 220*2^-0
  rawBias/Si*Sw -0.0122356 /  ( 0.0670743 * 0.000756184 ) = -241.235 -> -241 -> -241*2^-0
  rawBias/Si*Sw -0.00163696 /  ( 0.0670743 * 0.000756184 ) = -32.2742 -> -32 -> -32*2^-0
  rawBias/Si*Sw 0.00874573 /  ( 0.0670743 * 0.000756184 ) = 172.43 -> 172 -> 172*2^-0
  rawBias/Si*Sw -0.00747964 /  ( 0.0670743 * 0.000756184 ) = -147.468 -> -147 -> -147*2^-0
  rawBias/Si*Sw 9.53092e-05 /  ( 0.0670743 * 0.000756184 ) = 1.8791 -> 2 -> 2*2^-0
  rawBias/Si*Sw 0.00125609 /  ( 0.0670743 * 0.000756184 ) = 24.765 -> 25 -> 25*2^-0
  rawBias/Si*Sw -0.00484556 /  ( 0.0670743 * 0.000756184 ) = -95.5345 -> -96 -> -96*2^-0
  rawBias/Si*Sw 0.00964425 /  ( 0.0670743 * 0.000756184 ) = 190.145 -> 190 -> 190*2^-0
  rawBias/Si*Sw -0.0049331 /  ( 0.0670743 * 0.000756184 ) = -97.2603 -> -97 -> -97*2^-0
  rawBias/Si*Sw -0.00674903 /  ( 0.0670743 * 0.000756184 ) = -133.063 -> -133 -> -133*2^-0
  rawBias/Si*Sw 0.00155267 /  ( 0.0670743 * 0.000756184 ) = 30.6122 -> 31 -> 31*2^-0
  rawBias/Si*Sw 0.00779544 /  ( 0.0670743 * 0.000756184 ) = 153.694 -> 154 -> 154*2^-0
  rawBias/Si*Sw -0.00475451 /  ( 0.0670743 * 0.000756184 ) = -93.7393 -> -94 -> -94*2^-0
  rawBias/Si*Sw -0.00446209 /  ( 0.0670743 * 0.000756184 ) = -87.974 -> -88 -> -88*2^-0
  rawBias/Si*Sw 0.00994591 /  ( 0.0670743 * 0.000756184 ) = 196.092 -> 196 -> 196*2^-0
  rawBias/Si*Sw -0.00069266 /  ( 0.0670743 * 0.000756184 ) = -13.6564 -> -14 -> -14*2^-0
  rawBias/Si*Sw 0.0028227 /  ( 0.0670743 * 0.000756184 ) = 55.652 -> 56 -> 56*2^-0
  rawBias/Si*Sw -0.00315651 /  ( 0.0670743 * 0.000756184 ) = -62.2334 -> -62 -> -62*2^-0
  rawBias/Si*Sw -0.00449883 /  ( 0.0670743 * 0.000756184 ) = -88.6983 -> -89 -> -89*2^-0
  rawBias/Si*Sw 0.0124552 /  ( 0.0670743 * 0.000756184 ) = 245.565 -> 246 -> 246*2^-0
  rawBias/Si*Sw 0.00346543 /  ( 0.0670743 * 0.000756184 ) = 68.324 -> 68 -> 68*2^-0
  rawBias/Si*Sw -0.00341063 /  ( 0.0670743 * 0.000756184 ) = -67.2436 -> -67 -> -67*2^-0
  rawBias/Si*Sw -0.00535677 /  ( 0.0670743 * 0.000756184 ) = -105.613 -> -106 -> -106*2^-0
  rawBias/Si*Sw -0.00169873 /  ( 0.0670743 * 0.000756184 ) = -33.4919 -> -33 -> -33*2^-0
  rawBias/Si*Sw -0.000445362 /  ( 0.0670743 * 0.000756184 ) = -8.7807 -> -9 -> -9*2^-0
  rawBias/Si*Sw -0.0113099 /  ( 0.0670743 * 0.000756184 ) = -222.984 -> -223 -> -223*2^-0
  rawBias/Si*Sw -0.000209029 /  ( 0.0670743 * 0.000756184 ) = -4.12119 -> -4 -> -4*2^-0
  rawBias/Si*Sw -0.000552506 /  ( 0.0670743 * 0.000756184 ) = -10.8931 -> -11 -> -11*2^-0
  rawBias/Si*Sw 0.00152527 /  ( 0.0670743 * 0.000756184 ) = 30.0721 -> 30 -> 30*2^-0
  rawBias/Si*Sw 0.00539479 /  ( 0.0670743 * 0.000756184 ) = 106.363 -> 106 -> 106*2^-0
  rawBias/Si*Sw 0.01207 /  ( 0.0670743 * 0.000756184 ) = 237.971 -> 238 -> 238*2^-0
  rawBias/Si*Sw 0.00102972 /  ( 0.0670743 * 0.000756184 ) = 20.3018 -> 20 -> 20*2^-0
  rawBias/Si*Sw 0.00209548 /  ( 0.0670743 * 0.000756184 ) = 41.3141 -> 41 -> 41*2^-0
  rawBias/Si*Sw -0.000864239 /  ( 0.0670743 * 0.000756184 ) = -17.0392 -> -17 -> -17*2^-0
  rawBias/Si*Sw 0.0142878 /  ( 0.0670743 * 0.000756184 ) = 281.696 -> 282 -> 282*2^-0
  rawBias/Si*Sw -0.00662105 /  ( 0.0670743 * 0.000756184 ) = -130.54 -> -131 -> -131*2^-0
  rawBias/Si*Sw -0.000244511 /  ( 0.0670743 * 0.000756184 ) = -4.82075 -> -5 -> -5*2^-0
  rawBias/Si*Sw 0.0163849 /  ( 0.0670743 * 0.000756184 ) = 323.043 -> 323 -> 323*2^-0
  rawBias/Si*Sw 0.0115843 /  ( 0.0670743 * 0.000756184 ) = 228.395 -> 228 -> 228*2^-0
  rawBias/Si*Sw -0.000624234 /  ( 0.0670743 * 0.000756184 ) = -12.3073 -> -12 -> -12*2^-0
  rawBias/Si*Sw 0.00149396 /  ( 0.0670743 * 0.000756184 ) = 29.4547 -> 29 -> 29*2^-0
  rawBias/Si*Sw -0.00786256 /  ( 0.0670743 * 0.000756184 ) = -155.017 -> -155 -> -155*2^-0
  rawBias/Si*Sw 0.00504049 /  ( 0.0670743 * 0.000756184 ) = 99.3777 -> 99 -> 99*2^-0
  rawBias/Si*Sw -0.00208822 /  ( 0.0670743 * 0.000756184 ) = -41.1711 -> -41 -> -41*2^-0
  rawBias/Si*Sw -0.0053671 /  ( 0.0670743 * 0.000756184 ) = -105.817 -> -106 -> -106*2^-0
  rawBias/Si*Sw 0.00282316 /  ( 0.0670743 * 0.000756184 ) = 55.661 -> 56 -> 56*2^-0
  rawBias/Si*Sw 0.0123619 /  ( 0.0670743 * 0.000756184 ) = 243.726 -> 244 -> 244*2^-0
  rawBias/Si*Sw 0.00466023 /  ( 0.0670743 * 0.000756184 ) = 91.8804 -> 92 -> 92*2^-0
  rawBias/Si*Sw -0.00132708 /  ( 0.0670743 * 0.000756184 ) = -26.1645 -> -26 -> -26*2^-0
  rawBias/Si*Sw -0.00287518 /  ( 0.0670743 * 0.000756184 ) = -56.6867 -> -57 -> -57*2^-0
  rawBias/Si*Sw 0.00934964 /  ( 0.0670743 * 0.000756184 ) = 184.336 -> 184 -> 184*2^-0
  rawBias/Si*Sw 0.0016545 /  ( 0.0670743 * 0.000756184 ) = 32.62 -> 33 -> 33*2^-0
  rawBias/Si*Sw -0.00141152 /  ( 0.0670743 * 0.000756184 ) = -27.8293 -> -28 -> -28*2^-0
  rawBias/Si*Sw -0.00762542 /  ( 0.0670743 * 0.000756184 ) = -150.342 -> -150 -> -150*2^-0
  rawBias/Si*Sw 0.00133011 /  ( 0.0670743 * 0.000756184 ) = 26.2243 -> 26 -> 26*2^-0
  rawBias/Si*Sw 0.00354479 /  ( 0.0670743 * 0.000756184 ) = 69.8886 -> 70 -> 70*2^-0
  rawBias/Si*Sw -0.000554709 /  ( 0.0670743 * 0.000756184 ) = -10.9366 -> -11 -> -11*2^-0
  rawBias/Si*Sw 0.00377187 /  ( 0.0670743 * 0.000756184 ) = 74.3657 -> 74 -> 74*2^-0
  rawBias/Si*Sw -0.00410209 /  ( 0.0670743 * 0.000756184 ) = -80.8764 -> -81 -> -81*2^-0
  rawBias/Si*Sw -0.0126422 /  ( 0.0670743 * 0.000756184 ) = -249.252 -> -249 -> -249*2^-0
  rawBias/Si*Sw 0.00330497 /  ( 0.0670743 * 0.000756184 ) = 65.1604 -> 65 -> 65*2^-0
  rawBias/Si*Sw 0.00150753 /  ( 0.0670743 * 0.000756184 ) = 29.7222 -> 30 -> 30*2^-0
  rawBias/Si*Sw 0.00671193 /  ( 0.0670743 * 0.000756184 ) = 132.331 -> 132 -> 132*2^-0
  rawBias/Si*Sw -0.000671467 /  ( 0.0670743 * 0.000756184 ) = -13.2386 -> -13 -> -13*2^-0
  rawBias/Si*Sw -0.0020892 /  ( 0.0670743 * 0.000756184 ) = -41.1904 -> -41 -> -41*2^-0
  rawBias/Si*Sw -0.00214926 /  ( 0.0670743 * 0.000756184 ) = -42.3746 -> -42 -> -42*2^-0
  rawBias/Si*Sw 0.0089044 /  ( 0.0670743 * 0.000756184 ) = 175.558 -> 176 -> 176*2^-0
  rawBias/Si*Sw -0.00174577 /  ( 0.0670743 * 0.000756184 ) = -34.4194 -> -34 -> -34*2^-0
  rawBias/Si*Sw -0.00241748 /  ( 0.0670743 * 0.000756184 ) = -47.6628 -> -48 -> -48*2^-0
  rawBias/Si*Sw 0.00212694 /  ( 0.0670743 * 0.000756184 ) = 41.9346 -> 42 -> 42*2^-0
  rawBias/Si*Sw 0.00360922 /  ( 0.0670743 * 0.000756184 ) = 71.159 -> 71 -> 71*2^-0
  rawBias/Si*Sw 0.000103615 /  ( 0.0670743 * 0.000756184 ) = 2.04286 -> 2 -> 2*2^-0
  rawBias/Si*Sw -0.00459254 /  ( 0.0670743 * 0.000756184 ) = -90.546 -> -91 -> -91*2^-0
  rawBias/Si*Sw 0.0130479 /  ( 0.0670743 * 0.000756184 ) = 257.251 -> 257 -> 257*2^-0
  rawBias/Si*Sw 0.00234114 /  ( 0.0670743 * 0.000756184 ) = 46.1575 -> 46 -> 46*2^-0
  rawBias/Si*Sw 0.00618787 /  ( 0.0670743 * 0.000756184 ) = 121.999 -> 122 -> 122*2^-0
  rawBias/Si*Sw 0.00378565 /  ( 0.0670743 * 0.000756184 ) = 74.6374 -> 75 -> 75*2^-0
  rawBias/Si*Sw 0.00384775 /  ( 0.0670743 * 0.000756184 ) = 75.8618 -> 76 -> 76*2^-0
  rawBias/Si*Sw -0.00576199 /  ( 0.0670743 * 0.000756184 ) = -113.603 -> -114 -> -114*2^-0
  rawBias/Si*Sw 0.00337918 /  ( 0.0670743 * 0.000756184 ) = 66.6235 -> 67 -> 67*2^-0
  rawBias/Si*Sw -0.00326975 /  ( 0.0670743 * 0.000756184 ) = -64.466 -> -64 -> -64*2^-0
  rawBias/Si*Sw -0.00731954 /  ( 0.0670743 * 0.000756184 ) = -144.311 -> -144 -> -144*2^-0
  rawBias/Si*Sw -0.00448419 /  ( 0.0670743 * 0.000756184 ) = -88.4096 -> -88 -> -88*2^-0
  rawBias/Si*Sw -0.00779377 /  ( 0.0670743 * 0.000756184 ) = -153.661 -> -154 -> -154*2^-0
  rawBias/Si*Sw -0.00449993 /  ( 0.0670743 * 0.000756184 ) = -88.7201 -> -89 -> -89*2^-0
  rawBias/Si*Sw -0.00854937 /  ( 0.0670743 * 0.000756184 ) = -168.558 -> -169 -> -169*2^-0
  rawBias/Si*Sw 0.00736093 /  ( 0.0670743 * 0.000756184 ) = 145.127 -> 145 -> 145*2^-0
  rawBias/Si*Sw 0.00287959 /  ( 0.0670743 * 0.000756184 ) = 56.7737 -> 57 -> 57*2^-0
  rawBias/Si*Sw 0.0059524 /  ( 0.0670743 * 0.000756184 ) = 117.357 -> 117 -> 117*2^-0
  rawBias/Si*Sw 0.000887351 /  ( 0.0670743 * 0.000756184 ) = 17.4949 -> 17 -> 17*2^-0
  rawBias/Si*Sw -0.00268741 /  ( 0.0670743 * 0.000756184 ) = -52.9846 -> -53 -> -53*2^-0
  rawBias/Si*Sw 0.00199675 /  ( 0.0670743 * 0.000756184 ) = 39.3677 -> 39 -> 39*2^-0
  rawBias/Si*Sw -0.00819081 /  ( 0.0670743 * 0.000756184 ) = -161.489 -> -161 -> -161*2^-0
  rawBias/Si*Sw -0.00205734 /  ( 0.0670743 * 0.000756184 ) = -40.5623 -> -41 -> -41*2^-0
  rawBias/Si*Sw 0.00159371 /  ( 0.0670743 * 0.000756184 ) = 31.4215 -> 31 -> 31*2^-0
  rawBias/Si*Sw -0.00959275 /  ( 0.0670743 * 0.000756184 ) = -189.13 -> -189 -> -189*2^-0
  rawBias/Si*Sw -0.00528138 /  ( 0.0670743 * 0.000756184 ) = -104.127 -> -104 -> -104*2^-0
  rawBias/Si*Sw 0.00541374 /  ( 0.0670743 * 0.000756184 ) = 106.737 -> 107 -> 107*2^-0
  rawBias/Si*Sw -0.00262693 /  ( 0.0670743 * 0.000756184 ) = -51.7922 -> -52 -> -52*2^-0
  rawBias/Si*Sw -0.00522113 /  ( 0.0670743 * 0.000756184 ) = -102.939 -> -103 -> -103*2^-0
  rawBias/Si*Sw 0.00781175 /  ( 0.0670743 * 0.000756184 ) = 154.016 -> 154 -> 154*2^-0
  rawBias/Si*Sw 0.0028028 /  ( 0.0670743 * 0.000756184 ) = 55.2598 -> 55 -> 55*2^-0
  rawBias/Si*Sw 0.00183135 /  ( 0.0670743 * 0.000756184 ) = 36.1068 -> 36 -> 36*2^-0
  rawBias/Si*Sw 0.00101005 /  ( 0.0670743 * 0.000756184 ) = 19.9141 -> 20 -> 20*2^-0
  rawBias/Si*Sw 0.00178905 /  ( 0.0670743 * 0.000756184 ) = 35.2726 -> 35 -> 35*2^-0
  rawBias/Si*Sw 0.00425291 /  ( 0.0670743 * 0.000756184 ) = 83.8499 -> 84 -> 84*2^-0
  rawBias/Si*Sw 0.00325326 /  ( 0.0670743 * 0.000756184 ) = 64.1408 -> 64 -> 64*2^-0
  rawBias/Si*Sw -8.29429e-05 /  ( 0.0670743 * 0.000756184 ) = -1.63529 -> -2 -> -2*2^-0
  rawBias/Si*Sw -0.00223922 /  ( 0.0670743 * 0.000756184 ) = -44.1482 -> -44 -> -44*2^-0
  rawBias/Si*Sw -0.0016051 /  ( 0.0670743 * 0.000756184 ) = -31.646 -> -32 -> -32*2^-0
  rawBias/Si*Sw -0.000555084 /  ( 0.0670743 * 0.000756184 ) = -10.944 -> -11 -> -11*2^-0
  rawBias/Si*Sw -0.00882739 /  ( 0.0670743 * 0.000756184 ) = -174.04 -> -174 -> -174*2^-0
  rawBias/Si*Sw 0.00275148 /  ( 0.0670743 * 0.000756184 ) = 54.2479 -> 54 -> 54*2^-0
  rawBias/Si*Sw -0.0073338 /  ( 0.0670743 * 0.000756184 ) = -144.592 -> -145 -> -145*2^-0
  rawBias/Si*Sw -0.0130495 /  ( 0.0670743 * 0.000756184 ) = -257.282 -> -257 -> -257*2^-0
  rawBias/Si*Sw -0.00503639 /  ( 0.0670743 * 0.000756184 ) = -99.2968 -> -99 -> -99*2^-0
  rawBias/Si*Sw -0.00335259 /  ( 0.0670743 * 0.000756184 ) = -66.0993 -> -66 -> -66*2^-0
  rawBias/Si*Sw -0.00762874 /  ( 0.0670743 * 0.000756184 ) = -150.407 -> -150 -> -150*2^-0
  rawBias/Si*Sw -0.000773354 /  ( 0.0670743 * 0.000756184 ) = -15.2474 -> -15 -> -15*2^-0
  rawBias/Si*Sw -0.0105282 /  ( 0.0670743 * 0.000756184 ) = -207.573 -> -208 -> -208*2^-0
  rawBias/Si*Sw -0.00604246 /  ( 0.0670743 * 0.000756184 ) = -119.132 -> -119 -> -119*2^-0
  rawBias/Si*Sw -0.0148792 /  ( 0.0670743 * 0.000756184 ) = -293.356 -> -293 -> -293*2^-0
  rawBias/Si*Sw 0.00509254 /  ( 0.0670743 * 0.000756184 ) = 100.404 -> 100 -> 100*2^-0
  rawBias/Si*Sw -0.00343132 /  ( 0.0670743 * 0.000756184 ) = -67.6514 -> -68 -> -68*2^-0
  rawBias/Si*Sw 0.000884271 /  ( 0.0670743 * 0.000756184 ) = 17.4342 -> 17 -> 17*2^-0
  rawBias/Si*Sw -0.00659211 /  ( 0.0670743 * 0.000756184 ) = -129.969 -> -130 -> -130*2^-0
  rawBias/Si*Sw -0.00150819 /  ( 0.0670743 * 0.000756184 ) = -29.7353 -> -30 -> -30*2^-0
  rawBias/Si*Sw -0.00066668 /  ( 0.0670743 * 0.000756184 ) = -13.1442 -> -13 -> -13*2^-0
  rawBias/Si*Sw -0.0123335 /  ( 0.0670743 * 0.000756184 ) = -243.166 -> -243 -> -243*2^-0
  rawBias/Si*Sw 0.00527709 /  ( 0.0670743 * 0.000756184 ) = 104.042 -> 104 -> 104*2^-0
  rawBias/Si*Sw 0.0030958 /  ( 0.0670743 * 0.000756184 ) = 61.0364 -> 61 -> 61*2^-0
  rawBias/Si*Sw -0.000243152 /  ( 0.0670743 * 0.000756184 ) = -4.79396 -> -5 -> -5*2^-0
  rawBias/Si*Sw 0.00163575 /  ( 0.0670743 * 0.000756184 ) = 32.2503 -> 32 -> 32*2^-0
  rawBias/Si*Sw 0.00305789 /  ( 0.0670743 * 0.000756184 ) = 60.2889 -> 60 -> 60*2^-0
  rawBias/Si*Sw -0.0032451 /  ( 0.0670743 * 0.000756184 ) = -63.9799 -> -64 -> -64*2^-0
  rawBias/Si*Sw 0.00757511 /  ( 0.0670743 * 0.000756184 ) = 149.35 -> 149 -> 149*2^-0
  rawBias/Si*Sw -0.00441922 /  ( 0.0670743 * 0.000756184 ) = -87.1289 -> -87 -> -87*2^-0
  rawBias/Si*Sw -0.00156038 /  ( 0.0670743 * 0.000756184 ) = -30.7643 -> -31 -> -31*2^-0
  rawBias/Si*Sw -0.00333063 /  ( 0.0670743 * 0.000756184 ) = -65.6662 -> -66 -> -66*2^-0
  rawBias/Si*Sw -0.00142264 /  ( 0.0670743 * 0.000756184 ) = -28.0485 -> -28 -> -28*2^-0
  rawBias/Si*Sw -0.0023367 /  ( 0.0670743 * 0.000756184 ) = -46.0701 -> -46 -> -46*2^-0
  rawBias/Si*Sw -0.000629858 /  ( 0.0670743 * 0.000756184 ) = -12.4182 -> -12 -> -12*2^-0
  rawBias/Si*Sw -0.00742183 /  ( 0.0670743 * 0.000756184 ) = -146.328 -> -146 -> -146*2^-0
  rawBias/Si*Sw -0.00525475 /  ( 0.0670743 * 0.000756184 ) = -103.602 -> -104 -> -104*2^-0
  rawBias/Si*Sw 0.00106295 /  ( 0.0670743 * 0.000756184 ) = 20.957 -> 21 -> 21*2^-0
  rawBias/Si*Sw 0.00557893 /  ( 0.0670743 * 0.000756184 ) = 109.994 -> 110 -> 110*2^-0
  rawBias/Si*Sw -0.00735296 /  ( 0.0670743 * 0.000756184 ) = -144.97 -> -145 -> -145*2^-0
  rawBias/Si*Sw 0.0104845 /  ( 0.0670743 * 0.000756184 ) = 206.71 -> 207 -> 207*2^-0
  rawBias/Si*Sw 0.00501339 /  ( 0.0670743 * 0.000756184 ) = 98.8435 -> 99 -> 99*2^-0
  rawBias/Si*Sw -8.00025e-05 /  ( 0.0670743 * 0.000756184 ) = -1.57732 -> -2 -> -2*2^-0
  rawBias/Si*Sw -0.00386855 /  ( 0.0670743 * 0.000756184 ) = -76.2719 -> -76 -> -76*2^-0
  rawBias/Si*Sw 0.0118741 /  ( 0.0670743 * 0.000756184 ) = 234.108 -> 234 -> 234*2^-0
  rawBias/Si*Sw 0.00171545 /  ( 0.0670743 * 0.000756184 ) = 33.8216 -> 34 -> 34*2^-0
  rawBias/Si*Sw -0.00241817 /  ( 0.0670743 * 0.000756184 ) = -47.6763 -> -48 -> -48*2^-0
  rawBias/Si*Sw -0.00286085 /  ( 0.0670743 * 0.000756184 ) = -56.4042 -> -56 -> -56*2^-0
  rawBias/Si*Sw 0.0102796 /  ( 0.0670743 * 0.000756184 ) = 202.671 -> 203 -> 203*2^-0
  rawBias/Si*Sw 0.00680886 /  ( 0.0670743 * 0.000756184 ) = 134.243 -> 134 -> 134*2^-0
  rawBias/Si*Sw 0.00343526 /  ( 0.0670743 * 0.000756184 ) = 67.7291 -> 68 -> 68*2^-0
  rawBias/Si*Sw -0.00503297 /  ( 0.0670743 * 0.000756184 ) = -99.2294 -> -99 -> -99*2^-0
  rawBias/Si*Sw 0.00212876 /  ( 0.0670743 * 0.000756184 ) = 41.9703 -> 42 -> 42*2^-0
  rawBias/Si*Sw 0.00683794 /  ( 0.0670743 * 0.000756184 ) = 134.816 -> 135 -> 135*2^-0
  rawBias/Si*Sw -0.003925 /  ( 0.0670743 * 0.000756184 ) = -77.3847 -> -77 -> -77*2^-0
  rawBias/Si*Sw 0.000772909 /  ( 0.0670743 * 0.000756184 ) = 15.2386 -> 15 -> 15*2^-0
  rawBias/Si*Sw 0.00182556 /  ( 0.0670743 * 0.000756184 ) = 35.9926 -> 36 -> 36*2^-0
  rawBias/Si*Sw -0.0014057 /  ( 0.0670743 * 0.000756184 ) = -27.7146 -> -28 -> -28*2^-0
  rawBias/Si*Sw 0.00537661 /  ( 0.0670743 * 0.000756184 ) = 106.005 -> 106 -> 106*2^-0
  rawBias/Si*Sw -0.00120791 /  ( 0.0670743 * 0.000756184 ) = -23.8149 -> -24 -> -24*2^-0
  rawBias/Si*Sw 0.00606066 /  ( 0.0670743 * 0.000756184 ) = 119.491 -> 119 -> 119*2^-0
  rawBias/Si*Sw -0.00678337 /  ( 0.0670743 * 0.000756184 ) = -133.74 -> -134 -> -134*2^-0
  rawBias/Si*Sw -0.00536369 /  ( 0.0670743 * 0.000756184 ) = -105.75 -> -106 -> -106*2^-0
  rawBias/Si*Sw 0.00152798 /  ( 0.0670743 * 0.000756184 ) = 30.1254 -> 30 -> 30*2^-0
  rawBias/Si*Sw 0.00113391 /  ( 0.0670743 * 0.000756184 ) = 22.356 -> 22 -> 22*2^-0
  rawBias/Si*Sw -0.00381443 /  ( 0.0670743 * 0.000756184 ) = -75.2048 -> -75 -> -75*2^-0
  rawBias/Si*Sw 0.00313435 /  ( 0.0670743 * 0.000756184 ) = 61.7964 -> 62 -> 62*2^-0
  rawBias/Si*Sw 0.00663192 /  ( 0.0670743 * 0.000756184 ) = 130.754 -> 131 -> 131*2^-0
  rawBias/Si*Sw -0.00171548 /  ( 0.0670743 * 0.000756184 ) = -33.8221 -> -34 -> -34*2^-0
  rawBias/Si*Sw -0.00413377 /  ( 0.0670743 * 0.000756184 ) = -81.5008 -> -82 -> -82*2^-0
  rawBias/Si*Sw -0.00378595 /  ( 0.0670743 * 0.000756184 ) = -74.6434 -> -75 -> -75*2^-0
  rawBias/Si*Sw 0.00658897 /  ( 0.0670743 * 0.000756184 ) = 129.907 -> 130 -> 130*2^-0
  rawBias/Si*Sw 0.0115695 /  ( 0.0670743 * 0.000756184 ) = 228.103 -> 228 -> 228*2^-0
  rawBias/Si*Sw -0.00290412 /  ( 0.0670743 * 0.000756184 ) = -57.2574 -> -57 -> -57*2^-0
  rawBias/Si*Sw 0.00122328 /  ( 0.0670743 * 0.000756184 ) = 24.1181 -> 24 -> 24*2^-0
  rawBias/Si*Sw -0.00573779 /  ( 0.0670743 * 0.000756184 ) = -113.126 -> -113 -> -113*2^-0
  rawBias/Si*Sw -0.00666367 /  ( 0.0670743 * 0.000756184 ) = -131.38 -> -131 -> -131*2^-0
  rawBias/Si*Sw 0.00148788 /  ( 0.0670743 * 0.000756184 ) = 29.3349 -> 29 -> 29*2^-0
  rawBias/Si*Sw -9.36032e-05 /  ( 0.0670743 * 0.000756184 ) = -1.84547 -> -2 -> -2*2^-0
  rawBias/Si*Sw -0.00523711 /  ( 0.0670743 * 0.000756184 ) = -103.254 -> -103 -> -103*2^-0
  rawBias/Si*Sw 0.00100092 /  ( 0.0670743 * 0.000756184 ) = 19.7339 -> 20 -> 20*2^-0
  rawBias/Si*Sw -0.00102205 /  ( 0.0670743 * 0.000756184 ) = -20.1506 -> -20 -> -20*2^-0
  rawBias/Si*Sw -0.00262461 /  ( 0.0670743 * 0.000756184 ) = -51.7465 -> -52 -> -52*2^-0
  rawBias/Si*Sw -0.00398723 /  ( 0.0670743 * 0.000756184 ) = -78.6118 -> -79 -> -79*2^-0
  rawBias/Si*Sw -0.0089795 /  ( 0.0670743 * 0.000756184 ) = -177.039 -> -177 -> -177*2^-0
  rawBias/Si*Sw -0.00639421 /  ( 0.0670743 * 0.000756184 ) = -126.068 -> -126 -> -126*2^-0
  rawBias/Si*Sw -0.00406703 /  ( 0.0670743 * 0.000756184 ) = -80.1851 -> -80 -> -80*2^-0
  rawBias/Si*Sw -0.00919813 /  ( 0.0670743 * 0.000756184 ) = -181.349 -> -181 -> -181*2^-0
  rawBias/Si*Sw -0.00157012 /  ( 0.0670743 * 0.000756184 ) = -30.9562 -> -31 -> -31*2^-0
  rawBias/Si*Sw -0.00926236 /  ( 0.0670743 * 0.000756184 ) = -182.616 -> -183 -> -183*2^-0
  rawBias/Si*Sw 0.000345377 /  ( 0.0670743 * 0.000756184 ) = 6.80942 -> 7 -> 7*2^-0
  rawBias/Si*Sw 0.0134298 /  ( 0.0670743 * 0.000756184 ) = 264.78 -> 265 -> 265*2^-0
  rawBias/Si*Sw 0.00564769 /  ( 0.0670743 * 0.000756184 ) = 111.349 -> 111 -> 111*2^-0
  rawBias/Si*Sw 0.0109816 /  ( 0.0670743 * 0.000756184 ) = 216.512 -> 217 -> 217*2^-0
  rawBias/Si*Sw 0.00467368 /  ( 0.0670743 * 0.000756184 ) = 92.1457 -> 92 -> 92*2^-0
  rawBias/Si*Sw -0.00677339 /  ( 0.0670743 * 0.000756184 ) = -133.543 -> -134 -> -134*2^-0
  rawBias/Si*Sw -0.00471123 /  ( 0.0670743 * 0.000756184 ) = -92.886 -> -93 -> -93*2^-0
  rawBias/Si*Sw 0.00475499 /  ( 0.0670743 * 0.000756184 ) = 93.7488 -> 94 -> 94*2^-0
  rawBias/Si*Sw -0.00101872 /  ( 0.0670743 * 0.000756184 ) = -20.0849 -> -20 -> -20*2^-0
  rawBias/Si*Sw 0.00660738 /  ( 0.0670743 * 0.000756184 ) = 130.27 -> 130 -> 130*2^-0
  rawBias/Si*Sw 0.00300622 /  ( 0.0670743 * 0.000756184 ) = 59.2703 -> 59 -> 59*2^-0
  rawBias/Si*Sw -0.00118807 /  ( 0.0670743 * 0.000756184 ) = -23.4239 -> -23 -> -23*2^-0
  rawBias/Si*Sw 0.00205993 /  ( 0.0670743 * 0.000756184 ) = 40.6133 -> 41 -> 41*2^-0
  rawBias/Si*Sw -0.00326071 /  ( 0.0670743 * 0.000756184 ) = -64.2877 -> -64 -> -64*2^-0
  rawBias/Si*Sw 0.00502191 /  ( 0.0670743 * 0.000756184 ) = 99.0113 -> 99 -> 99*2^-0
  rawBias/Si*Sw -0.010096 /  ( 0.0670743 * 0.000756184 ) = -199.051 -> -199 -> -199*2^-0
  rawBias/Si*Sw -0.0014445 /  ( 0.0670743 * 0.000756184 ) = -28.4796 -> -28 -> -28*2^-0
  rawBias/Si*Sw 0.00273264 /  ( 0.0670743 * 0.000756184 ) = 53.8764 -> 54 -> 54*2^-0
  rawBias/Si*Sw 0.00262441 /  ( 0.0670743 * 0.000756184 ) = 51.7425 -> 52 -> 52*2^-0
  rawBias/Si*Sw -0.000504633 /  ( 0.0670743 * 0.000756184 ) = -9.94928 -> -10 -> -10*2^-0
  rawBias/Si*Sw 0.0147181 /  ( 0.0670743 * 0.000756184 ) = 290.18 -> 290 -> 290*2^-0
  rawBias/Si*Sw -0.00540654 /  ( 0.0670743 * 0.000756184 ) = -106.595 -> -107 -> -107*2^-0
  rawBias/Si*Sw -0.00326369 /  ( 0.0670743 * 0.000756184 ) = -64.3465 -> -64 -> -64*2^-0
  rawBias/Si*Sw -0.0115329 /  ( 0.0670743 * 0.000756184 ) = -227.381 -> -227 -> -227*2^-0
  rawBias/Si*Sw 0.002148 /  ( 0.0670743 * 0.000756184 ) = 42.3497 -> 42 -> 42*2^-0
  rawBias/Si*Sw 0.00194639 /  ( 0.0670743 * 0.000756184 ) = 38.3747 -> 38 -> 38*2^-0
  rawBias/Si*Sw -0.0101629 /  ( 0.0670743 * 0.000756184 ) = -200.371 -> -200 -> -200*2^-0
  rawBias/Si*Sw -0.00186599 /  ( 0.0670743 * 0.000756184 ) = -36.7896 -> -37 -> -37*2^-0
  rawBias/Si*Sw 0.0116932 /  ( 0.0670743 * 0.000756184 ) = 230.541 -> 231 -> 231*2^-0
  rawBias/Si*Sw -0.000642768 /  ( 0.0670743 * 0.000756184 ) = -12.6727 -> -13 -> -13*2^-0
  rawBias/Si*Sw -0.00464428 /  ( 0.0670743 * 0.000756184 ) = -91.566 -> -92 -> -92*2^-0
  rawBias/Si*Sw -0.00737856 /  ( 0.0670743 * 0.000756184 ) = -145.475 -> -145 -> -145*2^-0
  rawBias/Si*Sw 0.0024578 /  ( 0.0670743 * 0.000756184 ) = 48.4578 -> 48 -> 48*2^-0
  rawBias/Si*Sw 0.00681272 /  ( 0.0670743 * 0.000756184 ) = 134.319 -> 134 -> 134*2^-0
  rawBias/Si*Sw 0.00416895 /  ( 0.0670743 * 0.000756184 ) = 82.1946 -> 82 -> 82*2^-0
  rawBias/Si*Sw 0.00396947 /  ( 0.0670743 * 0.000756184 ) = 78.2616 -> 78 -> 78*2^-0
  rawBias/Si*Sw -0.0062771 /  ( 0.0670743 * 0.000756184 ) = -123.759 -> -124 -> -124*2^-0
  rawBias/Si*Sw 0.000580598 /  ( 0.0670743 * 0.000756184 ) = 11.447 -> 11 -> 11*2^-0
  rawBias/Si*Sw -0.00101734 /  ( 0.0670743 * 0.000756184 ) = -20.0577 -> -20 -> -20*2^-0
  rawBias/Si*Sw -0.00331908 /  ( 0.0670743 * 0.000756184 ) = -65.4387 -> -65 -> -65*2^-0
  rawBias/Si*Sw -0.00147223 /  ( 0.0670743 * 0.000756184 ) = -29.0263 -> -29 -> -29*2^-0
  rawBias/Si*Sw 0.0143072 /  ( 0.0670743 * 0.000756184 ) = 282.079 -> 282 -> 282*2^-0
  rawBias/Si*Sw 0.00945186 /  ( 0.0670743 * 0.000756184 ) = 186.352 -> 186 -> 186*2^-0
  rawBias/Si*Sw 0.00212467 /  ( 0.0670743 * 0.000756184 ) = 41.8898 -> 42 -> 42*2^-0
  rawBias/Si*Sw 0.00702214 /  ( 0.0670743 * 0.000756184 ) = 138.448 -> 138 -> 138*2^-0
  rawBias/Si*Sw 0.00546126 /  ( 0.0670743 * 0.000756184 ) = 107.674 -> 108 -> 108*2^-0
  rawBias/Si*Sw 0.0012048 /  ( 0.0670743 * 0.000756184 ) = 23.7537 -> 24 -> 24*2^-0
  rawBias/Si*Sw 0.00901155 /  ( 0.0670743 * 0.000756184 ) = 177.671 -> 178 -> 178*2^-0
  rawBias/Si*Sw -0.00802692 /  ( 0.0670743 * 0.000756184 ) = -158.258 -> -158 -> -158*2^-0
  rawBias/Si*Sw 0.0110263 /  ( 0.0670743 * 0.000756184 ) = 217.393 -> 217 -> 217*2^-0
  rawBias/Si*Sw -0.00302939 /  ( 0.0670743 * 0.000756184 ) = -59.7271 -> -60 -> -60*2^-0
  rawBias/Si*Sw -0.000107664 /  ( 0.0670743 * 0.000756184 ) = -2.12269 -> -2 -> -2*2^-0
  rawBias/Si*Sw 0.00674109 /  ( 0.0670743 * 0.000756184 ) = 132.906 -> 133 -> 133*2^-0
  rawBias/Si*Sw -0.00715094 /  ( 0.0670743 * 0.000756184 ) = -140.987 -> -141 -> -141*2^-0
  rawBias/Si*Sw -0.00416368 /  ( 0.0670743 * 0.000756184 ) = -82.0907 -> -82 -> -82*2^-0
  rawBias/Si*Sw -0.00109759 /  ( 0.0670743 * 0.000756184 ) = -21.64 -> -22 -> -22*2^-0
  rawBias/Si*Sw -0.00181006 /  ( 0.0670743 * 0.000756184 ) = -35.6869 -> -36 -> -36*2^-0
  rawBias/Si*Sw 0.00966231 /  ( 0.0670743 * 0.000756184 ) = 190.501 -> 191 -> 191*2^-0
  rawBias/Si*Sw -0.00492121 /  ( 0.0670743 * 0.000756184 ) = -97.0259 -> -97 -> -97*2^-0
  rawBias/Si*Sw 0.00562308 /  ( 0.0670743 * 0.000756184 ) = 110.864 -> 111 -> 111*2^-0
  rawBias/Si*Sw 0.00988432 /  ( 0.0670743 * 0.000756184 ) = 194.878 -> 195 -> 195*2^-0
  rawBias/Si*Sw 0.00530393 /  ( 0.0670743 * 0.000756184 ) = 104.572 -> 105 -> 105*2^-0
  rawBias/Si*Sw -0.000590022 /  ( 0.0670743 * 0.000756184 ) = -11.6328 -> -12 -> -12*2^-0
  rawBias/Si*Sw 0.00142536 /  ( 0.0670743 * 0.000756184 ) = 28.1022 -> 28 -> 28*2^-0
  rawBias/Si*Sw -0.00840012 /  ( 0.0670743 * 0.000756184 ) = -165.616 -> -166 -> -166*2^-0
  rawBias/Si*Sw -0.00277301 /  ( 0.0670743 * 0.000756184 ) = -54.6724 -> -55 -> -55*2^-0
  rawBias/Si*Sw -0.00480831 /  ( 0.0670743 * 0.000756184 ) = -94.8001 -> -95 -> -95*2^-0
  rawBias/Si*Sw 0.00468205 /  ( 0.0670743 * 0.000756184 ) = 92.3108 -> 92 -> 92*2^-0
  rawBias/Si*Sw 0.00824822 /  ( 0.0670743 * 0.000756184 ) = 162.621 -> 163 -> 163*2^-0
  rawBias/Si*Sw 0.0036084 /  ( 0.0670743 * 0.000756184 ) = 71.1428 -> 71 -> 71*2^-0
  rawBias/Si*Sw 0.00174475 /  ( 0.0670743 * 0.000756184 ) = 34.3992 -> 34 -> 34*2^-0
  rawBias/Si*Sw 0.00406262 /  ( 0.0670743 * 0.000756184 ) = 80.0982 -> 80 -> 80*2^-0
  rawBias/Si*Sw 0.00370747 /  ( 0.0670743 * 0.000756184 ) = 73.0961 -> 73 -> 73*2^-0
  rawBias/Si*Sw -0.00549466 /  ( 0.0670743 * 0.000756184 ) = -108.332 -> -108 -> -108*2^-0
  rawBias/Si*Sw 0.00743413 /  ( 0.0670743 * 0.000756184 ) = 146.57 -> 147 -> 147*2^-0
  rawBias/Si*Sw 0.0100927 /  ( 0.0670743 * 0.000756184 ) = 198.987 -> 199 -> 199*2^-0
  rawBias/Si*Sw 0.0071719 /  ( 0.0670743 * 0.000756184 ) = 141.4 -> 141 -> 141*2^-0
  rawBias/Si*Sw -0.00537823 /  ( 0.0670743 * 0.000756184 ) = -106.037 -> -106 -> -106*2^-0
  rawBias/Si*Sw 0.00443552 /  ( 0.0670743 * 0.000756184 ) = 87.4501 -> 87 -> 87*2^-0
  rawBias/Si*Sw 0.00363082 /  ( 0.0670743 * 0.000756184 ) = 71.5848 -> 72 -> 72*2^-0
  rawBias/Si*Sw 0.00574112 /  ( 0.0670743 * 0.000756184 ) = 113.191 -> 113 -> 113*2^-0
  rawBias/Si*Sw -0.0126982 /  ( 0.0670743 * 0.000756184 ) = -250.355 -> -250 -> -250*2^-0
  rawBias/Si*Sw -0.00116731 /  ( 0.0670743 * 0.000756184 ) = -23.0145 -> -23 -> -23*2^-0
  rawBias/Si*Sw -0.00286545 /  ( 0.0670743 * 0.000756184 ) = -56.4949 -> -56 -> -56*2^-0
  rawBias/Si*Sw 0.0194753 /  ( 0.0670743 * 0.000756184 ) = 383.972 -> 384 -> 384*2^-0
  rawBias/Si*Sw 0.0103246 /  ( 0.0670743 * 0.000756184 ) = 203.559 -> 204 -> 204*2^-0
  rawBias/Si*Sw 0.00858375 /  ( 0.0670743 * 0.000756184 ) = 169.236 -> 169 -> 169*2^-0
  rawBias/Si*Sw 0.000202983 /  ( 0.0670743 * 0.000756184 ) = 4.00198 -> 4 -> 4*2^-0
  rawBias/Si*Sw -0.00622996 /  ( 0.0670743 * 0.000756184 ) = -122.829 -> -123 -> -123*2^-0
  rawBias/Si*Sw -0.00117932 /  ( 0.0670743 * 0.000756184 ) = -23.2514 -> -23 -> -23*2^-0
  rawBias/Si*Sw -0.0201595 /  ( 0.0670743 * 0.000756184 ) = -397.463 -> -397 -> -397*2^-0
  rawBias/Si*Sw 0.0038894 /  ( 0.0670743 * 0.000756184 ) = 76.683 -> 77 -> 77*2^-0
  rawBias/Si*Sw 0.0100873 /  ( 0.0670743 * 0.000756184 ) = 198.88 -> 199 -> 199*2^-0
  rawBias/Si*Sw 0.00150142 /  ( 0.0670743 * 0.000756184 ) = 29.6019 -> 30 -> 30*2^-0
  rawBias/Si*Sw -0.0285562 /  ( 0.0561515 * 0.00157656 ) = -322.574 -> -323 -> -323*2^-0
  rawBias/Si*Sw 0.102577 /  ( 0.0561515 * 0.00157656 ) = 1158.73 -> 1159 -> 1159*2^-0
  rawBias/Si*Sw -0.0604402 /  ( 0.0561515 * 0.00157656 ) = -682.74 -> -683 -> -683*2^-0
  rawBias/Si*Sw -0.0525417 /  ( 0.0561515 * 0.00157656 ) = -593.517 -> -594 -> -594*2^-0
  rawBias/Si*Sw -0.00433865 /  ( 0.0561515 * 0.00157656 ) = -49.0098 -> -49 -> -49*2^-0
  rawBias/Si*Sw 0.028257 /  ( 0.0561515 * 0.00157656 ) = 319.194 -> 319 -> 319*2^-0
  rawBias/Si*Sw -0.0173933 /  ( 0.0561515 * 0.00157656 ) = -196.477 -> -196 -> -196*2^-0
  rawBias/Si*Sw 0.0503867 /  ( 0.0561515 * 0.00157656 ) = 569.174 -> 569 -> 569*2^-0
  rawBias/Si*Sw 0.0181165 /  ( 0.0561515 * 0.00157656 ) = 204.646 -> 205 -> 205*2^-0
  rawBias/Si*Sw -0.0360676 /  ( 0.0561515 * 0.00157656 ) = -407.424 -> -407 -> -407*2^-0
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  bias-0 Si * Sw / So = 0.00784518 * 0.00345303 / 0.0137246 = 16557* 2^-23
  bias-1 Si * Sw / So = 0.0137246 * 0.00122683 / 0.0670743 = 16846* 2^-26
  bias-2 Si * Sw / So = 0.0670743 * 0.000756184 / 0.0561515 = 30309* 2^-25
  bias-3 Si * Sw / So = 0.0561515 * 0.00157656 / 0.141637 = 20972* 2^-25
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  translating weights for n-1 bias-dims kcrs = 1,20,1,1 and size= 20
  translating weights for n-3 kernel-dims kcrs = 50,20,5,5 and size= 25000
  translating weights for n-4 bias-dims kcrs = 1,50,1,1 and size= 50
  translating weights for n-6 kernel-dims kcrs = 500,50,4,4 and size= 400000
  translating weights for n-7 bias-dims kcrs = 1,500,1,1 and size= 500
  translating weights for n-10 kernel-dims kcrs = 10,500,1,1 and size= 5000
  translating weights for n-11 bias-dims kcrs = 1,10,1,1 and size= 10
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  (dc-conv-0) CONV_DIRECT
  total b4d needed 1
  total b4w needed 1
  min b4w needed 1
  reserved WMB bank 0
  (dc-conv-0) FI + FW mode. Nothing to split
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  (pdp-0) maxflywidth 128 out width 12
  (pdp-0) maxFlyingWidth >= output_width. No need to do hw/sw PDP splits
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  (dc-conv-1) CONV_DIRECT
  total b4d needed 1
  total b4w needed 1
  min b4w needed 1
  reserved WMB bank 0
  (dc-conv-1) FI + FW mode. Nothing to split
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  (pdp-1) maxflywidth 128 out width 4
  (pdp-1) maxFlyingWidth >= output_width. No need to do hw/sw PDP splits
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  (fc-0) CONV_DIRECT
  total b4d needed 1
  total b4w needed 13
  min b4w needed 1
  reserved WMB bank 0
  (fc-0) FI + FW mode. Nothing to split
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  (fc-1) CONV_DIRECT
  total b4d needed 1
  total b4w needed 1
  min b4w needed 1
  reserved WMB bank 0
  (fc-1) FI + FW mode. Nothing to split
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  printGraph: pree fuseSDPSubEngineOps
  n-0:dc-conv-0/conv1:    (in)e-0[1x4x28x28][tsd-1][tt-1],        (Aux)e-9[20x20x5x1][tsd-0][tt-4],       (out)e-11[1x20x24x24][tsd-10][tt-8], 
  n-1:bias-0/conv1:       (Aux)e-10[1x20x1x1][tsd-2][tt-5],       (in)e-11[1x20x24x24][tsd-10][tt-8],     (out)e-1[1x20x24x24][tsd-11][tt-3], 
  n-2:pdp-0/pool1:        (in)e-1[1x20x24x24][tsd-11][tt-3],      (out)e-2[1x20x12x12][tsd-12][tt-3], 
  n-3:dc-conv-1/conv2:    (in)e-2[1x20x12x12][tsd-12][tt-3],      (Aux)e-12[50x20x5x5][tsd-3][tt-4],      (out)e-14[1x50x8x8][tsd-13][tt-8], 
  n-4:bias-1/conv2:       (Aux)e-13[1x50x1x1][tsd-4][tt-5],       (in)e-14[1x50x8x8][tsd-13][tt-8],       (out)e-3[1x50x8x8][tsd-14][tt-3], 
  n-5:pdp-1/pool2:        (in)e-3[1x50x8x8][tsd-14][tt-3],        (out)e-4[1x50x4x4][tsd-15][tt-3], 
  n-6:fc-0/ip1:   (in)e-4[1x50x4x4][tsd-15][tt-3],        (Aux)e-15[500x50x4x4][tsd-5][tt-4],     (out)e-17[1x500x1x1][tsd-16][tt-8], 
  n-7:bias-2/ip1: (Aux)e-16[1x500x1x1][tsd-6][tt-5],      (in)e-17[1x500x1x1][tsd-16][tt-8],      (out)e-6[1x500x1x1][tsd-19][tt-3], 
  n-10:fc-1/ip2:  (in)e-6[1x500x1x1][tsd-19][tt-3],       (Aux)e-20[10x500x1x1][tsd-8][tt-4],     (out)e-22[1x10x1x1][tsd-20][tt-8], 
  n-11:bias-3/ip2:        (Aux)e-21[1x10x1x1][tsd-9][tt-5],       (in)e-22[1x10x1x1][tsd-20][tt-8],       (out)e-7[1x10x1x1][tsd-21][tt-3], 
  n-12:cpu-sm-0/prob:     (in)e-7[1x10x1x1][tsd-21][tt-3],        (out)e-8[1x10x1x1][tsd-22][tt-2], 
  e-9 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-0 edge already has set surface format NVDLA_IMG_A8B8G8R8
  e-10 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-12 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-13 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-15 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-16 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-20 edge already has set surface format NVDLA_WEIGHT_DC_INT8
  e-21 edge already has set surface format NVDLA_BIAS_DATA_INT16
  e-11 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-1 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-2 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-14 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-3 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-4 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-17 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-6 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-22 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-7 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  e-8 edge already has set surface format NVDLA_FEATURE_DATA_INT8
  tb-0 for tsd-0 for e-9 with NVDLA_WEIGHT_DC_INT8
  tb-1 for tsd-1 for e-0 with NVDLA_IMG_A8B8G8R8
  tb-2 for tsd-2 for e-10 with NVDLA_BIAS_DATA_INT16
  tb-3 for tsd-3 for e-12 with NVDLA_WEIGHT_DC_INT8
  tb-4 for tsd-4 for e-13 with NVDLA_BIAS_DATA_INT16
  tb-5 for tsd-5 for e-15 with NVDLA_WEIGHT_DC_INT8
  tb-6 for tsd-6 for e-16 with NVDLA_BIAS_DATA_INT16
  tb-8 for tsd-8 for e-20 with NVDLA_WEIGHT_DC_INT8
  tb-9 for tsd-9 for e-21 with NVDLA_BIAS_DATA_INT16
  tb-10 for tsd-10 for e-11 with NVDLA_FEATURE_DATA_INT8
  tb-11 for tsd-11 for e-1 with NVDLA_FEATURE_DATA_INT8
  tb-12 for tsd-12 for e-2 with NVDLA_FEATURE_DATA_INT8
  tb-13 for tsd-13 for e-14 with NVDLA_FEATURE_DATA_INT8
  tb-14 for tsd-14 for e-3 with NVDLA_FEATURE_DATA_INT8
  tb-15 for tsd-15 for e-4 with NVDLA_FEATURE_DATA_INT8
  tb-16 for tsd-16 for e-17 with NVDLA_FEATURE_DATA_INT8
  tb-19 for tsd-19 for e-6 with NVDLA_FEATURE_DATA_INT8
  tb-20 for tsd-20 for e-22 with NVDLA_FEATURE_DATA_INT8
  tb-21 for tsd-21 for e-7 with NVDLA_FEATURE_DATA_INT8
  tb-22 for tsd-22 for e-8 with NVDLA_FEATURE_DATA_INT8
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  printGraph: post fuseSDPSubEngineOps
  n-0:dc-conv-0/conv1:    (in)e-0[1x4x28x28][tsd-1][tt-1],        (Aux)e-9[20x20x5x1][tsd-0][tt-4],       (out)e-11[1x20x24x24][tsd-10][tt-8], 
  n-1:bias-0/conv1:       (Aux)e-10[1x20x1x1][tsd-2][tt-5],       (in)e-11[1x20x24x24][tsd-10][tt-8],     (out)e-1[1x20x24x24][tsd-11][tt-3], 
  n-2:pdp-0/pool1:        (in)e-1[1x20x24x24][tsd-11][tt-3],      (out)e-2[1x20x12x12][tsd-12][tt-3], 
  n-3:dc-conv-1/conv2:    (in)e-2[1x20x12x12][tsd-12][tt-3],      (Aux)e-12[50x20x5x5][tsd-3][tt-4],      (out)e-14[1x50x8x8][tsd-13][tt-8], 
  n-4:bias-1/conv2:       (Aux)e-13[1x50x1x1][tsd-4][tt-5],       (in)e-14[1x50x8x8][tsd-13][tt-8],       (out)e-3[1x50x8x8][tsd-14][tt-3], 
  n-5:pdp-1/pool2:        (in)e-3[1x50x8x8][tsd-14][tt-3],        (out)e-4[1x50x4x4][tsd-15][tt-3], 
  n-6:fc-0/ip1:   (in)e-4[1x50x4x4][tsd-15][tt-3],        (Aux)e-15[500x50x4x4][tsd-5][tt-4],     (out)e-17[1x500x1x1][tsd-16][tt-8], 
  n-7:bias-2/ip1: (Aux)e-16[1x500x1x1][tsd-6][tt-5],      (in)e-17[1x500x1x1][tsd-16][tt-8],      (out)e-6[1x500x1x1][tsd-19][tt-3], 
  n-10:fc-1/ip2:  (in)e-6[1x500x1x1][tsd-19][tt-3],       (Aux)e-20[10x500x1x1][tsd-8][tt-4],     (out)e-22[1x10x1x1][tsd-20][tt-8], 
  n-11:bias-3/ip2:        (Aux)e-21[1x10x1x1][tsd-9][tt-5],       (in)e-22[1x10x1x1][tsd-20][tt-8],       (out)e-7[1x10x1x1][tsd-21][tt-3], 
  n-12:cpu-sm-0/prob:     (in)e-7[1x10x1x1][tsd-21][tt-3],        (out)e-8[1x10x1x1][tsd-22][tt-2], 
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-8 bindable=1
  ::Edge edge=e-9 bindable=0
  ::Surface surface=tsd-0: bindable=0
  ::Edge edge=e-0 bindable=1
  ::Edge edge=e-10 bindable=0
  ::Surface surface=tsd-2: bindable=0
  ::Edge edge=e-12 bindable=0
  ::Surface surface=tsd-3: bindable=0
  ::Edge edge=e-13 bindable=0
  ::Surface surface=tsd-4: bindable=0
  ::Edge edge=e-15 bindable=0
  ::Surface surface=tsd-5: bindable=0
  ::Edge edge=e-16 bindable=0
  ::Surface surface=tsd-6: bindable=0
  ::Edge edge=e-20 bindable=0
  ::Surface surface=tsd-8: bindable=0
  ::Edge edge=e-21 bindable=0
  ::Surface surface=tsd-9: bindable=0
  ::Edge edge=e-11 bindable=0
  ::Surface surface=tsd-10: bindable=0
  ::Edge edge=e-1 bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Edge edge=e-2 bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Edge edge=e-14 bindable=0
  ::Surface surface=tsd-13: bindable=0
  ::Edge edge=e-3 bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Edge edge=e-4 bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Edge edge=e-17 bindable=0
  ::Surface surface=tsd-16: bindable=0
  ::Edge edge=e-6 bindable=0
  ::Surface surface=tsd-19: bindable=0
  ::Edge edge=e-22 bindable=0
  ::Surface surface=tsd-20: bindable=0
  ::Edge edge=e-7 bindable=0
  ::Surface surface=tsd-21: bindable=0
  ::Edge edge=e-8 bindable=1
  annid=0 node=dc-conv-0.B0 deps=1
  producer: [, , , , , , , , , , ]
  consumer: [, dc-conv-1(annId:3):OP_PROGRAMMED, , bias-0(annId:1):OP_PROGRAMMED, , , , , , , ]
  annid=1 node=bias-0.B0 deps=1
  producer: [, dc-conv-0(annId:0):OP_PROGRAMMED, , , , , , , , , ]
  consumer: [, , , bias-1(annId:4):OP_PROGRAMMED, pdp-0(annId:2):OP_COMPLETED, , , , , , ]
  annid=2 node=pdp-0.B0 deps=1
  producer: [, , , bias-0(annId:1):OP_COMPLETED, , , , , , , ]
  consumer: [, dc-conv-1(annId:3):OP_COMPLETED, , , pdp-1(annId:5):OP_PROGRAMMED, , , , , , ]
  annid=3 node=dc-conv-1.B0 deps=3
  producer: [, dc-conv-0(annId:0):OP_PROGRAMMED, , , pdp-0(annId:2):OP_COMPLETED, , , , , , ]
  consumer: [, fc-0(annId:6):OP_PROGRAMMED, , bias-1(annId:4):OP_PROGRAMMED, , , , , , , ]
  annid=4 node=bias-1.B0 deps=2
  producer: [, dc-conv-1(annId:3):OP_PROGRAMMED, , bias-0(annId:1):OP_PROGRAMMED, , , , , , , ]
  consumer: [, , , bias-2(annId:7):OP_PROGRAMMED, pdp-1(annId:5):OP_COMPLETED, , , , , , ]
  annid=5 node=pdp-1.B0 deps=2
  producer: [, , , bias-1(annId:4):OP_COMPLETED, pdp-0(annId:2):OP_PROGRAMMED, , , , , , ]
  consumer: [, fc-0(annId:6):OP_COMPLETED, , , , , , , , , ]
  annid=6 node=fc-0.B0 deps=3
  producer: [, dc-conv-1(annId:3):OP_PROGRAMMED, , , pdp-1(annId:5):OP_COMPLETED, , , , , , ]
  consumer: [, fc-1(annId:8):OP_PROGRAMMED, , bias-2(annId:7):OP_PROGRAMMED, , , , , , , ]
  annid=7 node=bias-2.B0 deps=2
  producer: [, fc-0(annId:6):OP_PROGRAMMED, , bias-1(annId:4):OP_PROGRAMMED, , , , , , , ]
  consumer: [, fc-1(annId:8):OP_COMPLETED, , bias-3(annId:9):OP_PROGRAMMED, , , , , , , ]
  annid=8 node=fc-1.B0 deps=3
  producer: [, fc-0(annId:6):OP_PROGRAMMED, , bias-2(annId:7):OP_COMPLETED, , , , , , , ]
  consumer: [, , , bias-3(annId:9):OP_PROGRAMMED, , , , , , , ]
  annid=9 node=bias-3.B0 deps=2
  producer: [, fc-1(annId:8):OP_PROGRAMMED, , bias-2(annId:7):OP_PROGRAMMED, , , , , , , ]
  consumer: [cpu-sm-0(annId:0):OP_COMPLETED, , , , , , , , , , ]
  annid=0 node=cpu-sm-0.B0 deps=1
  producer: [, , , bias-3(annId:9):OP_COMPLETED, , , , , , , ]
  consumer: [, , , , , , , , , , ]
  beginning resolveMemory phase
  begin memory resolver pooling=1 reuse=1 greedy_eviction=1
  local cvsram size=0 local sdram size=1073741824 global sdram size=536870912
  node=dc-conv-0 anno=0.0 in=[tsd-1] aux=[tsd-0] io=[] out=[tsd-10]
  tsd=tsd-10 (stream tensor)
  ::Surface surface=tsd-1: bindable=1
  ::Buffer buffer=tb-1 surface=tsd-1 bindable=1
  ::Surface surface=tsd-1: bindable=1
  resolve placement/alloc for tsd=tsd-1/tb-1 aux=0 pooling=0
  placed tsd-1/tb-1 batch-0 inside DRAM @0
  ::Surface surface=tsd-0: bindable=0
  ::Buffer buffer=tb-0 surface=tsd-0 bindable=0
  ::Surface surface=tsd-0: bindable=0
  resolve placement/alloc for tsd=tsd-0/tb-0 aux=1 pooling=1
  GLOBAL_DRAM_POOL alloc tb-0 @0 +2048 loc=1
  [MEMTOOL] t = 0     dc-conv-0's     AUX     tb-0-B0 ALLOC 
  placed tb-0 batch-0 inside GLOBAL_DRAM_POOL@0
  tsd=tsd-0 set content pooled=1
  done node=dc-conv-0 rc=0
  node=bias-0 anno=0.1 in=[tsd-10] aux=[tsd-2] io=[] out=[tsd-11]
  tsd=tsd-10 (stream tensor)
  ::Surface surface=tsd-2: bindable=0
  ::Buffer buffer=tb-2 surface=tsd-2 bindable=0
  ::Surface surface=tsd-2: bindable=0
  resolve placement/alloc for tsd=tsd-2/tb-2 aux=1 pooling=1
  GLOBAL_DRAM_POOL alloc tb-2 @4096 +40 loc=1
  [MEMTOOL] t = 1     bias-0's        AUX     tb-2-B0 ALLOC 
  placed tb-2 batch-0 inside GLOBAL_DRAM_POOL@4096
  tsd=tsd-2 set content pooled=1
  ::Surface surface=tsd-11: bindable=0
  ::Buffer buffer=tb-11 surface=tsd-11 bindable=0
  ::Surface surface=tsd-11: bindable=0
  resolve placement/alloc for tsd=tsd-11/tb-11 aux=0 pooling=1
  LOCAL_DRAM_POOL alloc tb-11 @0 +18432 loc=1
  [MEMTOOL] t = 2     bias-0's        OUTPUT  tb-11-B0        ALLOC 
  placed tb-11 batch-0 inside LOCAL_DRAM_POOL@0
  done node=bias-0 rc=0
  node=pdp-0 anno=0.2 in=[tsd-11] aux=[] io=[] out=[tsd-12]
  ::Surface surface=tsd-12: bindable=0
  ::Buffer buffer=tb-12 surface=tsd-12 bindable=0
  ::Surface surface=tsd-12: bindable=0
  resolve placement/alloc for tsd=tsd-12/tb-12 aux=0 pooling=1
  LOCAL_DRAM_POOL alloc tb-12 @32768 +4608 loc=1
  [MEMTOOL] t = 3     pdp-0's OUTPUT  tb-12-B0        ALLOC 
  placed tb-12 batch-0 inside LOCAL_DRAM_POOL@32768
  done node=pdp-0 rc=0
  node=dc-conv-1 anno=0.3 in=[tsd-12] aux=[tsd-3] io=[] out=[tsd-13]
  tsd=tsd-13 (stream tensor)
  deallocating tsd-11/tb-11@0x7fd126829010
  LOCAL_DRAM_POOL dealloc tb-11 @ 0 @ 140536270983184
  [MEMTOOL] t = 4     tb-11-B0        DEALLOC 
  ::Surface surface=tsd-3: bindable=0
  ::Buffer buffer=tb-3 surface=tsd-3 bindable=0
  ::Surface surface=tsd-3: bindable=0
  resolve placement/alloc for tsd=tsd-3/tb-3 aux=1 pooling=1
  GLOBAL_DRAM_POOL alloc tb-3 @32768 +25088 loc=1
  [MEMTOOL] t = 5     dc-conv-1's     AUX     tb-3-B0 ALLOC 
  placed tb-3 batch-0 inside GLOBAL_DRAM_POOL@32768
  tsd=tsd-3 set content pooled=1
  done node=dc-conv-1 rc=0
  node=bias-1 anno=0.4 in=[tsd-13] aux=[tsd-4] io=[] out=[tsd-14]
  tsd=tsd-13 (stream tensor)
  ::Surface surface=tsd-4: bindable=0
  ::Buffer buffer=tb-4 surface=tsd-4 bindable=0
  ::Surface surface=tsd-4: bindable=0
  resolve placement/alloc for tsd=tsd-4/tb-4 aux=1 pooling=1
  GLOBAL_DRAM_POOL alloc tb-4 @8192 +100 loc=1
  [MEMTOOL] t = 6     bias-1's        AUX     tb-4-B0 ALLOC 
  placed tb-4 batch-0 inside GLOBAL_DRAM_POOL@8192
  tsd=tsd-4 set content pooled=1
  ::Surface surface=tsd-14: bindable=0
  ::Buffer buffer=tb-14 surface=tsd-14 bindable=0
  ::Surface surface=tsd-14: bindable=0
  resolve placement/alloc for tsd=tsd-14/tb-14 aux=0 pooling=1
  LOCAL_DRAM_POOL alloc tb-14 @40960 +4096 loc=1
  [MEMTOOL] t = 7     bias-1's        OUTPUT  tb-14-B0        ALLOC 
  placed tb-14 batch-0 inside LOCAL_DRAM_POOL@40960
  done node=bias-1 rc=0
  node=pdp-1 anno=0.5 in=[tsd-14] aux=[] io=[] out=[tsd-15]
  deallocating tsd-12/tb-12@0x7fd126831010
  LOCAL_DRAM_POOL dealloc tb-12 @ 32768 @ 140536271015952
  [MEMTOOL] t = 8     tb-12-B0        DEALLOC 
  ::Surface surface=tsd-15: bindable=0
  ::Buffer buffer=tb-15 surface=tsd-15 bindable=0
  ::Surface surface=tsd-15: bindable=0
  resolve placement/alloc for tsd=tsd-15/tb-15 aux=0 pooling=1
  LOCAL_DRAM_POOL alloc tb-15 @45056 +1024 loc=1
  [MEMTOOL] t = 9     pdp-1's OUTPUT  tb-15-B0        ALLOC 
  placed tb-15 batch-0 inside LOCAL_DRAM_POOL@45056
  done node=pdp-1 rc=0
  node=fc-0 anno=0.6 in=[tsd-15] aux=[tsd-5] io=[] out=[tsd-16]
  tsd=tsd-16 (stream tensor)
  deallocating tsd-14/tb-14@0x7fd126833010
  LOCAL_DRAM_POOL dealloc tb-14 @ 40960 @ 140536271024144
  [MEMTOOL] t = 10    tb-14-B0        DEALLOC 
  ::Surface surface=tsd-5: bindable=0
  ::Buffer buffer=tb-5 surface=tsd-5 bindable=0
  ::Surface surface=tsd-5: bindable=0
  resolve placement/alloc for tsd=tsd-5/tb-5 aux=1 pooling=1
  GLOBAL_DRAM_POOL alloc tb-5 @524288 +400000 loc=1
  [MEMTOOL] t = 11    fc-0's  AUX     tb-5-B0 ALLOC 
  placed tb-5 batch-0 inside GLOBAL_DRAM_POOL@524288
  tsd=tsd-5 set content pooled=1
  done node=fc-0 rc=0
  node=bias-2 anno=0.7 in=[tsd-16] aux=[tsd-6] io=[] out=[tsd-19]
  tsd=tsd-16 (stream tensor)
  ::Surface surface=tsd-6: bindable=0
  ::Buffer buffer=tb-6 surface=tsd-6 bindable=0
  ::Surface surface=tsd-6: bindable=0
  resolve placement/alloc for tsd=tsd-6/tb-6 aux=1 pooling=1
  GLOBAL_DRAM_POOL alloc tb-6 @12288 +1000 loc=1
  [MEMTOOL] t = 12    bias-2's        AUX     tb-6-B0 ALLOC 
  placed tb-6 batch-0 inside GLOBAL_DRAM_POOL@12288
  tsd=tsd-6 set content pooled=1
  ::Surface surface=tsd-19: bindable=0
  ::Buffer buffer=tb-19 surface=tsd-19 bindable=0
  ::Surface surface=tsd-19: bindable=0
  resolve placement/alloc for tsd=tsd-19/tb-19 aux=0 pooling=1
  LOCAL_DRAM_POOL alloc tb-19 @40960 +512 loc=1
  [MEMTOOL] t = 13    bias-2's        OUTPUT  tb-19-B0        ALLOC 
  placed tb-19 batch-0 inside LOCAL_DRAM_POOL@40960
  done node=bias-2 rc=0
  node=fc-1 anno=0.8 in=[tsd-19] aux=[tsd-8] io=[] out=[tsd-20]
  tsd=tsd-20 (stream tensor)
  deallocating tsd-15/tb-15@0x7fd126834010
  LOCAL_DRAM_POOL dealloc tb-15 @ 45056 @ 140536271028240
  [MEMTOOL] t = 14    tb-15-B0        DEALLOC 
  ::Surface surface=tsd-8: bindable=0
  ::Buffer buffer=tb-8 surface=tsd-8 bindable=0
  ::Surface surface=tsd-8: bindable=0
  resolve placement/alloc for tsd=tsd-8/tb-8 aux=1 pooling=1
  GLOBAL_DRAM_POOL alloc tb-8 @16384 +5120 loc=1
  [MEMTOOL] t = 15    fc-1's  AUX     tb-8-B0 ALLOC 
  placed tb-8 batch-0 inside GLOBAL_DRAM_POOL@16384
  tsd=tsd-8 set content pooled=1
  done node=fc-1 rc=0
  node=bias-3 anno=0.9 in=[tsd-20] aux=[tsd-9] io=[] out=[tsd-21]
  tsd=tsd-20 (stream tensor)
  ::Surface surface=tsd-9: bindable=0
  ::Buffer buffer=tb-9 surface=tsd-9 bindable=0
  ::Surface surface=tsd-9: bindable=0
  resolve placement/alloc for tsd=tsd-9/tb-9 aux=1 pooling=1
  GLOBAL_DRAM_POOL alloc tb-9 @24576 +20 loc=1
  [MEMTOOL] t = 16    bias-3's        AUX     tb-9-B0 ALLOC 
  placed tb-9 batch-0 inside GLOBAL_DRAM_POOL@24576
  tsd=tsd-9 set content pooled=1
  ::Surface surface=tsd-21: bindable=0
  ::Buffer buffer=tb-21 surface=tsd-21 bindable=0
  ::Surface surface=tsd-21: bindable=0
  resolve placement/alloc for tsd=tsd-21/tb-21 aux=0 pooling=1
  LOCAL_DRAM_POOL alloc tb-21 @45056 +32 loc=1
  [MEMTOOL] t = 17    bias-3's        OUTPUT  tb-21-B0        ALLOC 
  placed tb-21 batch-0 inside LOCAL_DRAM_POOL@45056
  done node=bias-3 rc=0
  node=cpu-sm-0 anno=1.0 in=[tsd-21] aux=[] io=[] out=[tsd-22]
  ::Surface surface=tsd-22: bindable=1
  ::Buffer buffer=tb-22 surface=tsd-22 bindable=1
  ::Surface surface=tsd-22: bindable=1
  resolve placement/alloc for tsd=tsd-22/tb-22 aux=0 pooling=0
  placed tsd-22/tb-22 batch-0 inside DRAM @0
  done node=cpu-sm-0 rc=0
  end memory resolver
  printGraph: Final
  n-0:dc-conv-0/conv1:    (in)e-0[1x4x28x28][tsd-1][tt-1],        (Aux)e-9[20x20x5x1][tsd-0][tt-4],       (out)e-11[1x20x24x24][tsd-10][tt-8], 
  n-1:bias-0/conv1:       (Aux)e-10[1x20x1x1][tsd-2][tt-5],       (in)e-11[1x20x24x24][tsd-10][tt-8],     (out)e-1[1x20x24x24][tsd-11][tt-3], 
  n-2:pdp-0/pool1:        (in)e-1[1x20x24x24][tsd-11][tt-3],      (out)e-2[1x20x12x12][tsd-12][tt-3], 
  n-3:dc-conv-1/conv2:    (in)e-2[1x20x12x12][tsd-12][tt-3],      (Aux)e-12[50x20x5x5][tsd-3][tt-4],      (out)e-14[1x50x8x8][tsd-13][tt-8], 
  n-4:bias-1/conv2:       (Aux)e-13[1x50x1x1][tsd-4][tt-5],       (in)e-14[1x50x8x8][tsd-13][tt-8],       (out)e-3[1x50x8x8][tsd-14][tt-3], 
  n-5:pdp-1/pool2:        (in)e-3[1x50x8x8][tsd-14][tt-3],        (out)e-4[1x50x4x4][tsd-15][tt-3], 
  n-6:fc-0/ip1:   (in)e-4[1x50x4x4][tsd-15][tt-3],        (Aux)e-15[500x50x4x4][tsd-5][tt-4],     (out)e-17[1x500x1x1][tsd-16][tt-8], 
  n-7:bias-2/ip1: (Aux)e-16[1x500x1x1][tsd-6][tt-5],      (in)e-17[1x500x1x1][tsd-16][tt-8],      (out)e-6[1x500x1x1][tsd-19][tt-3], 
  n-10:fc-1/ip2:  (in)e-6[1x500x1x1][tsd-19][tt-3],       (Aux)e-20[10x500x1x1][tsd-8][tt-4],     (out)e-22[1x10x1x1][tsd-20][tt-8], 
  n-11:bias-3/ip2:        (Aux)e-21[1x10x1x1][tsd-9][tt-5],       (in)e-22[1x10x1x1][tsd-20][tt-8],       (out)e-7[1x10x1x1][tsd-21][tt-3], 
  n-12:cpu-sm-0/prob:     (in)e-7[1x10x1x1][tsd-21][tt-3],        (out)e-8[1x10x1x1][tsd-22][tt-2], 
  compiler targeting dla (fw) interface 0.12
  compiler targeting emu (cpu) interface 0.0
  (Pool) Memory list entry=1 size=925696 used=924288 domain=0 flags=3
  content: tb-0 @ 0
  content: tb-2 @ 4096
  content: tb-3 @ 32768
  content: tb-4 @ 8192
  content: tb-5 @ 524288
  content: tb-6 @ 12288
  content: tb-8 @ 16384
  content: tb-9 @ 24576

  (Pool) Memory list entry=2 size=49152 used=46080 domain=0 flags=1

  ::Surface surface=tsd-0: bindable=0
  ::Buffer buffer=tb-0 surface=tsd-0 bindable=0
  ::Surface surface=tsd-0: bindable=0
  ::Surface surface=tsd-0: bindable=0
  ::Buffer buffer=tb-0 surface=tsd-0 bindable=0
  ::Surface surface=tsd-0: bindable=0
  ::Surface surface=tsd-1: bindable=1
  ::Buffer buffer=tb-1 surface=tsd-1 bindable=1
  ::Surface surface=tsd-1: bindable=1
  ::Surface surface=tsd-1: bindable=1
  ::Buffer buffer=tb-1 surface=tsd-1 bindable=1
  ::Surface surface=tsd-1: bindable=1
  ::Surface surface=tsd-1: bindable=1
  ::Buffer boundSurface(i=0) -> tsd-1
  create tensor desc precision=1 category=1 sf=12
  name         : data
  n,c,h,w      : 1,4,28,28
  data format  : 2
  data type    : 4
  data category: 0
  pixel format : 12
  pixel mapping: 0
  strides  : 1 128 0 00 0 0 0
  ::Buffer bindId(buffer=tb-1) [                          ::Surface surface=tsd-1: bindable=1
  tsd-1 bind_id=1
  ::Surface surface=tsd-1: bindable=1
  ::Surface bindId(tsd-1, 0) -> 0
  ]
  (Bindable)(Buffer) Memory list entry for tbd=tb-1:0 : 0 size=3584 domain=0 flags=5
  ::Surface surface=tsd-10: bindable=0
  ::Buffer buffer=tb-10 surface=tsd-10 bindable=0
  ::Surface surface=tsd-10: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Buffer buffer=tb-11 surface=tsd-11 bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Buffer buffer=tb-11 surface=tsd-11 bindable=0
  ::Surface surface=tsd-11: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Buffer buffer=tb-12 surface=tsd-12 bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Buffer buffer=tb-12 surface=tsd-12 bindable=0
  ::Surface surface=tsd-12: bindable=0
  ::Surface surface=tsd-13: bindable=0
  ::Buffer buffer=tb-13 surface=tsd-13 bindable=0
  ::Surface surface=tsd-13: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Buffer buffer=tb-14 surface=tsd-14 bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Buffer buffer=tb-14 surface=tsd-14 bindable=0
  ::Surface surface=tsd-14: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Buffer buffer=tb-15 surface=tsd-15 bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Buffer buffer=tb-15 surface=tsd-15 bindable=0
  ::Surface surface=tsd-15: bindable=0
  ::Surface surface=tsd-16: bindable=0
  ::Buffer buffer=tb-16 surface=tsd-16 bindable=0
  ::Surface surface=tsd-16: bindable=0
  ::Surface surface=tsd-19: bindable=0
  ::Buffer buffer=tb-19 surface=tsd-19 bindable=0
  ::Surface surface=tsd-19: bindable=0
  ::Surface surface=tsd-19: bindable=0
  ::Buffer buffer=tb-19 surface=tsd-19 bindable=0
  ::Surface surface=tsd-19: bindable=0
  ::Surface surface=tsd-2: bindable=0
  ::Buffer buffer=tb-2 surface=tsd-2 bindable=0
  ::Surface surface=tsd-2: bindable=0
  ::Surface surface=tsd-2: bindable=0
  ::Buffer buffer=tb-2 surface=tsd-2 bindable=0
  ::Surface surface=tsd-2: bindable=0
  ::Surface surface=tsd-20: bindable=0
  ::Buffer buffer=tb-20 surface=tsd-20 bindable=0
  ::Surface surface=tsd-20: bindable=0
  ::Surface surface=tsd-21: bindable=0
  ::Buffer buffer=tb-21 surface=tsd-21 bindable=0
  ::Surface surface=tsd-21: bindable=0
  ::Surface surface=tsd-21: bindable=0
  ::Buffer buffer=tb-21 surface=tsd-21 bindable=0
  ::Surface surface=tsd-21: bindable=0
  ::Surface surface=tsd-22: bindable=1
  ::Buffer buffer=tb-22 surface=tsd-22 bindable=1
  ::Surface surface=tsd-22: bindable=1
  ::Surface surface=tsd-22: bindable=1
  ::Buffer buffer=tb-22 surface=tsd-22 bindable=1
  ::Surface surface=tsd-22: bindable=1
  ::Surface surface=tsd-22: bindable=1
  ::Buffer boundSurface(i=0) -> tsd-22
  create tensor desc precision=1 category=3 sf=63
  name         : prob
  n,c,h,w      : 1,10,1,1
  data format  : 3
  data type    : 4
  data category: 2
  pixel format : 36
  pixel mapping: 0
  strides  : 1 32 32 00 0 0 0
  ::Buffer bindId(buffer=tb-22) [                         ::Surface surface=tsd-22: bindable=1
  tsd-22 bind_id=1
  ::Surface surface=tsd-22: bindable=1
  ::Surface bindId(tsd-22, 1) -> 0
  ]
  (Bindable)(Buffer) Memory list entry for tbd=tb-22:0 : 0 size=32 domain=0 flags=9
  ::Surface surface=tsd-3: bindable=0
  ::Buffer buffer=tb-3 surface=tsd-3 bindable=0
  ::Surface surface=tsd-3: bindable=0
  ::Surface surface=tsd-3: bindable=0
  ::Buffer buffer=tb-3 surface=tsd-3 bindable=0
  ::Surface surface=tsd-3: bindable=0
  ::Surface surface=tsd-4: bindable=0
  ::Buffer buffer=tb-4 surface=tsd-4 bindable=0
  ::Surface surface=tsd-4: bindable=0
  ::Surface surface=tsd-4: bindable=0
  ::Buffer buffer=tb-4 surface=tsd-4 bindable=0
  ::Surface surface=tsd-4: bindable=0
  ::Surface surface=tsd-5: bindable=0
  ::Buffer buffer=tb-5 surface=tsd-5 bindable=0
  ::Surface surface=tsd-5: bindable=0
  ::Surface surface=tsd-5: bindable=0
  ::Buffer buffer=tb-5 surface=tsd-5 bindable=0
  ::Surface surface=tsd-5: bindable=0
  ::Surface surface=tsd-6: bindable=0
  ::Buffer buffer=tb-6 surface=tsd-6 bindable=0
  ::Surface surface=tsd-6: bindable=0
  ::Surface surface=tsd-6: bindable=0
  ::Buffer buffer=tb-6 surface=tsd-6 bindable=0
  ::Surface surface=tsd-6: bindable=0
  ::Surface surface=tsd-8: bindable=0
  ::Buffer buffer=tb-8 surface=tsd-8 bindable=0
  ::Surface surface=tsd-8: bindable=0
  ::Surface surface=tsd-8: bindable=0
  ::Buffer buffer=tb-8 surface=tsd-8 bindable=0
  ::Surface surface=tsd-8: bindable=0
  ::Surface surface=tsd-9: bindable=0
  ::Buffer buffer=tb-9 surface=tsd-9 bindable=0
  ::Surface surface=tsd-9: bindable=0
  ::Surface surface=tsd-9: bindable=0
  ::Buffer buffer=tb-9 surface=tsd-9 bindable=0
  ::Surface surface=tsd-9: bindable=0
  set symbol content name=tb-0 size=2048
  (Surface) Address list entry for tsd=tsd-0/tb-0:0 -> 1 offset=0 size=2048
  (Surface) Address list entry for tsd=tsd-1/tb-1:0 -> 3 offset=0 size=3584
  (Surface) Address list entry for tsd=tsd-11/tb-11:0 -> 2 offset=0 size=18432
  (Surface) Address list entry for tsd=tsd-12/tb-12:0 -> 2 offset=32768 size=4608
  (Surface) Address list entry for tsd=tsd-14/tb-14:0 -> 2 offset=40960 size=4096
  (Surface) Address list entry for tsd=tsd-15/tb-15:0 -> 2 offset=45056 size=1024
  (Surface) Address list entry for tsd=tsd-19/tb-19:0 -> 2 offset=40960 size=512
  set symbol content name=tb-2 size=40
  (Surface) Address list entry for tsd=tsd-2/tb-2:0 -> 1 offset=4096 size=40
  (Surface) Address list entry for tsd=tsd-21/tb-21:0 -> 2 offset=45056 size=32
  (Surface) Address list entry for tsd=tsd-22/tb-22:0 -> 4 offset=0 size=32
  set symbol content name=tb-3 size=25088
  (Surface) Address list entry for tsd=tsd-3/tb-3:0 -> 1 offset=32768 size=25088
  set symbol content name=tb-4 size=100
  (Surface) Address list entry for tsd=tsd-4/tb-4:0 -> 1 offset=8192 size=100
  set symbol content name=tb-5 size=400000
  (Surface) Address list entry for tsd=tsd-5/tb-5:0 -> 1 offset=524288 size=400000
  set symbol content name=tb-6 size=1000
  (Surface) Address list entry for tsd=tsd-6/tb-6:0 -> 1 offset=12288 size=1000
  set symbol content name=tb-8 size=5120
  (Surface) Address list entry for tsd=tsd-8/tb-8:0 -> 1 offset=16384 size=5120
  set symbol content name=tb-9 size=20
  (Surface) Address list entry for tsd=tsd-9/tb-9:0 -> 1 offset=24576 size=20
  emit discovered 2 tasks
  the initial mem list size is  4 entries
  the initial addr list size is 4 entries
  task_id=0 has 10 op slots and 1 batches 
  address list task context at [5, 11)
  data cube access by tsd:batch=tsd-1:0 id[offs]=3
  ::Surface surface=tsd-1: bindable=1
  data cube access by tsd:batch=tsd-0:0 id[offs]=1
  ::Surface surface=tsd-0: bindable=0
  Convolution node @ op_slot = 0 batch_id = 0
  src data loc: 0
  dst data loc: 2
  post y extension: 1
  in_precision 0
  out_precision 0
  pad_val 0
  conv mode 0
  data_reuse 0
  weight_reuse 0
  skip_data_rls 0
  skip_wt_rls 0
  eps 1
  fetch_grain 1
  data_format 12
  pixel_mapping 0
  batch 1
  weight_format 0
  b4d 1
  b4w 1
  batch_stride 0
  release 28
  post_extension 1
  pixel_override 1
  mean_format 0
  stride-x 1
  stride-y 1
  pad-left 0
  pad-top 0
  pad-right 0
  pad-bottom 0
  dilationx-x 1
  dilation-y 1
  pra_truncate 0
  inputwidthcsc 28
  inputheightcsc 28
  inputchannelcsc 4
  kernelwidthcsc 1
  kernelheightcsc 5
  kernelchannelcsc 20
  inputwidthcmac 24
  inputheightcmac 24
  bytesperkernel 100
  offsetU 0
  dependencyCount 1
  src tsd:tsd-1
  src addr=3
  src size 3584
  src width 28
  src height 28
  src channel 4
  src linestride 128
  src surfstride 0
  dst tsd:tsd-10
  dst addr=-1
  dst size 18432
  dst width 24
  dst height 24
  dst channel 20
  dst linestride 768
  dst surfstride 18432
  wt  tsd:tsd-0
  weight addr=1
  wt size 2048
  wt width 1
  wt height 5
  wt channel 20
  data cube access by tsd:batch=tsd-11:0 id[offs]=2
  ::Surface surface=tsd-11: bindable=0
  data cube access by tsd:batch=tsd-2:0 id[offs]=1
  ::Surface surface=tsd-2: bindable=0
  SDP bias node @ op_slot = 1 batch_id = 0
  src precision 0
  dst precision 0
  x1 enable 1
  x1 precision 1
  x1 aluType 2
  x1 type 2
  x1 mode 1
  x1 act 0
  x1 shiftValue 0
  x1 aluOperand 0
  x1 mulOperand 1
  x1 truncate 0
  x2 enable 0
  y enable 0
  src tsd:tsd-10
  dst tsd:tsd-11/tb-11:off= 0
  bias tsd:tsd-2
  dependencyCount1
  conv_mode 0
  src addr=-1
  src type=2
  src size 18432
  src width 24
  src height 24
  src channel 20
  src linestride 768
  src surfstride 18432
  bias addr=1
  bias type=0
  bias size 40
  bias width 1
  bias height 1
  bias channel 20
  bias linestride 64
  bias surfstride 64
  dst addr=2
  dst type=0
  dst size 18432
  dst width 24
  dst height 24
  dst channel 20
  dst linestride 768
  dst surfstride 18432
  out_cvt enable 1
  out_cvt scale 16557
  out_cvt offset 0
  out_cvt truncate 23
  data cube access by tsd:batch=tsd-11:0 id[offs]=2
  ::Surface surface=tsd-11: bindable=0
  data cube access by tsd:batch=tsd-12:0 id[offs]=2
  ::Surface surface=tsd-12: bindable=0
  PDP node @ op_slot = 2 batch_id = 0
  pdp precision0
  pdp pool mode1
  src tsd:tsd-11
  dst tsd:tsd-12
  src addr=2
  src type=0
  dependencyCount1
  splitNum 1
  padLeft 0
  padTop 0
  padRight 0
  padBottom 0
  pool height 1
  pool width 1
  stride x 2
  stride y 2
  src size 18432
  src width 24
  src height 24
  src channel 20
  src linestride 768
  src surfstride 18432
  dst addr=2
  dst type=0
  dst size 4608
  dst width 12
  dst height 12
  dst channel 20
  dst linestride 384
  dst surfstride 4608
  data cube access by tsd:batch=tsd-12:0 id[offs]=2
  ::Surface surface=tsd-12: bindable=0
  data cube access by tsd:batch=tsd-3:0 id[offs]=1
  ::Surface surface=tsd-3: bindable=0
  Convolution node @ op_slot = 3 batch_id = 0
  src data loc: 0
  dst data loc: 2
  post y extension: 0
  in_precision 0
  out_precision 0
  pad_val 0
  conv mode 0
  data_reuse 0
  weight_reuse 0
  skip_data_rls 0
  skip_wt_rls 0
  eps 3
  fetch_grain 1
  data_format 36
  pixel_mapping 0
  batch 1
  weight_format 0
  b4d 1
  b4w 1
  batch_stride 0
  release 12
  post_extension 0
  pixel_override 1
  mean_format 0
  stride-x 1
  stride-y 1
  pad-left 0
  pad-top 0
  pad-right 0
  pad-bottom 0
  dilationx-x 1
  dilation-y 1
  pra_truncate 0
  inputwidthcsc 12
  inputheightcsc 12
  inputchannelcsc 20
  kernelwidthcsc 5
  kernelheightcsc 5
  kernelchannelcsc 20
  inputwidthcmac 8
  inputheightcmac 8
  bytesperkernel 500
  offsetU 0
  dependencyCount 3
  src tsd:tsd-12
  src addr=2
  src size 4608
  src width 12
  src height 12
  src channel 20
  src linestride 384
  src surfstride 4608
  dst tsd:tsd-13
  dst addr=-1
  dst size 4096
  dst width 8
  dst height 8
  dst channel 50
  dst linestride 256
  dst surfstride 2048
  wt  tsd:tsd-3
  weight addr=1
  wt size 25088
  wt width 5
  wt height 5
  wt channel 20
  data cube access by tsd:batch=tsd-14:0 id[offs]=2
  ::Surface surface=tsd-14: bindable=0
  data cube access by tsd:batch=tsd-4:0 id[offs]=1
  ::Surface surface=tsd-4: bindable=0
  SDP bias node @ op_slot = 4 batch_id = 0
  src precision 0
  dst precision 0
  x1 enable 1
  x1 precision 1
  x1 aluType 2
  x1 type 2
  x1 mode 1
  x1 act 0
  x1 shiftValue 0
  x1 aluOperand 0
  x1 mulOperand 1
  x1 truncate 0
  x2 enable 0
  y enable 0
  src tsd:tsd-13
  dst tsd:tsd-14/tb-14:off= 0
  bias tsd:tsd-4
  dependencyCount2
  conv_mode 0
  src addr=-1
  src type=2
  src size 4096
  src width 8
  src height 8
  src channel 50
  src linestride 256
  src surfstride 2048
  bias addr=1
  bias type=0
  bias size 100
  bias width 1
  bias height 1
  bias channel 50
  bias linestride 64
  bias surfstride 64
  dst addr=2
  dst type=0
  dst size 4096
  dst width 8
  dst height 8
  dst channel 50
  dst linestride 256
  dst surfstride 2048
  out_cvt enable 1
  out_cvt scale 16846
  out_cvt offset 0
  out_cvt truncate 26
  data cube access by tsd:batch=tsd-14:0 id[offs]=2
  ::Surface surface=tsd-14: bindable=0
  data cube access by tsd:batch=tsd-15:0 id[offs]=2
  ::Surface surface=tsd-15: bindable=0
  PDP node @ op_slot = 5 batch_id = 0
  pdp precision0
  pdp pool mode1
  src tsd:tsd-14
  dst tsd:tsd-15
  src addr=2
  src type=0
  dependencyCount2
  splitNum 1
  padLeft 0
  padTop 0
  padRight 0
  padBottom 0
  pool height 1
  pool width 1
  stride x 2
  stride y 2
  src size 4096
  src width 8
  src height 8
  src channel 50
  src linestride 256
  src surfstride 2048
  dst addr=2
  dst type=0
  dst size 1024
  dst width 4
  dst height 4
  dst channel 50
  dst linestride 128
  dst surfstride 512
  data cube access by tsd:batch=tsd-15:0 id[offs]=2
  ::Surface surface=tsd-15: bindable=0
  data cube access by tsd:batch=tsd-5:0 id[offs]=1
  ::Surface surface=tsd-5: bindable=0
  FullyConnected node @ op_slot = 6 batch_id = 0
  src data loc: 0
  dst data loc: 2
  post y extension: 0
  in_precision 0
  out_precision 0
  pad_val 0
  conv mode 0
  data_reuse 0
  weight_reuse 0
  skip_data_rls 0
  skip_wt_rls 0
  eps 2
  fetch_grain 1
  data_format 36
  pixel_mapping 0
  batch 1
  weight_format 0
  b4d 1
  b4w 13
  batch_stride 0
  release 4
  post_extension 0
  pixel_override 0
  mean_format 0
  stride-x 1
  stride-y 1
  pad-left 0
  pad-top 0
  pad-right 0
  pad-bottom 0
  dilationx-x 1
  dilation-y 1
  pra_truncate 0
  inputwidthcsc 4
  inputheightcsc 4
  inputchannelcsc 50
  kernelwidthcsc 4
  kernelheightcsc 4
  kernelchannelcsc 50
  inputwidthcmac 1
  inputheightcmac 1
  bytesperkernel 800
  offsetU 0
  dependencyCount 3
  src tsd:tsd-15
  src addr=2
  src size 1024
  src width 4
  src height 4
  src channel 50
  src linestride 128
  src surfstride 512
  dst tsd:tsd-16
  dst addr=-1
  dst size 512
  dst width 1
  dst height 1
  dst channel 500
  dst linestride 32
  dst surfstride 32
  wt  tsd:tsd-5
  weight addr=1
  wt size 400000
  wt width 4
  wt height 4
  wt channel 50
  data cube access by tsd:batch=tsd-19:0 id[offs]=2
  ::Surface surface=tsd-19: bindable=0
  data cube access by tsd:batch=tsd-6:0 id[offs]=1
  ::Surface surface=tsd-6: bindable=0
  SDP bias node @ op_slot = 7 batch_id = 0
  src precision 0
  dst precision 0
  x1 enable 1
  x1 precision 1
  x1 aluType 2
  x1 type 2
  x1 mode 1
  x1 act 1
  x1 shiftValue 0
  x1 aluOperand 0
  x1 mulOperand 1
  x1 truncate 0
  x2 enable 0
  y enable 0
  src tsd:tsd-16
  dst tsd:tsd-19/tb-19:off= 0
  bias tsd:tsd-6
  dependencyCount2
  conv_mode 0
  src addr=-1
  src type=2
  src size 512
  src width 1
  src height 1
  src channel 500
  src linestride 32
  src surfstride 32
  bias addr=1
  bias type=0
  bias size 1000
  bias width 1
  bias height 1
  bias channel 500
  bias linestride 64
  bias surfstride 64
  dst addr=2
  dst type=0
  dst size 512
  dst width 1
  dst height 1
  dst channel 500
  dst linestride 32
  dst surfstride 32
  out_cvt enable 1
  out_cvt scale 30309
  out_cvt offset 0
  out_cvt truncate 25
  data cube access by tsd:batch=tsd-19:0 id[offs]=2
  ::Surface surface=tsd-19: bindable=0
  data cube access by tsd:batch=tsd-8:0 id[offs]=1
  ::Surface surface=tsd-8: bindable=0
  FullyConnected node @ op_slot = 8 batch_id = 0
  src data loc: 0
  dst data loc: 2
  post y extension: 0
  in_precision 0
  out_precision 0
  pad_val 0
  conv mode 0
  data_reuse 0
  weight_reuse 0
  skip_data_rls 0
  skip_wt_rls 0
  eps 4
  fetch_grain 1
  data_format 36
  pixel_mapping 0
  batch 1
  weight_format 0
  b4d 1
  b4w 1
  batch_stride 0
  release 1
  post_extension 0
  pixel_override 0
  mean_format 0
  stride-x 1
  stride-y 1
  pad-left 0
  pad-top 0
  pad-right 0
  pad-bottom 0
  dilationx-x 1
  dilation-y 1
  pra_truncate 0
  inputwidthcsc 1
  inputheightcsc 1
  inputchannelcsc 500
  kernelwidthcsc 1
  kernelheightcsc 1
  kernelchannelcsc 500
  inputwidthcmac 1
  inputheightcmac 1
  bytesperkernel 500
  offsetU 0
  dependencyCount 3
  src tsd:tsd-19
  src addr=2
  src size 512
  src width 1
  src height 1
  src channel 500
  src linestride 32
  src surfstride 32
  dst tsd:tsd-20
  dst addr=-1
  dst size 32
  dst width 1
  dst height 1
  dst channel 10
  dst linestride 32
  dst surfstride 32
  wt  tsd:tsd-8
  weight addr=1
  wt size 5120
  wt width 1
  wt height 1
  wt channel 500
  data cube access by tsd:batch=tsd-21:0 id[offs]=2
  ::Surface surface=tsd-21: bindable=0
  data cube access by tsd:batch=tsd-9:0 id[offs]=1
  ::Surface surface=tsd-9: bindable=0
  SDP bias node @ op_slot = 9 batch_id = 0
  src precision 0
  dst precision 0
  x1 enable 1
  x1 precision 1
  x1 aluType 2
  x1 type 2
  x1 mode 1
  x1 act 0
  x1 shiftValue 0
  x1 aluOperand 0
  x1 mulOperand 1
  x1 truncate 0
  x2 enable 0
  y enable 0
  src tsd:tsd-20
  dst tsd:tsd-21/tb-21:off= 0
  bias tsd:tsd-9
  dependencyCount2
  conv_mode 0
  src addr=-1
  src type=2
  src size 32
  src width 1
  src height 1
  src channel 10
  src linestride 32
  src surfstride 32
  bias addr=1
  bias type=0
  bias size 20
  bias width 1
  bias height 1
  bias channel 10
  bias linestride 64
  bias surfstride 64
  dst addr=2
  dst type=0
  dst size 32
  dst width 1
  dst height 1
  dst channel 10
  dst linestride 32
  dst surfstride 32
  out_cvt enable 1
  out_cvt scale 20972
  out_cvt offset 0
  out_cvt truncate 25
  set symbol content name=task-0-addr0 size=40
  set symbol content name=task-0-dep_graph size=360
  set symbol content name=task-0-op_list size=1160
  set symbol content name=task-0-surf_list size=6440
  set symbol content name=task-0-lut_list size=700
  task address list (indices into global address list): 
  5
  1
  2
  3
  4
  6
  7
  8
  9
  10
  <>
  gathered reloc entry: address id=3 writeId=8 useBase=1ed9130 originalOffset=1ed91a4 offset=74 interface=1 subInterface=4 relocType=1
  gathered reloc entry: address id=3 writeId=8 useBase=1ed9130 originalOffset=1ed91a8 offset=78 interface=1 subInterface=4 relocType=2
  task_id=1 has 1 op slots and 1 batches 
  address list task context at [11, 17)
  ::Surface surface=tsd-21: bindable=0
  ::Surface surface=tsd-22: bindable=1
  Softmax node @ op_slot = 0 batch_id = 0
  src addr=2[45056]
  dst addr=4[0]
  input scale factor 0.141637
  output scale factor 0.00831604
  src size=32
  src format=0
  src width=1
  src height=1
  src channel=10
  dst size=32
  dst format=0
  dst width=1
  dst height=1
  dst channel=10
  set symbol content name=task-1-addr0 size=256
  set symbol content name=task-1-op_list size=24
  set symbol content name=task-1-op_buf_list size=512
  task address list (indices into global address list): 
  11
  1
  2
  3
  4
  12
  13
  14
  15
  16
  <>
  gathered reloc entry: address id=4 writeId=13 useBase=1edf5f0 originalOffset=1edf702 offset=112 interface=2 subInterface=4 relocType=1
  gathered reloc entry: address id=4 writeId=13 useBase=1edf5f0 originalOffset=1edf706 offset=116 interface=2 subInterface=4 relocType=2
  profile insertLoadable saving loadable with name fast-math
  profile getLoadable looked for loadable with name fast-math
  profile getLoadable looked for loadable with name fast-math
  closing wisdom context...
  </details>

下图是我整理的，Compiler工作的主要路线图，其中蓝色标注的部分是编译部分的主体，在正式进行编译之前，编译器会把 CaffeModel 转换成 Network 对象；把输入的profileName 转换成 Profile 对象，例如指定 fast-math 那么在profile 对象里就会把一些优化的开关打开，把targetname，就是nv_small/nv_large 这种，会转换成TargetConfig对象，内有硬件相关的配置信息。

![WorkFlow of  NVDLA Compiler](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/WorkFlow%20of%20%20NVDLA%20Compiler.png)

之后，通过 Network，生成一个标准的抽象语法树，再转换成与硬件相关的 engine_ast，engine_ast 就是编译器最核心的 IR，接下来的各种优化都是对该 IR 做变换。

<img src="https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/nvdla_ast.jpg" alt="cangraph2enggraph" style="zoom: 33%;" />

在这里可以看到 engine_ast 已经包含了硬件信息，在左边的抽象语法树里，只有算子的信息，右边会把算子对应的映射到硬件上去，映射关系在 log 里也可以找到：

```html
n-0:->n-0:dc-conv-0 n-1:bias-0 
n-1:->n-2:pdp-0 
n-2:->n-3:dc-conv-1 n-4:bias-1 
n-3:->n-5:pdp-1 
n-4:->n-6:fc-0 n-7:bias-2 
n-5:->n-8:sdp-scale-0 n-9:act-0 
n-6:->n-10:fc-1 n-11:bias-3 
n-7:->n-12:cpu-sm-0 
```

例如，n-0 原来是卷积运算的op节点，在engine_ast 中，乘 weight 运算被映射到 Convolution Engine 上，bias 添加运算被映射到 SDP Engine 上。

#### 2.2 main.c

在 main 函数里，编译器首先做的事情是对命令行的参数进行解析，不得不感叹这部分的逻辑用的居然是if、else语句来解析参数：

```cpp
if (ii >= argc)
  break;
const char* arg = argv[ii];

if (std::strcmp(arg, "-h") == 0) // help
{
  // Print usage
  showHelp = true;
  break;
}
```

我觉得，改成类似 Caffe 的，使用 gtest 的思路不会更好吗？

```cpp
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
    "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0,
    "Optional; network level.");
```

解析参数的过程中，会初始化TestAppArgs这个结构体，而在程序的开始，已经给其赋了初始值：

```cpp
static TestAppArgs defaultTestAppArgs =
{
    /* .project = */ "OpenDLA",
    /* .inputPath = */ "./",
    /* .inputName = */ "",
    /* .outputPath = */ "./",
    /* .testname = */ "",
    /* .testArgs = */ "",
    /* .prototxt = */ "",
    /* .caffemodel = */ "",
    /* .cachemodel = */ "",
    /* .profileName = */ "fast-math",
    /* .profileFile = */ "",
    /* .configtarget = */ TARGET_CONFIG_NAME,
    /* .calibtable = */ "",
    /* .quantizationMode = */ DEFAULT_QUANT_MODE,
    /* .numBatches = */ DEFAULT_BATCH_SIZE,
    /* .inDataFormat = */ DEFAULT_DATA_FMT,
    /* .computePrecision = */ nvdla::DataType::INT8
};
```

初始化完成之后，再调用同样写在该文件的 LaunchTest 函数:

```cpp
NvDlaError launchTest(const TestAppArgs* appArgs)
{
    NvDlaError e = NvDlaSuccess;
    TestInfo testInfo;

    PROPAGATE_ERROR_FAIL(testSetup(appArgs, &testInfo));

    PROPAGATE_ERROR_FAIL(parseAndCompile(appArgs, &testInfo));

    return NvDlaSuccess;

fail:
    return e;
}
```

在 testSetup 函数里，主要是检验一下对应的文件是否真的存在等。

其中又有一个非常重要的结构体：

```cpp
struct TestInfo
{
    // runtime
    nvdla::IRuntime* runtime;
    std::string inputLoadablePath;
    NvU8 *inputHandle;
    NvU8 *outputHandle;
    NvU8 *pData;
    bool dlaServerRunning;
    NvS32 dlaRemoteSock;
    NvS32 dlaServerSock;
    NvU32 numInputs;
    NvU32 numOutputs;
    NvDlaImage* inputImage;
    NvDlaImage* outputImage;
};
```

在该函数里，还有一个让人窒息的操作，给大家康康：

```cpp
    // Clear wisdomPath if any exist
    removeCmd += "rm -rf " + wisdomPath;
    ii = std::system(removeCmd.c_str()); // This is pretty awful
    if (ii != 0)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "system command failed: \"%s\"", removeCmd.c_str());
```

#### 2.3 工厂设计模式

在 NVDLA 的软件栈代码里，用了很多工厂模式的设计，例如创建一个 Network 对象，需要调用 NetworkFactory 的 newNetwork 方法，其他的，像是 Tensor、Node、Layer 都有对应的工厂，来生产不同的 Tensor、 Node、Layer。

此外，针对 Network、Tensor、Node、Layer 等等对象，其都定义了对应的虚接口类，如 INetwork、ITensor、INode、ILayer、这些接口类里通过纯虚函数定义了继承的子类必须实现的函数。

比如举个 Network 的例子：

```cpp
class Network : public INetwork
{
public: // externally facing

    virtual ITensor* addInput(const char* name, Dims4 dimensions);

    //	virtual void markChanged(const ILayer*);
    virtual bool markInput(ITensor * tensor);
    virtual void markOutput(ITensor* tensor);

    virtual IConvolutionLayer *    addConvolution(ITensor* input, int numOutputs, int paddingValue,
                                                  Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation,
                                                  Weights kernelWeights, Weights biasWeights, BiasMode biasmode, int numGroups);
    virtual IFullyConnectedLayer * addFullyConnected(ITensor* input, int outputSize, Weights kernelWeights, Weights biasWeights, BiasMode biasMode);
    virtual IActivationLayer *     addActivation(ITensor* input, ActivationType type);
    virtual IPoolingLayer *        addPooling(ITensor* input, PoolingType type,
                                              Dims2 windowSize, Dims2 stride, Dims2 tlPadding, Dims2 brPadding);
    virtual ILRNLayer *            addLRN(ITensor* input, int window, float alpha, float beta, float k);
    virtual IScaleLayer *          addScale(ITensor* input, ScaleMode mode, Weights shift, Weights scale, Weights power);
    virtual IBatchNormLayer *      addBatchNorm(ITensor* input, BatchNormMode mode, Weights mean, Weights variance, float epsilon);
    virtual ISoftMaxLayer *        addSoftMax(ITensor* input);
    virtual IConcatenationLayer *  addConcatenation(ITensor * const * inputs, int numInputs);
    virtual ISliceLayer *          addSlice(ITensor* input, int numOutputs);
    virtual IDeconvolutionLayer *  addDeconvolution(ITensor* input, int numOutputs, int paddingValue,
                                                    Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation,
                                                    Weights kernelWeights, Weights biasWeights, BiasMode biasMode, int numGroups);
    virtual IElementWiseLayer *    addElementWise(ITensor* input0, ITensor* input1, ElementWiseOperation op);

    virtual int  getNumInputs() const;
    virtual int  getNumOutputs() const;
    virtual int  getNumLayers() const ;

    virtual ILayer  * getLayer(int index)  const;
    virtual ITensor * getOutput(int index) const;
    virtual ITensor * getInput(int index)  const;

    virtual void setPoolingOutputDimensionsFormula      (OutputDimensionsFormula* callback);
    virtual void setConvolutionOutputDimensionsFormula  (OutputDimensionsFormula* callback);
    virtual void setDeconvolutionOutputDimensionsFormula(OutputDimensionsFormula* callback);

    virtual OutputDimensionsFormula& getPoolingOutputDimensionsFormula()       const;
    virtual OutputDimensionsFormula& getConvolutionOutputDimensionsFormula()   const;
    virtual OutputDimensionsFormula& getDeconvolutionOutputDimensionsFormula() const;

    virtual const std::vector<ITensor *>& getInputs()  const;
    virtual const std::vector<ILayer * >& getLayers()  const;
    virtual const std::vector<ITensor *>& getOutputs() const;

    virtual NvU16 getFactoryType() const;


public: // internally facing
    Network();
    virtual ~Network();
    virtual bool serializeTo(WisdomContainerEntry *) const;
    virtual bool deserializeFrom(WisdomContainerEntry *);
    virtual bool assignSymbols(Wisdom *);

protected:
    friend class Wisdom;
    friend class NetworkFactory;

    void destroy();

private:

    std::string newLayerName() const;
    std::string newTensorName() const;

    ITensor* addTensor(const std::string & s);
    const ILayer* findLayer(const std::string& name) const;
    bool checkNames(const char* name);

    std::vector<ITensor *> mTensors;
    std::vector<ILayer *>  mLayers;
    std::vector<ITensor *> mInputs;
    std::vector<ITensor *> mOutputs;

    // provides layer dimension caching. Layers can be mutated in any order and dimensions queried at any point.
    // So mutating a layer trims this, and querying always refills the cache up to the queried layer
    //	mutable std::vector<Dims3> mDimensions;

    // internal flags used by the builder that are not accessible through the API
    // int mInternalBuildFlags{ InternalBuildFlags::kENABLE_GRAPH_OPTIMIZATIONS };
    OutputDimensionsFormula* mConvDims, *mDeconvDims, *mPoolDims;
};
```

### 3. Caffe Parser

该部分是编译器的最前端，将 Caffemodel 和 Prototxt 转化成内部通用的 Network 对象。

![image-20210629203635668](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210629203635668.png)

将模型转化成 Network 对象，搞不懂返回 IBlobNameToTensor 这个对象有什么用，后面也没有用到数据，只做判断解析是否完成用了：

```cpp
const IBlobNameToTensor* CaffeParser::parse(const char* deployFile,
                                            const char* modelFile,
                                            INetwork * network)
{

    CHECK_NULL_RET_NULL(deployFile);
    CHECK_NULL_RET_NULL(modelFile);
    assert(mDimsCallback == 0);

    if (!mDimsCallback) {
        mDimsCallback = new CaffeParserPoolingDimsCallback;
    }
    // 设置 Network 输出层长宽的计算公式（向上取整）
    network->setPoolingOutputDimensionsFormula(mDimsCallback);

    // this is used to deal with dropout layers which have different input and output
    // 解析 Caffemodel 文件，提取权重数据等，转化到 mModel 对象中
    mModel = new dc::NetParameter();
    if (!readBinaryProto(mModel/*.get()*/, modelFile, mProtobufBufferSize))
    {
        gLogError << "Could not parse model file" << std::endl;
        return 0;
    }
    // 解析 Prototxt 文件，转化到 mDeploy 对象中
    mDeploy = new dc::NetParameter();
    if (!readTextProto(mDeploy/*.get()*/, deployFile))
    {
        gLogError << "Could not parse deploy file" << std::endl;
        return 0;
    }

    bool ok = true;
    CaffeWeightFactory weights(*mModel/**mModel.get()*/,
                               false /*weightType == DataType::kHALF*/, mTmpAllocs);

    mBlobNameToTensor = new BlobNameToTensor();

    for (int i = 0; i < mDeploy->input_size(); i++)
    {
        Dims4 dims;
        if (mDeploy->input_shape_size()) {
            dims.n = (int)mDeploy->input_shape().Get(i).dim().Get(0);
            dims.c = (int)mDeploy->input_shape().Get(i).dim().Get(1);
            dims.h = (int)mDeploy->input_shape().Get(i).dim().Get(2);
            dims.w = (int)mDeploy->input_shape().Get(i).dim().Get(3);
        }
        else { // deprecated, but still used in a lot of networks
            dims.n = (int)mDeploy->input_dim().Get(i * 4 + 0);
            dims.c = (int)mDeploy->input_dim().Get(i * 4 + 1);
            dims.h = (int)mDeploy->input_dim().Get(i * 4 + 2);
            dims.w = (int)mDeploy->input_dim().Get(i * 4 + 3);
        }

        ITensor* tensor = network->addInput(mDeploy->input().Get(0).c_str(), dims);
        mBlobNameToTensor->add(mDeploy->input().Get(0), tensor);

    }

    for (int i = 0; i < mDeploy->layer_size() && ok; i++)
    {
        const dc::LayerParameter& layerMsg = mDeploy->layer(i);
        if (layerMsg.has_phase() && layerMsg.phase() == dc::TEST) {
            continue;
        }

        if (layerMsg.type() == "Dropout")
        {
            mBlobNameToTensor->add(layerMsg.top().Get(0),
                                   mBlobNameToTensor->find(layerMsg.bottom().Get(0).c_str()));
            continue;
        }

        if (layerMsg.type() == "Input")
        {
            const dc::InputParameter& p = layerMsg.input_param();
            for (int i = 0; i < layerMsg.top_size(); i++)
            {
                const dc::BlobShape& shape = p.shape().Get(i);
                Dims4 dims(shape.dim().Get(0), shape.dim().Get(1), shape.dim().Get(2), shape.dim().Get(3));
                ITensor* tensor = network->addInput(layerMsg.top(i).c_str(), dims);
                mBlobNameToTensor->add(layerMsg.top().Get(i), tensor);
            }
            continue;
        }
        if (layerMsg.type() == "Flatten")
        {
            ITensor* tensor = (*mBlobNameToTensor)[layerMsg.bottom().Get(0)];
            (*mBlobNameToTensor)[layerMsg.top().Get(0)] = tensor;
            std::cout << "Warning: Flatten layer ignored." << std::endl;
            continue;
        }

        LayerParseFnMap::iterator v = gParseTable.find(layerMsg.type());

        if (v == gParseTable.end())
        {
            gLogError << "could not parse layer type " << layerMsg.type() << std::endl;
            ok = false;
        }
        else
        {
            ILayer* layer = (*v->second)(network, layerMsg, weights, mBlobNameToTensor);

            if (layer == 0)
            {
                gLogError << "error: parsing layer type " << layerMsg.type() <<
                    " index " << i << std::endl;
                ok = false;
            }
            else
            {
                layer->setName(layerMsg.name().c_str());
                mBlobNameToTensor->add(layerMsg.top(0), layer->getOutput(0));
            }
        }
    }

    mBlobNameToTensor->setTensorNames();
    return ok && weights.isOK() ? mBlobNameToTensor : 0;
}
```

确定最终网络的输出层是哪一层：

```cpp

int CaffeParser::identifyOutputs(INetwork * network)
{
    std::set< ITensor* > outputTensors;
    std::set< ITensor* > inputTensors;

    // 缓存每层的输入 Tensor 和输出 Tensor
    for (int l = 0; l < network->getNumLayers(); ++l)
    {
        // 获取每一层
        ILayer* layer = network->getLayer(l);
        if (!layer)
            return -1;
        // 将输入 Tensor 存入inputTensors 里
        for (int ii = 0; ii < layer->getNumInputs(); ++ii) {
            inputTensors.insert(layer->getInput(ii));
        }
        // 将输出 Tensor 存入 OutputTensors 里
        for (int oo = 0; oo < layer->getNumOutputs(); ++oo)
        {
            outputTensors.insert(layer->getOutput(oo));
        }
    }

    for (std::set<ITensor*>::iterator oi = outputTensors.begin(); oi != outputTensors.end(); ++oi)
    {
        // oi 是遍历 outputTensors，如果这个 Tensors 不是输入的 Tensor 那他就是最后一层输出的 Tensor，这个逻辑没毛病。
        // an output tensor which is not an input to any other layers is a network output tensor
        if (inputTensors.find(*oi) == inputTensors.end())
        {
            network->markOutput(*oi);
            gLogInfo << "mark " << (*oi)->getName() << std::endl;
        }
    }

    return network->getNumOutputs();
}
```

### 4. parseTensorScales

然后是解析量化的 Scales Json 文件：

```cpp
NvDlaError parseTensorScales(const TestAppArgs* appArgs, TestInfo *i, nvdla::INetwork* network)
{
    NvDlaError e = NvDlaSuccess;
    NvDlaStatType stat;
    std::string calibTableFile = /*i->calibTablesPath + "/" + */appArgs->calibTable;

    PROPAGATE_ERROR_FAIL(NvDlaStat(calibTableFile.c_str(), &stat));

    // populate the scaling factor/dynamic range of each of the tensors on the network
    {
        FILE* fp = fopen(calibTableFile.c_str(), "r");
        char readBuffer[TEST_PARAM_FILE_MAX_SIZE] = {0};

        rapidjson::Document doc;
        rapidjson::FileReadStream inStr(fp, readBuffer, sizeof(readBuffer));

        doc.ParseStream(inStr);
        if (doc.HasParseError())
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "JSON parsing error: %s", GetParseError_En(doc.GetParseError()));
        }

        {
            std::vector<nvdla::ILayer*> networkLayers = network->getLayers();
            std::vector<nvdla::ITensor*> networkInputs = network->getInputs();

            std::vector<nvdla::ILayer*>::iterator li = networkLayers.begin();
            std::vector<nvdla::ITensor*>::iterator nii = networkInputs.begin();

            // set scaling factor for the network input tensors
            // 这里是对输入的数据进行量化，一般是 data 层
            for (; nii != networkInputs.end(); ++nii)
            {
                NvF32 scale = 0.0f;
                NvF32 min = 0.0f;
                NvF32 max = 0.0f;
                std::string tName = (*nii)->getName();
                // 可以发现这里是根据每个 Layer 的 Name 来索引网络的 scale，也就是说 TensorRT 量化出来的那个是不能直接用的，详情见之前的博客
                if (doc[tName.c_str()].HasMember("scale")) {
                    // 因为出来的都是 scale，所以总是会使用这个分支的
                    scale = doc[tName.c_str()]["scale"].GetFloat();
                    // min 和 max 都是根据 scale 来调整上下届，具体的量化算法是什么呢？
                    min = scale * -127.0f;
                    max = scale * 127.0f;
                }
                else if (doc[tName.c_str()].HasMember("min") && doc[tName.c_str()].HasMember("max")) {
                    min = doc[tName.c_str()]["min"].GetFloat();
                    max = doc[tName.c_str()]["max"].GetFloat();
                }
                else {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Atleast 1 of scale or min-max should be specified for %s\n", tName.c_str());
                }

                // set same dynamic range for all channels of the tensor (cIndex = -1)
                PROPAGATE_ERROR_FAIL( (*nii)->setChannelDynamicRange(-1, min, max) );
                const_cast<TestAppArgs*>(appArgs)->tensorScales.insert(std::pair<std::string, NvF32>(tName, scale));
            }
          //  这里是提取每一层Layer的Scale信息
          for (; li != networkLayers.end(); ++li)
            {
                NvF32 scale = 0.0f;
                NvF32 min = 0.0f;
                NvF32 max = 0.0f;
                std::string lName = (*li)->getName();
                nvdla::ITensor* outTensor = (*li)->getOutput(0);

                if (doc[lName.c_str()].HasMember("scale")) {
                    scale = doc[lName.c_str()]["scale"].GetFloat();
                    min = scale * -127.0f;
                    max = scale * 127.0f;
                }
                else if (doc[lName.c_str()].HasMember("min") && doc[lName.c_str()].HasMember("max")) {
                    min = doc[lName.c_str()]["min"].GetFloat();
                    max = doc[lName.c_str()]["max"].GetFloat();
                }
                else {
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Atleast 1 of scale or min-max should be specified for %s\n", lName.c_str());
                }

                // set same dynamic range for all channels of the tensor (cIndex = -1)
                PROPAGATE_ERROR_FAIL( outTensor->setChannelDynamicRange(-1, min, max) );
                const_cast<TestAppArgs*>(appArgs)->tensorScales.insert(std::pair<std::string, NvF32>(lName, scale));
            }
        }

        fclose(fp);
    }

fail:
    return e;
}
```

```cpp
NvDlaError Tensor::setChannelDynamicRange(NvS32 chnlIndx, NvF32 min, NvF32 max)
{
    NvDlaError e = NvDlaSuccess;
    // min/max = scale * +- 127 -> max(min,max)/127.0 = scale
    NvF32 scaleFactor = std::max<NvF32>(std::fabs(min), std::fabs(max))/127.0;

    // 输入的 chnlIndx 数不能大于 Tensor 的 Channel 数
    if (chnlIndx >= mDimensions.c)
    {
        e = NvDlaError_BadParameter;
        goto fail;
    }
    else if (chnlIndx == -1)
    {
        // clear existing scales and adjust vector capacity if need be before inserting
        mChnlScales.clear();
        // mChnlScales 是一个 Vector、reserve 是为其开辟了内存空间，但是这个 Vector 的 Size 没有改变
        mChnlScales.reserve(mDimensions.c);
        // As we can see, 如果指定 -1，那就是对 Tensor 的每个通道都使用同一个 scaleFactor
        for (NvU32 cc = 0; cc < static_cast<NvU32>(mDimensions.c); ++cc)
        {
            mChnlScales.push_back(scaleFactor);
        }
    }
    else
    {
        // adjust vector capacity before inserting
        // 否则就只设置 chnlIndx 这个通道的 scaleFactor
        if (mChnlScales.capacity() < (size_t)mDimensions.c)
        {
            mChnlScales.reserve(mDimensions.c);
        }
        mChnlScales.insert(mChnlScales.begin() + chnlIndx, scaleFactor);
    }

fail:
    return e;
}
```

### 5. setNetworkTransient

```cpp
bool Wisdom::setNetworkTransient(INetwork *inetwork)
{
    Network *network = NetworkFactory::priv(inetwork);
    if ( !network ) {
        gLogError << "unrecognized network presented to Wisdom" << endl;
        return false;
    }
    //
    // prior to serialization all objects need entries in the symbol table.
    // since the network ultimately refers to them all, just triggering
    // it's symbols to be resolved is sufficient to be able to deserialize
    // it later.
    // 设置符号表（symbol table），具体可以看 wisdom -> symbol table
    bool ok = network->assignSymbols(m_container->wisdom_priv());
    m_network = network;
    return ok;
}
```

### 6.profile

根据 "fast-math"  配置到  i->wisdom->m_profiler 里去，也就是进行一个字符串到具体对象的转化；

```cpp
NvDlaError Compiler::compileInternal(const char *tp_name, const char *target_config_name, ILoadable **peli, bool fullCompile)
{
    NvDlaError e = NvDlaSuccess;
    DLAInterface *target_dla = 0;
    DLAInterface *dla_if = 0;
    Profiler *profiler = 0;
    ProfileFactory::ProfilePrivPair p_profile;
    Profile *profile = 0;
    TargetConfig *target_config = 0;
    vector<engine_ast::Graph *> g;
    NVDLA_UNUSED(target_dla);
    NVDLA_UNUSED(dla_if);


    LoadableFactory::LoadablePrivPair l(0, 0);

    // && == with fullCompile or otherwise ok during compileCheck?

    DumpCanonicalGraphJson dump_can;
    DumpEngineGraphJson dump_eng;

    if ( !m_wisdom )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "No wisdom available.");
    }

    profiler = ProfilerFactory::priv(m_wisdom->getProfiler());
    if ( !profiler )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "No profiler available.");
    }
    // ？？？？这里为什么要再来一下，总之很迷惑，这个在之前已经解析过了啊
    profile = ProfileFactory::priv(profiler->getProfile(tp_name));
    if ( !profile )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Couldn't find profile to compile.");
    }
    // 将 target_config_name 转换为 target_config 对象
    target_config = TargetConfigFactory::priv(profiler->getTargetConfig(target_config_name));
    if ( !target_config )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Couldn't find target config to compile.");
    }

		// 正式的编译在这里
    PROPAGATE_ERROR_FAIL( compileInternal(profile, target_config, peli, fullCompile) );
fail:
    return e;

}
```

### 7. Network 到 canonical_ast

```cpp
//
// the following generates a 1:1 mapping with the Canonical graph input.
//
canonical_ast::Graph *canonical_ast::generateGraph(Network *network)
{
    vector<canonical_ast::Edge *> input_edges;
    vector<canonical_ast::Edge *> output_edges;

    map<canonical_ast::Node *, Layer *, Graph::nodeCompareFn>  node_layer;
    map<canonical_ast::Node *, Layer *, Graph::nodeCompareFn>::iterator lni;

    map<Tensor *, canonical_ast::Edge *>  tensor_edge;
    map<Tensor *, Tensor *>  nw_tensor_to_can_tensor;
    map<Tensor *, canonical_ast::Edge *>::iterator tei;

    Graph *graph = new Graph();
    // 下面这个循环把network中的inputTensor汇总成一个数组，并把Graph中input_edges数组大小设定成
    // network中的inputTensor的数组大小
    vector<Tensor *> network_inputs;
    for (int ni = 0; ni < network->getNumInputs(); ++ni)
    {
        network_inputs.push_back(TensorFactory::priv(network->getInput(ni)));
    }
    input_edges.resize(network_inputs.size());

    //下面这个循环把network中的outputTensor汇总成一个数组，并把Graph中outputedges数组大小设定成
    //network中的outputTensor的数组大小
    vector<Tensor *> network_outputs;
    for (int ni = 0; ni < network->getNumOutputs(); ++ni)
    {
        network_outputs.push_back(TensorFactory::priv(network->getOutput(ni)));
    }
    output_edges.resize(network_outputs.size());

    //    gLogInfo << "canonical_ast::" << __func__ << " network shows " << network_inputs.size() << " inputs and " <<
    //        network_outputs.size() << " outputs" << endl;
    //下面这个循环迭代network中的layer序列，根据每个layer的信息分别建立Graph中的相应的Node
  for (int li = 0; li < network->getNumLayers(); li++)
    {
        ILayer *ilayer = network->getLayer(li);
        Layer *layer = LayerFactory::priv(ilayer);
        if ( !(ilayer && layer) )
        {
            gLogError << __func__ << " encountered null layer at network layer index=" << li << endl;
            continue;
        }
        //根据network中的layer，建立相应的Node
        canonical_ast::Node *can_node = newCanonicalNode(layer);
        if ( !can_node )
        {
            delete graph; // blow up
            graph = 0;
            goto done;
        }
        can_node->setGraph(graph);  // 这样可以从layer直接找到layer所在的graph
        graph->insertNode(can_node);  // 为 graph 插入node

        can_node->setId(graph->nextNodeId()); // 设置 node 的id，n-0 n-1 这样
        can_node->setName(layer->getName());  // 设置 node 的name

        node_layer[can_node] = layer;     //  设置 node 和 layer 的 map 关系

    }

    //
    // Now all the layer nodes are in the graph.
    // For each layer assemble the edges.
    //
    //现在，所有network中的layer都在graph中建立了相应的node，并且这个对应关系也记录在了node_layer的MAP中
    //下面循环迭代这个MAP中的每一项
    for (lni = node_layer.begin(); lni != node_layer.end(); ++lni)
    {
        canonical_ast::Node *node = lni->first;
        Layer *l = lni->second;

        size_t input_tensors = 0, output_tensors = 0, aux_input_tensors = 0;
        vector<Tensor *> io_tensors, aux_tensors;
        NVDLA_UNUSED(aux_input_tensors);
        //针对network中当前迭代的这个layer，找出其全部inputTensors并加入io_tensors列表
      for(int ii = 0, II = l->getNumInputs(); ii < II; ++ii)
        {
            Tensor *tensor = TensorFactory::priv(l->getInput(ii));
            if ( !tensor )
            {
                gLogError << __func__ << " 3.<null>.i." << ii << endl;
                continue;
            }
            io_tensors.push_back(tensor);
            input_tensors++;
        }
      //针对network中当前迭代的这个layer，找出其全部outputTensors并加入io_tensors列表
      for(int oo = 0, OO = l->getNumOutputs(); oo < OO; ++oo)
        {
            Tensor *tensor = TensorFactory::priv(l->getOutput(oo));
            if ( ! tensor )
            {
                gLogError << __func__ << " 3.<null>.o." << oo << endl;
                continue;
            }
            io_tensors.push_back(tensor);
            output_tensors++;
        }
      //针对当前layer，迭代刚刚找到的全部iotensor的列表
      for(size_t io = 0, IO = io_tensors.size(); io < IO; ++io)
        {
            Tensor *nw_tensor = io_tensors[io];
            bool is_input = io < input_tensors;//根据当前tensor在列表中的位置判断是input还是output
                                               // edge_side是个enum值，input=SECOND，output=FIRST
            ast::EdgeSide edge_side( is_input ? ast::EdgeSideEnum::SECOND : ast::EdgeSideEnum::FIRST);
            // edge_dir是个enum值，有单向双向和无方向三种，这里统一设定为单向
            ast::EdgeDirection edge_dir(ast::EdgeDirectionEnum::DIRECTED);
            // 在tensor_edge映射MAP中查找当前tensor的对应项
            map<Tensor *, canonical_ast::Edge *>::iterator f = tensor_edge.find(nw_tensor);
            canonical_ast::Edge *can_edge = 0;// graph中的edge
            Tensor* can_tensor = 0;// graph中的tensor
            if ( f == tensor_edge.end() ) //如果没有在MAP中找到对应项
            {
                can_edge = new canonical_ast::Edge(); //新建一个graph中的edge
                can_edge->setGraph(graph); //把新建的edge的container设定为graph

                can_tensor = nw_tensor->clone();//把network中的tensor复制到一个新的变量can_tensor
                can_tensor->setNetwork(NULL);  //由于这个新的tensor变量将加入graph所以其network指针清空，不在指向原来的network(这里是复制一份tensor，network中原来的tensor还在)
                can_tensor->setTensorType(TensorType::kIO);//graph中的tensor设定为IO类型
                can_edge->setId(graph->nextEdgeId()); //graph中edge的Id设定为string，e-0,e-1,e-2等
                can_edge->setOriginalTensor(can_tensor);//graph中的edge的原始tensor设定为can_tensor，注意，这里的OriginalTensor指向的是从network中复制clone过来的一个副本，并不在network中，可以看出这里的包含关系，graph-->can_edge-->can_tensor
                graph->insertEdge(can_edge);//把根据network中1个layer的iotensor新建的edge加入graph列表
                //tensor_edge映射MAP加入nw中tensor到graph中edge映射
                //nw_tensor_to_can_tensor映射MAP加入nw中tensor到graph中edge的tensor映射
                tensor_edge[nw_tensor] = can_edge;
                nw_tensor_to_can_tensor[nw_tensor] = can_tensor;
            } else {
                can_edge = f->second;
            }
            //把当前新建的edge加入到node的edge_side侧列表当中
            graph->appendNodeToEdge(can_edge, edge_side, node);

            // if this is an input node it could be one of the network inputs.
            // if so keep track of it.
            if ( is_input )
            {
              //迭代整个network的inputTensors列表
                for ( size_t iti = 0; iti < network_inputs.size(); iti++)
                {
                  //如果当前node对应的这个inputTensor在整个network的inputTensors列表当中
                  if ( nw_tensor == network_inputs[iti] )
                    {
                        // gLogInfo << " identified input edge: " << (int)iti << " tensor id " << tensor->getName() << endl;
                        input_edges[iti] = can_edge; //把当前edge加入graph的input_edges列表当中
                        can_tensor = nw_tensor_to_can_tensor[nw_tensor];
                        //设定当前tensor属性为INPUT
                        can_tensor->setTensorType(TensorType::kNW_INPUT);
                        break;
                    }
                }
                node->markInputEdge(can_edge); //告诉当前node，你的这个edge是一个网络inputedge
            }
            else
            {
                // 相对的，mark output
                for ( size_t oti = 0; oti < network_outputs.size(); oti++)
                {
                    if ( nw_tensor == network_outputs[oti] )
                    {
                        // gLogInfo << " identified output edge: " << (int)oti << " tensor id " << tensor->getName() << endl;
                        output_edges[oti] = can_edge;
                        can_tensor = nw_tensor_to_can_tensor[nw_tensor];
                        can_tensor->setTensorType(TensorType::kNW_OUTPUT);
                        break;
                    }
                }
                node->markOutputEdge(can_edge);
            }
        }
    }

    if ( input_edges.size() )
    {
        graph->setInputEdges(input_edges); //设定整个graph的inputedges队列为input_edges
    }
    if ( output_edges.size() )
    {
        graph->setOutputEdges(output_edges); //设定整个graph的outputedges队列为output_edges
    }

    graph->scoredOrdering()->generate(); // graph 计分牌生成，这部分比较复杂
    graph->markClean(); // 清除graph的m_dirty脏标志，所有对graph的更改都要设定m_dirty为true

done:
    return graph; // 把按照network生成的graph作为返回值返回
}
```

### 8. canonical_ast to engine_ast

```cpp
//这个函数完成的是两个graph的转换，通过参数可以看到，输入不仅仅由can_graph,还有编译器的profile和编译目标配置target_config，说明转换后的graph应该反应部分硬件和编译选项的要求
engine_ast::Graph *engine_ast::generateGraph(Profile *profile, TargetConfig *target_config, canonical_ast::Graph *can_graph)
{
    NvDlaError e = NvDlaSuccess;
    vector<engine_ast::Edge *> input_edges;
    vector<engine_ast::Edge *> output_edges;

    vector<canonical_ast::Node *> can_edge_first_nodes, can_edge_second_nodes;
    map<canonical_ast::Node *, engine_ast::Node *> can_to_eng_sink_node_map, can_to_eng_source_node_map;
    map<canonical_ast::Edge *, engine_ast::Edge *> can_to_eng_edge_map;
    vector<canonical_ast::Node *>::const_iterator f, begin, end;
    vector<engine_ast::Node *> first_nodes, second_nodes;
    engine_ast::Graph *eng_graph;

    if ( !profile )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "must associate profile with Engine AST generateGraph");
    }

    if ( !target_config )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "must associate target_config with Engine AST generateGraph");
    }
    //编译目标是否支持批处理
    if (target_config->isBatchModeCapable())
    {
        NvU32 numBatches = profile->multiBatchSize();
        NvU32 maxBatches = target_config->maxBatchSize();

        if (numBatches > maxBatches)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "numbatches is greater than allowed maxbatches (%d)", maxBatches);
        }
    }
    // 建立engine_graph对象，参数是profile和target_config
    eng_graph  = new engine_ast::Graph(profile, target_config);
    if ( !eng_graph )
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_InsufficientMemory, "Can't create a new Engine AST");
    }
    //初始化eng_graph的资源，主要是内存池和LutManager，内存池包括GLOBAL_DRAM_POOL，LOCAL_DRAM_POOL
    //如果profile开启了SRAM，那么还有LOCAL_CVSRAM_POOL，这三个mempool的大小由profile参数指定
    e = eng_graph->initGraphResources();
    if (e != NvDlaSuccess)
    {
        delete eng_graph;
        eng_graph = NULL;
        ORIGINATE_ERROR_FAIL(NvDlaError_InsufficientMemory, "Couldn't initialize all graph resources");
    }
    // engine graph 访问计分板
    eng_graph->setScoredOrdering( new ScoredDependencyOrdering(eng_graph) );
    eng_graph->setOrdering(new DependencyOrdering(eng_graph->scoredOrdering()));

    //
    // create edges to mirror the canonical edges.
    // 这里可以看语法树的转换，canonical_ast 的 edge 是被复制到 engine_ast 上的
    for ( set<canonical_ast::Edge *>::iterator cei = can_graph->edges().begin(), CEI = can_graph->edges().end();
          cei != CEI; ++cei )
    {
        //根据canonical_ast::Edge建立engine_ast::Edge对象
        engine_ast::Edge* engine_edge = new engine_ast::Edge(*cei);
        Tensor* engine_tensor = 0;
        if ( !engine_edge )
        {
            delete eng_graph; // blow up
            eng_graph = NULL;
            ORIGINATE_ERROR_FAIL(NvDlaError_InsufficientMemory, "Couldn't transform canonical edge '%s' into engine edge", (*cei)->id().c_str());
        }
      //engine_tensor复制自can_tensor,前面讲过can_tensor其实是clone自network的tensor
      engine_tensor = (*cei)->originalTensor()->clone();
        engine_tensor->setDataFormat(nvdla::DataFormat::NCHW); // all tensors are NCHW unless otherwise specified
        engine_tensor->setNetwork(NULL);                       // get rid of any connections back to the network builder 这里在创建can_tensor的时候已经变成 NULL 了。。

        engine_edge->setGraph(eng_graph); //指定engine_edge的container为eng_graph
        engine_edge->setId(eng_graph->nextEdgeId()); //设定engine_edge的Id，string类型，e-0,e-1等
        engine_edge->setDataEdge(); //设定edge的type为DATA
        engine_edge->setOriginalTensor(engine_tensor); //指定edge关联的tensor

        can_to_eng_edge_map[*cei] = engine_edge; //建立can_edge和engine_edge的关联MAP
        eng_graph->insertEdge(engine_edge); //把engine_edge加入eng_graph的edge列表

    }
    //如果没有指定multibatchsize，则根据network的input tensor的n指定推导multibatchsize
    //如果指定了multibatchsize，那就按照multibatchsize来执行
    if (profile->multiBatchSize() == 0)
    {
        // Patch up profile->multiBatchSize()
        // The compiler should be querying this information from the network instead of the profile

        // Collect the multibatch size of the network, based on the input tensor dimensions
        // 设置输入的维度
        for ( vector<canonical_ast::Edge *>::const_iterator cie = can_graph->inputEdges().begin();
                    cie != can_graph->inputEdges().end(); ++cie)
        {
            engine_ast::Edge *input_edge = can_to_eng_edge_map[*cie];
            Dims4 networkDims = input_edge->originalTensor()->getDimensions();

            //根据input_edge的tensor Dimension的n，设定profile的multibatchsize
          PROPAGATE_ERROR_FAIL(profile->setMultiBatchSize(networkDims.n));
        }
    }

    //
    // create nodes to mirror the canonical nodes
    // 迭代can_graph的所有nodes
    for ( set<canonical_ast::Node *>::iterator cni = can_graph->nodes().begin(), CNI = can_graph->nodes().end();
          cni != CNI; ++cni )
    {
        engine_ast::Graph::EdgeSequence engSrcEdges; //engine_graph的SrcEdges
        engine_ast::Graph::EdgeSequence engSinkEdges; //engine_graph的SinkEdges
        engine_ast::Graph::NodeSequence engNodes; //engine_graph的Nodes
        canonical_ast::Graph::EdgeSequence canSrcEdges = can_graph->nodeEdges(*cni, ast::EdgeSideEnum::SECOND); //can_graph的当前node的inputedge的总和
        canonical_ast::Graph::EdgeSequence canSinkEdges = can_graph->nodeEdges(*cni, ast::EdgeSideEnum::FIRST); //can_graph的当前node的outputedge的总和
        canonical_ast::Graph::EdgeSequenceIterator cei;
        // 这里就是复制一下Src、SinkEdge
        // 找出所有canSrcEdges对应的engine_edge,放入engSrcEdges列表
        for (cei = canSrcEdges.begin(); cei != canSrcEdges.end(); ++cei)
        {
            engSrcEdges.push_back(can_to_eng_edge_map[*cei]);
        }
        // 找出所有canSinkEdges对应的engine_edge,放入engSinkEdges列表
        for (cei = canSinkEdges.begin(); cei != canSinkEdges.end(); ++cei)
        {
            engSinkEdges.push_back(can_to_eng_edge_map[*cei]);
        }
        //从当前的can_node转化出eng_nodes，之所以是end_nodes是因为一个can_node可以对应2，3个eng_nodes
        //转换完毕是否把结果的engNodes挂在eng_graph上？？？需要详细看transformCanNode()函数代码
        e = transformCanNode(eng_graph, *cni, engSrcEdges, engSinkEdges, engNodes);
        if ( e != NvDlaSuccess )
        {
            delete eng_graph; // blow up
            eng_graph = NULL;
            ORIGINATE_ERROR_FAIL(e, "Couldn't transform canonical node '%s' into engine node", (*cni)->id().c_str());
        }
        //n-0:->n-0:dc-conv-0 n-1:bias-0
        //n-1:->n-2:pdp-0
        //n-2:->n-3:dc-conv-1 n-4:bias-1
        //n-3:->n-5:pdp-1
        //n-4:->n-6:fc-0 n-7:bias-2
        //n-5:->n-8:sdp-scale-0 n-9:act-0
        //n-6:->n-10:fc-1 n-11:bias-3
        //n-7:->n-12:cpu-sm-0
        //上面列出的就是transformCanNode()函数的转换结果，可以看到1个can_node有可能转换成2个eng_node
        //是因为can_node是直接对那个network模型的node，而在engine中，一个network模型中的node有可能是
        //需要2个engine前后协同计算才能得到结果，所有这里的eng_node其实已经是映射到硬件上的node了
        if ( eng_graph->debugGraphDump() )
        {
            gLogInfo << (*cni)->id() << ":->";
            for (vector<engine_ast::Node *>::iterator ni = engNodes.begin(); ni != engNodes.end(); ++ni)
            {
                gLogInfo << (*ni)->id() << ":" << (*ni)->name() << " ";
            }
            gLogInfo << std::endl;
        }
    }

  //迭代can_graph的所有inputEdges
  for ( vector<canonical_ast::Edge *>::const_iterator cie = can_graph->inputEdges().begin();
            cie != can_graph->inputEdges().end(); ++cie)
    {
      //找出can_graph的首个inputEdge对应的eng_edge
      engine_ast::Edge *first_edge = can_to_eng_edge_map[can_graph->inputEdges().front()];
      //当前迭代的can_edge对应的eng_edge
      engine_ast::Edge *input_edge = can_to_eng_edge_map[*cie];
      //当前eng_edge对应的tensor格式设定为profile指定的InputDataFormat
      input_edge->originalTensor()->setDataFormat(profile->networkInputDataFormat());
      // 要求所有的inputedge的multibatch参数n必须一致
      // Determine if multibatch parameters are consistent for all input tensors
        if (first_edge->originalTensor()->getDimensions().n != input_edge->originalTensor()->getDimensions().n)
        {
            ORIGINATE_ERROR_FAIL(NvDlaError_BadValue, "Input tensor multibatch dimensions mismatch: %d != %d", first_edge->originalTensor()->getDimensions().n, input_edge->originalTensor()->getDimensions().n);
        }

        Dims4 networkDims = input_edge->originalTensor()->getDimensions();
      //拿所有inputedge的multibatch参数n和profile指定的multibatch参数进行比较，如果不一致
      //则以profile指定的参数为准，并把inputedge中的tensor变量的networkDims.n更新为profile指定的值
        if ( networkDims.n != (NvS32)profile->multiBatchSize() )
        {
            gLogWarning << "Overriding input multibatch size from " << networkDims.n << " to " << profile->multiBatchSize() << endl;
            networkDims.n = profile->multiBatchSize();
            input_edge->originalTensor()->setDimensions(networkDims);
        }

        // if it is IMG input format, ensure #chnls match between model and profile params
        // 如果profile指定的输入IMG tensor的channel数与network提供的networkDims.c不一致
        // 则以profile设定的input tensor的channel值为准，同时更新engine_graph的inputedge对应的tensor
        // 的networkDims.c的值
        if ( profile->networkInputSurfaceFormat().category() == surface::SurfaceCategoryEnum::IMG &&
             networkDims.c != profile->networkInputSurfaceFormat().channelsPerAtom())
        {
            gLogWarning << "Prototxt #chnls (C = "
                        << networkDims.c
                        << ") != Profile #chnls for input ("
                        << profile->networkInputSurfaceFormat().c_str()
                        << ": C = "
                        << (int)profile->networkInputSurfaceFormat().channelsPerAtom()
                        << "). Preferring #chnls from Profile for compiling."
                        << endl;
            networkDims.c = profile->networkInputSurfaceFormat().channelsPerAtom();
            input_edge->originalTensor()->setDimensions(networkDims);

            // copy the tensor scales and offsets to the extra channel if any
            if (input_edge->originalTensor()->getChannelScales().size())
            {
                NvF32 tensorScale  = input_edge->originalTensor()->getChannelScales().at(0);
                std::vector<NvF32> channelScales;
                for (NvU32 cc = 0; cc < (NvU32)networkDims.c; ++cc)
                {
                    channelScales.push_back(tensorScale);
                }
                input_edge->originalTensor()->setChannelScales(channelScales);
            }

            if (input_edge->originalTensor()->getChannelOffsets().size())
            {
                NvF32 tensorOffset = input_edge->originalTensor()->getChannelOffsets().at(0);
                std::vector<NvF32> channelOffsets;
                for (NvU32 cc = 0; cc < (NvU32)networkDims.c; ++cc)
                {
                    channelOffsets.push_back(tensorOffset);
                }
                input_edge->originalTensor()->setChannelOffsets(channelOffsets);
            }
        }
      // 这个bindid好像只是整个图的input和output的edge才设定，这个函数只是设定两个变量而已
      // m_bindDomain = bindDomain; m_bindId = id; bingDomain有input output debug三种
      // 这个bindid也是随着inputedge的增加顺序往后排
        input_edge->setBindId(input_edges.size(), IOD_Input);
        if ( eng_graph->debugBinding() )
        {
            gLogInfo << "EngineAST graph level input edge[" << input_edges.size() << "] is " << input_edge->id() << endl;
            gLogInfo << "input bind id: " << input_edge->bindId() << endl;
        }

        input_edges.push_back( input_edge );
    };

    // 设定整个eng_graph的inputedge列表为input_edges
    if ( input_edges.size() )
    {
        eng_graph->setInputEdges(input_edges);
    }

  //按照以上处理inputedge的方法，处理所有的outputedges
  for ( vector<canonical_ast::Edge *>::const_iterator coe = can_graph->outputEdges().begin();
            coe != can_graph->outputEdges().end(); ++coe)
    {
        engine_ast::Edge *output_edge = can_to_eng_edge_map[*coe];
        output_edge->originalTensor()->setDataFormat(profile->networkOutputDataFormat());

        Dims4 networkDims = output_edge->originalTensor()->getDimensions();
        if ( networkDims.n != (NvS32)profile->multiBatchSize() )
        {
            gLogWarning << "Overriding output multibatch size from " << networkDims.n << " to " << profile->multiBatchSize() << endl;
            networkDims.n = profile->multiBatchSize();
            output_edge->originalTensor()->setDimensions(networkDims);
        }

        output_edge->setBindId(output_edges.size(), IOD_Output);
        if ( eng_graph->debugBinding() )
        {
            gLogInfo << "EngineAST graph level output edge[" << output_edges.size() << "] is " << output_edge->id() << endl;
            gLogInfo << "output bind id: " << output_edge->bindId() << endl;
        }

        output_edges.push_back( output_edge );
    };
  //设定整个eng_graph的outputedge列表为output_edges

  if ( output_edges.size() )
    {
        eng_graph->setOutputEdges(output_edges);
    }
    // 打印所有eng_node的name，编号，以及对应的can_node的name
    // 同时打印每个eng_node的所有input output aux类型的edge
    //libnvdla<3> dc-conv-0/n-0/conv1:
    //libnvdla<3>      in e-0
    //libnvdla<3>      out e-11
    //libnvdla<3>      aux e-9
    //libnvdla<3> bias-0/n-1/conv1:
    //libnvdla<3>      in e-11
    //libnvdla<3>      out e-1
    //libnvdla<3>      aux e-10
    // cache input/output/aux edges of each node into their respective data ports
    if ( eng_graph->debugGraphDump() )
    {
        engine_ast::Graph::NodeSet engineNodes = eng_graph->nodes();
        engine_ast::Graph::NodeSetIterator eni = engineNodes.begin();
        for ( ; eni != engineNodes.end(); ++eni)
        {
            typedef std::vector<Edge*>::const_iterator ESI;

            std::string canNodeName;
            if ((*eni)->canonicalNode() == NULL)
            {
                canNodeName = "(No canonical node)";
            }
            else
            {
                canNodeName = (*eni)->canonicalNode()->name();
            }
            gLogInfo << (*eni)->name() << "/" << (*eni)->id() << "/"
                     << canNodeName << ":" << endl;
            for (ESI ii = (*eni)->inputEdges().begin(); ii != (*eni)->inputEdges().end(); ++ii)
                gLogInfo << "\tin " << (*ii)->id() << endl;
            for (ESI ii = (*eni)->outputEdges().begin(); ii != (*eni)->outputEdges().end(); ++ii)
                gLogInfo << "\tout " << (*ii)->id() << endl;
            for (ESI ii = (*eni)->auxEdges().begin(); ii != (*eni)->auxEdges().end(); ++ii)
                gLogInfo << "\taux " << (*ii)->id() << endl;
        }
    }

    eng_graph->ordering()->generate();
    eng_graph->markClean();

    // force N = 1 for all non-Aux tensors represented by non-bindable edges;
    // until we allow contiguous non-bindable tensors for multi-batch
    {
        engine_ast::Graph::EdgeSequence engineEdges = eng_graph->orderedEdges();
        for (engine_ast::Graph::EdgeSequenceIterator eei = engineEdges.begin(); eei != engineEdges.end(); ++eei)
        {
            if (!(*eei)->bindable() && !(*eei)->isAuxEdge() && (*eei)->originalTensor())
            {
                Dims4 nonBindableTensorDims = (*eei)->originalTensor()->getDimensions();
                if ( eng_graph->debugGraphDump() )
                {
                    if (nonBindableTensorDims.n != 1)
                        gLogInfo << "Forcing batch size '1' for non-bindable non-aux edge " << (*eei)->id() << endl;
                }
                nonBindableTensorDims.n = 1;
                (*eei)->originalTensor()->setDimensions(nonBindableTensorDims);
            }
        }
    }
    return eng_graph;

fail:
    return NULL;
}
```

```cpp
NvDlaError engine_ast::transformCanNode
(
    engine_ast::Graph* engGraph,
    canonical_ast::Node *canNode,
    engine_ast::Graph::EdgeSequence engSrcEdges,
    engine_ast::Graph::EdgeSequence engSinkEdges,
    engine_ast::Graph::NodeSequence& transformedEngNodes
)
{
    NvDlaError e = NvDlaSuccess;
switch (canNode->canonicalOpType().v())
{
    case canonical_ast::CONVOLUTION:
        PROPAGATE_ERROR_FAIL(transformCanConvOp(engGraph, canNode, engSrcEdges, engSinkEdges, transformedEngNodes)); break;
    case canonical_ast::FULLY_CONNECTED:
        PROPAGATE_ERROR_FAIL(transformCanFCOp(engGraph, canNode, engSrcEdges, engSinkEdges, transformedEngNodes)); break;
    case canonical_ast::ACTIVATION:
        PROPAGATE_ERROR_FAIL(transformCanActOp(engGraph, canNode, engSrcEdges, engSinkEdges, transformedEngNodes)); break;
    case canonical_ast::POOLING:
        PROPAGATE_ERROR_FAIL(transformCanPoolingOp(engGraph, canNode, engSrcEdges, engSinkEdges, transformedEngNodes)); break;
    case canonical_ast::LRN:
        PROPAGATE_ERROR_FAIL(transformCanLRNOp(engGraph, canNode, engSrcEdges, engSinkEdges, transformedEngNodes)); break;
    case canonical_ast::SCALE:
        PROPAGATE_ERROR_FAIL(transformCanScaleOp(engGraph, canNode, engSrcEdges, engSinkEdges, transformedEngNodes)); break;
    case canonical_ast::BATCH_NORM:
        PROPAGATE_ERROR_FAIL(transformCanBNOp(engGraph, canNode, engSrcEdges, engSinkEdges, transformedEngNodes)); break;
    case canonical_ast::SOFTMAX:
        PROPAGATE_ERROR_FAIL(transformCanSoftMaxOp(engGraph, canNode, engSrcEdges, engSinkEdges, transformedEngNodes)); break;
    case canonical_ast::DECONVOLUTION:
        PROPAGATE_ERROR_FAIL(transformCanDeconvOp(engGraph, canNode, engSrcEdges, engSinkEdges, transformedEngNodes)); break;
    case canonical_ast::CONCATENATION:
        PROPAGATE_ERROR_FAIL(transformCanConcatOp(engGraph, canNode, engSrcEdges, engSinkEdges, transformedEngNodes)); break;
    case canonical_ast::ELEMENTWISE:
        PROPAGATE_ERROR_FAIL(transformCanEWOp(engGraph, canNode, engSrcEdges, engSinkEdges, transformedEngNodes)); break;
    case canonical_ast::SPLIT:
        PROPAGATE_ERROR_FAIL(transformCanSplitOp(engGraph, canNode, engSrcEdges, engSinkEdges, transformedEngNodes)); break;
    default:
         ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unexpected canonical node '%s' of type '%s' ", canNode->id().c_str(), canNode->canonicalOpType().c_str());
}
fail:
    return e;
}
```


### 4. Compiler 量化原理

<div class="info">
  在C程序中，可以用宏代码提高执行效率。宏代码本身不是函数，但使用起来象函数。预处理器用复制宏代码的方式代替函数调用，省去了参数压栈、生成汇编语言的CALL调用、返回参数、执行return等过程，从而提高了速度。 
  但使用宏代码最大的缺点是容易出错，预处理器在复制宏代码时常常产生意想不到的边际效应。
</div>

```cpp
#define PRECISION_SWITCH(modelPrec, computePrec, retVal, func, ...)     \
    switch(modelPrec) {                                                 \
        case nvdla::DataType::INT8:                                     \
        switch(computePrec) {                                           \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8:   \
                retVal = func<NvS8, NvS8>(__VA_ARGS__); break;          \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16:  \
                retVal = func<NvS8, NvS16>(__VA_ARGS__); break;         \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16:   \
                retVal = func<NvS8, half_float::half>(__VA_ARGS__); break;          \
            default:                                                    \
            REPORT_ERROR(NvDlaError_NotSupported, "Don't support %d", computePrec);      \
        }; break;                                                       \
        case nvdla::DataType::INT16:                                    \
        switch(computePrec) {                                           \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8:   \
                retVal = func<NvS16, NvS8>(__VA_ARGS__); break;         \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16:  \
                retVal = func<NvS16, NvS16>(__VA_ARGS__); break;        \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16:   \
                retVal = func<NvS16, half_float::half>(__VA_ARGS__); break;         \
            default:                                                    \
            REPORT_ERROR(NvDlaError_NotSupported, "Don't support %d", computePrec);      \
        }; break;                                                       \
        case nvdla::DataType::HALF:                                     \
        switch(computePrec) {                                           \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8:   \
                retVal = func<half_float::half, NvS8>(__VA_ARGS__); break;          \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16:  \
                retVal = func<half_float::half, NvS16>(__VA_ARGS__); break;         \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16:   \
                retVal = func<half_float::half, half_float::half>(__VA_ARGS__); break;          \
            default:                                                    \
            REPORT_ERROR(NvDlaError_NotSupported, "Don't support %d", computePrec);      \
        }; break;                                                       \
        case nvdla::DataType::FLOAT:                                    \
        switch(computePrec) {                                           \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT8:   \
                retVal = func<NvF32, NvS8>(__VA_ARGS__); break;         \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_INT16:  \
                retVal = func<NvF32, NvS16>(__VA_ARGS__); break;        \
            case surface::SurfacePrecisionEnum::NVDLA_PRECISION_FP16:   \
                retVal = func<NvF32, half_float::half>(__VA_ARGS__); break;         \
            default:                                                    \
            REPORT_ERROR(NvDlaError_NotSupported, "Don't support %d", computePrec);      \
        }; break;                                                       \
        default:                                                        \
        REPORT_ERROR(NvDlaError_NotSupported, "Don't support %d", modelPrec);            \
    }
```

具体的量化套路在 preProcessAuxData 和 quantizeAuxData 两个函数中：

```cpp
template <typename MP, typename CP>
static std::vector<NvF32> perKernelQuantizeWts
(
    Weights highPrecWts,
    NvS32 G, NvS32 K, NvS32 C, NvS32 RS, NvS32 kStride, NvS32 cStride,
    NvS8* quantizedWts
)
{
    std::vector<NvF32> filterScales;
    NvF32 max = std::numeric_limits<NvF32>::lowest(); // 先把 max 初始化为 FLoat 32 的最小值
    const MP* origWts = reinterpret_cast<const MP*>(const_cast<void*>(highPrecWts.values));
    // 获取所有Tensor的最大值
    for (NvS32 g = 0; g < G; g++)
    {
        NvS32 gOffset = g * K * C * RS;
        for (NvS32 k = 0; k < K; k++)
        {
            for (NvS32 c = 0; c < C; c++)
            {
                for (NvS32 rs = 0; rs < RS; rs++)
                    // max = std::max(max, std::abs(origWts[gOffset + k * kStride + c * cStride + rs] * inScale));
                    max = std::max<NvF32>(max, std::fabs(origWts[gOffset + k * kStride + c * cStride + rs]));
            }
        }
    }

    NvF32 scale = max / 127, invScale = 1 / scale;
    // 开始量化
    for (NvS32 g = 0; g < G; g++)
    {
        NvS32 gOffset = g * K * C * RS;
        for (NvS32 k = 0; k < K; k++)
        {
            for (NvS32 c = 0; c < C; c++)
            {
                for (NvS32 rs = 0; rs < RS; rs++)
                {
                    NvS32 index = gOffset + k * kStride + c * cStride + rs;
                    // quantizedWts[index] = int8_t(std::floor(origWts[index] * inScale * invScale + 0.5f));

                    // quantizedWts[index] = static_cast<NvS8>(std::floor(origWts[index] * invScale + 0.5f));
                    NvS32 int32Weight = static_cast<NvS32>(std::floor(origWts[index] * invScale + 0.5f));
                    quantizedWts[index] = static_cast<NvS8>(std::max(std::min(int32Weight, static_cast<NvS32>(std::numeric_limits<NvS8>::max())),
                                                            static_cast<NvS32>(std::numeric_limits<NvS8>::lowest())));
                }
            }
            filterScales.push_back(scale);
        }
    }
    return filterScales;
}
```

$$
Scale = \frac{max}{127} \\
invScale = \frac{1}{Scale} = \frac{127}{max} \\
Tensor_{int32} = floor(Tensor_{float32} * invScale + 0.5) = floor(\frac{Tensor_{float32}}{max} * 127 + 0.5) \\
Tensor_{int8} = max(min(Tensor_{int32}, 127), -128) 
$$



## 待解决的问题

1. 量化前后的数据输出对比。
3. symbol table (符号表)有什么用？
3. Graph 计分牌机制？
4. raw weights of bias 到底是权重数据还是bias数据

