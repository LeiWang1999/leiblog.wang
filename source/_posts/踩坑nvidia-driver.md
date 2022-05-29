---
title: 踩坑nvidia driver
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2022-05-29 00:04:37
---

![Ночной дозор by Foto Vishnya / 500px | Облака, Пруды, Зеркало](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/61a60c4d2b21a337fe62a12842ddaffc.jpg)

刚开始碰到的问题是这样的：在Azure上开的一台HPC（4块 V100 16G）在运行了大概七八个小时之后，nvidia的显卡会挂掉，具体的表现为`nvidia-smi`会卡住十几分钟，之后输出`No devices were found`，但是执行`lspci | grep -i nvidia`还是可以看到四块显卡好好的挂在上面，这种情况应该直接reboot就可以修复，但是reboot了之后同样的程序运行一段时间之后显卡还是会掉。

<!-- more -->

最后根据分析，是因为没有开启GPU的Persistence Mode。

### NVIDIA DRIVER PERSISTENCE

查看Nvidia的文档 [Driver Persistence](https://docs.nvidia.com/pdf/Driver_Persistence.pdf)：

> Under Linux systems where X runs by default on the target GPU the kernel mode driver will generally be initalized and kept alive from machine startup to shutdown, courtesy of the X process. On headless systems or situations where no long-lived X-like client maintains a handle to the target GPU, the kernel mode driver will initilize and deinitialize the target GPU each time a target GPU application starts and stops. In HPC environments this situation is quite common. Since it is often desireable to keep the GPU initialized in these cases, NVIDIA provides two options for changing driver behavior: Persistence Mode (Legacy) and the Persistence Daemon.

一般的机器上安装GPU，GPU的驱动程序会在机器的开启时被加载，机器关闭时再被卸载。而在在没有显示器的Linux操作系统(headless systems)中，尤其是HPC中非常常见，GPU的驱动程序会随着GPU运行的程序开始的时候自动被加载，程序关闭时自动被卸载，NVIDIA提供了两种方法来设置GPU的Persistence Mode，我们使用这一种：`sudo nvidia-smi -pm 1`。

开启了该模式之后，GPU的响应速度会变快，但是待机功耗会增加一点。

开启Persistence Mode之前使用nvidia-smi，输入命令之后需要等到四到五秒加载驱动程序:

```
Thu May 26 03:11:57 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.191.01   Driver Version: 450.191.01   CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00005B0A:00:00.0 Off |                  Off |
| N/A   31C    P0    35W / 250W |      0MiB / 16160MiB |      2%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-PCIE...  Off  | 00009E65:00:00.0 Off |                  Off |
| N/A   33C    P0    36W / 250W |    309MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-PCIE...  Off  | 0000B111:00:00.0 Off |                  Off |
| N/A   33C    P0    34W / 250W |    309MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-PCIE...  Off  | 0000BD71:00:00.0 Off |                  Off |
| N/A   32C    P0    35W / 250W |    614MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
```

开启Persistence Mode之后使用nvidia-smi，输入命令之后立刻产生输出，并且可以看到`Persistence-M`这里从off变成了ON,但没有运行程序的时候功耗增加了几瓦:

```
Sun May 29 03:04:47 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00005B0A:00:00.0 Off |                  Off |
| N/A   33C    P0    41W / 250W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-PCIE...  On   | 00009E65:00:00.0 Off |                  Off |
| N/A   34C    P0    42W / 250W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-PCIE...  On   | 0000B111:00:00.0 Off |                  Off |
| N/A   34C    P0    39W / 250W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-PCIE...  On   | 0000BD71:00:00.0 Off |                  Off |
| N/A   33C    P0    40W / 250W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

**为什么不开启Persistence Mode，GPU会掉卡？**

根据我的分析，因为我在运行的程序是Antares/Ansor，这个程序不是一个单一的GPU Application，而是生成数以万计的GPU Application去验证性能，则在每一个Application运行的过程中，GPU的驱动都需要被反复加载和卸载，一方面会损失很多性能，另一方面driver频繁卸载加载，GPU频繁被初始化，CPU访问PCIe config registers时间过长导致 softlock，从而造成GPU的死机。(引用自：https://bbs.gpuworld.cn/index.php?topic=10353.0)

在开启了Persistence Mode之后，机器正常运行Antares/Ansor程序一天多，还没有掉卡，问题应该是解决了。

### 其他踩坑记录

可恶的是一开始并没有意识到是Persistence Mode的问题，以为是别的问题导致掉卡，踩了无数坑，而且需要复现这个错误需要让程序跑上个几个小时，又比较花费时间。

#### 1. 可能是GPU散热出问题了？

网络上看到的情况：https://zhuanlan.zhihu.com/p/375331159

因为机箱的风扇损坏，导致散热不够，GPU的温度过高而强制关闭了，为了排除这个问题，我写了一个脚本，每隔一分钟记录nvidia-smi的输出，再次运行程序，脚本如下：

```bash
while true
do 
timestamp=$(date +%s)
nvidia-smi > ./nvidia-smi-logs/nvidia-smi-${timestamp}.log
sleep 60s
done
```

在几个小时之后，显卡不负众望的掉卡了，但是查看log发现显卡在掉卡前温度一切正常，于是这种情况被PASS了。

#### 2. 可能是显卡坏了?

leader分析可能是显卡出了问题，帮忙redeploy了机器, 这个过程中会重新换几张新显卡（`lspci | grep -i nvidia`能看到显卡的id）。

> 这里要吐槽一下Azure，redeploy把一块硬盘也给换了，当时mentor和我说这块硬盘有风险，原来是在这里等着我，跑了两个星期的实验数据和写的代码为了省空间都挂在这块硬盘上的，差点破防了，还好我足够冷静，在vscode的cache里把所有的代码都翻了出来，而数据可以再跑。

但是，程序运行了几个小时之后，显卡还是掉了。。

#### 3.只可能是驱动问题了吧？

硬件没有问题，那就只能是软件的问题了，涉及到驱动会非常折磨，因为每次更改驱动之后都需要重启机器，而这个过程需要知会一下负责机器的leader，这几天叨扰了他七八次，就只是为了帮忙重启一下机器，非常抱歉！

最后一次掉卡的`nvidia-smi`的记录里，使用到的驱动型号是**460**.**32**.**03**,于是我尝试：

1. 将驱动回退一个版本: **450**.**191**.**01** 

   问题重现..

2. 将驱动回退到ubuntu-drivers recommend的版本：**440**.**118**.**02**

   问题重现..

3. 将驱动往后更新一个版本：**465**.**19**.**01**

   问题重现...

4. 索性将驱动回退到Azure的HPC默认的版本：**390**.**116** 

   这个时候我感觉已经不是GPU驱动的问题了，将GPU的Persistence Mode开启，程序运行了超过八个小时GPU仍然没有掉卡，我就意识到应该是驱动反复加载的问题。

5. 察觉到问题的核心之后，将驱动还原到**460**.**32**.**03**

这里记录一下怎样安装和卸载Nvidia驱动，我的机器是ubuntu16.04的操作系统，与高版本的ubuntu安装驱动的姿势稍有不同。

##### 卸载驱动

我的建议是，把所有和nvidia驱动有关的东西都删干净了，使用该命令进行卸载：

```bash
sudo apt-get --purge remove "*cublas*" "cuda*"
sudo apt-get --purge remove "*nvidia*"
sudo apt autoremove
```

这个时候，nvcc，nvidia-smi这些命令就都找不到了。

##### 安装驱动

安装驱动有多种方法：

1. 通过 apt 自动安装，不过这种情况下要安装新版的驱动要自己手动下载一个deb装上。
2. 官方下载驱动程序安装安装脚本。
3. 直接安装cuda，在安装的过程中会提示是否安装对应的驱动。

方法2、3实践起来问题多多，我这里使用第一种方案。

首先，安装`ubuntu-drivers-common`

```bash
sudo apt-get install ubuntu-drivers-common
sudo ubuntu-drivers devices
**
vendor   : NVIDIA Corporation
driver   : nvidia-410 - third-party free
driver   : nvidia-418 - third-party free
driver   : nvidia-415 - third-party free
driver   : nvidia-384 - distro non-free
driver   : nvidia-430 - third-party free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

如果上述的驱动有你想安装的，比如`nvidia-430`直接

```bash
sudo apt-get install nvidia-430
```

就可以安装完成nvidia-430的驱动，但是我要安装的nvidia-460不在这里面，就需要我们手动打个补丁了。

因为高版本的cuda都有最低版本的驱动要求，所以驱动还是安装的版本高一些比较好。

前往[DATA CENTER DRIVER FOR UBUNTU 16.04](https://www.nvidia.com/download/driverResults.aspx/169401/en-us)下载驱动：

```bash
wget https://us.download.nvidia.com/tesla/460.32.03/nvidia-driver-local-repo-ubuntu1604-460.32.03_1.0-1_amd64.deb
sudo dpkg -i nvidia-driver-local-repo-ubuntu1604-460.32.03_1.0-1_amd64.deb
```

再次查看支持的driver:

```bash
$ ubuntu-drivers devices
**
vendor   : NVIDIA Corporation
driver   : nvidia-418 - third-party free
driver   : nvidia-415 - third-party free
driver   : nvidia-430 - third-party free
driver   : nvidia-410 - third-party free
driver   : nvidia-384 - distro non-free
driver   : nvidia-460 - third-party non-free recommended
driver   : xserver-xorg-video-nouveau - distro free builtin
```

安装完成驱动之后，`nvidia-smi`这个命令就可以使用了，但是还是会报错`Failed to initialize NVML: Driver/library version mismatch`,这是因为重新安装了显卡驱动之后需要重启一下系统，才可以正常work。

##### 安装cuda

但是因为之前把cuda也都一起卸载了，应用程序跑不起来，需要我们重新安装一下cuda.

```bash
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
```

因为cuda安装程序自带对应cuda版本最低要求的驱动，所以他会抛出一个检测到已安装驱动的异常，直接continue，并且在安装包选择的页面里把驱动的包去除掉，这样cuda版本就可以正常work了，并且会在home目录下安装一份cuda sample代码，编译以测试cuda能否正常工作。

```bash
~/NVIDIA_CUDA-10.1_Samples/0_Simple/vectorAdd$ ./vectorAdd
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 196 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Done
```

##### nvidia-docker

虽然在主机上cuda程序已经可以正常工作了，但是在启动docker的时候还是会出现提示，` no CUDA-capable devices were detected`，在docker里运行cuda程序还是不行,运行`nvidia-smi`的时候，cuda verison 显示为 N/A。

```bash
Sun May 29 05:59:31 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: N/A     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  On   | 00005B0A:00:00.0 Off |                  Off |
| N/A   31C    P0    24W / 250W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-PCIE...  On   | 00009E65:00:00.0 Off |                  Off |
| N/A   34C    P0    41W / 250W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-PCIE...  On   | 0000B111:00:00.0 Off |                  Off |
| N/A   34C    P0    39W / 250W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-PCIE...  On   | 0000BD71:00:00.0 Off |                  Off |
| N/A   32C    P0    40W / 250W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

尝试重启docker服务也没有作用。

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

我在容器里编译程序，检查程序的链接库，发现在` /usr/lib/x86_64-linux-gnu/`目录下，有这么诡异的一幕：

```bash
cd /usr/lib/x86_64-linux-gnu/
ls -al | grep libcuda
lrwxrwxrwx 1 root root         12 May  5 12:22 libcuda.so -> libcuda.so.1
lrwxrwxrwx 1 root root         20 May 28 15:49 libcuda.so.1 -> libcuda.so.465.19.01
-rw-r--r-- 1 root root          0 May  5 12:22 libcuda.so.465.19.01
```

死去的driver 465正在攻击我，为什么这个image还是用的465的驱动，检查主机上的配置：

```bash
ls -al | grep libcuda
lrwxrwxrwx   1 root root        12 Dec 29  2020 libcuda.so -> libcuda.so.1
lrwxrwxrwx   1 root root        20 May 28 15:35 libcuda.so.1 -> libcuda.so.460.32.03
-rw-r--r--   1 root root  21803296 Dec 27  2020 libcuda.so.460.32.03
```

幸好在学校里还有一台可以正常work的机器来做为对照组，`/usr/lib/x86_64-linux-gnu/`下的`libcuda.so` , `libcuda.so.1`,`libcuda.so.1.{driver version}`，我做了很多实验，发现docker里起的容器的driver版本总会严格的和宿主机一致，不管起的容器是用的什么cuda版本和什么cudnn版本、什么ubuntu版本。

通过`nvidia-container-cli -k -d /dev/tty info`这个命令，发现了`could not start driver service: load library failed: libnvidia-fatbinaryloader.so.465.19.01: cannot open shared object file: no such file or directory`这样一条日志，但是奇怪的是，我查看对应driver的libcuda，也是安装了的。

```bash
sudo apt search libcuda
***
libcuda1-460/unknown,now 460.32.03-0ubuntu1 amd64 [installed]
  NVIDIA CUDA runtime library
***
```

最后，凭借我的理解，我认为可能是安装了但是动态链接库没有update，于是试了一下

```bash
sudo ldconfig
```

再起docker，就可以正常检测到gpu了。

有点回忆起以前的时候运维学校的机器的时候，出了问题要紧急响应的紧迫感，但还是花了一两天才cover了所有的问题，非常理解做运维的同学的辛苦了，尤其是在解决一个问题又马上蹦出下一个新问题的时候，真的是头到要炸，而且在很多情况下都需要保持冷静，分析问题，如果不能保持冷静的话，之前的工作都有可能功亏一篑。

