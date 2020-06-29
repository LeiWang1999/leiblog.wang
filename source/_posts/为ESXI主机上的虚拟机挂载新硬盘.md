---
categories:
  - Technical
tags:
  - ESXI
date: 2020-04-04 16:35:47	
---

最近我维护的某内网网站，已经吃了 20 个 T 的资源，现在面临存储不够的问题。好在上个学期老师又给了我们 20 个 T 的存储，现在是时候挂载到服务器上了，这篇文章则记录了我挂载的过程。

我们的站点，存在着多个安装在 ESXI 系统中的虚拟机作为文件服务器，现在对其中一个进行硬盘的扩容。

<!-- more -->

##### 一、编辑虚拟机配置、选择添加硬件

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404163018877.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70#pic_center)

##### 二、类型选择硬盘

![](https://img-blog.csdnimg.cn/2020040416303085.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70#pic_center)

##### 三、选择创建新的虚拟磁盘

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404163103131.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70#pic_center)

##### 四、配置磁盘容量等信息

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020040416311378.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70#pic_center)

磁盘置备选项说明：

- 置备延迟置零（zeroed thick）：以默认的厚格式创建虚拟磁盘。创建过程中为虚拟磁盘分配所需空间。创建时不会擦除物理设备上保留的任何数据，但是以后从虚拟机首次执行写操作时会按需要将其置零。**也就是说立刻分配指定大小的空间，空间内数据暂时不清空，以后按需清空。**
- 厚置备置零（eager zeroed thick）：创建支持群集功能（如 Fault Tolerance）的厚磁盘。在创建时为虚拟磁盘分配所需的空间。与平面格式相反，在创建过程中会将物理设备上保留的数据置零。创建这种格式的磁盘所需的时间可能会比创建其他类型的磁盘长。**也就是说立刻分配指定大小的空间，并将该空间内所有数据清空。**
- 精简置备（Thin）：使用精简置备格式。最初，精简置备的磁盘只使用该磁盘最初所需要的数据存储空间。如果以后精简磁盘需要更多空间，则它可以增长到为其分配的最大容量。**也就是说是为该磁盘文件指定增长的最大空间，需要增长的时候检查是否超过限额。**

指定数据存储或者数据存储集群、浏览的时候选择空间足够的存储

##### 五、配置高级选项

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020040416312640.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70#pic_center)

虚拟设备节点用默认的就好

因为我这里存的文件是要长久保存的，所以独立模式的持久选项勾选上。

##### 六、回顾配置、完成

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404163136534.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70#pic_center)

然后 vsphere 的任务台就会开始格式化挂载的硬盘空间，由于我这里要挂载的是 4TB 的硬盘，所以花了半小时左右的时间。

##### 七、在虚拟机里查看挂载的硬盘情况

```bash
df -lh # 查看已经挂载的磁盘
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404163151185.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70#pic_center)

```bash
sudo fdisk -l # 查看所有磁盘
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404163203202.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70#pic_center)
可以看到 sdf 没有被挂载，并且大小与我们刚才一致、其实为了稳妥起见，我还特地去/dev/下面看了 sdf 的创建时间，不过忘了截图了。总之，我们要将 sdf 挂载到我们需要的目录下。

##### 八、新建目录用于挂载

```bash
sudo mkdir /mnt/mounted_dir_name # 创建你的文件夹，将mounted_dir_name替换成你的目录名
sudo chown -R username:username /mnt/mounted_dir_name # 将此处的username替换成你的用户名
```

##### 九、格式化磁盘，挂载到目录

如果你希望将你心挂载的磁盘进行分区，那么你可能要先进行分区的操作。有关操作可以百度，我这里直接将整个 4TB 的硬盘格式化了。

```bash
sudo mkfs -t ext4 /dev/sdf # 将硬盘格式化为ext4的格式
```

也正是由于 ext4 格式、导致最大只能挂在 4TB 的硬盘，不过这足够大了。

格式化的速度很快，然后将格式化过的磁盘挂载目录就好了：

```bash
sudo mount -t ext4 /dev/sdf /mnt/mounted_dir_name # 挂载磁盘
```

##### 十、配置系统启动的时候自动挂载

```bash
sudo vim /etc/fstab # 编辑配置文件
```

在最后一行插入

```bash
/dev/sdf     /mnt/mounted_dir_name    ext4     defaults       0 0
```

然后退出 vim，如果你不知道怎么退出，按 esc，然后按住 shift 不放手，按两下 z，松开 shift。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404163216853.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70#pic_center)

**挂载完成！**
