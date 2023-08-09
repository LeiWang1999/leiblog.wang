---
title: 李宏毅CNN作业的笔记-食物分类
categories:
  - Technical
tags:
  - Tensorflow
  - CNN
  - Machine Learning
date: 2020-04-11 20:56:39	
---

第一次做 MachineLearning 的笔记，因为实在是碰到问题了。最近跟着李宏毅老师学习到了很多知识，课程地址：[点击前往](https://www.bilibili.com/video/BV1JE411g7XF)

第三个作业是 CNN。但是课程网站里的 ExampleCode 给的是 Pytorch 版本的，而我用的是 Tensorflow，懒得再安装 pytorch 环境运行，所以从头到尾自己用 keras 写了一下，在这篇文章里我记录一下我出现的一些问题。

<!-- more -->

### 作业描述

其实当作业给的数据集有 1.0GBytes 的时候，我就知道问题没这么简单。

![[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-HRO6LbAA-1586609341890)(/Users/wanglei/Desktop/Study Monitor/作业/简书/李宏毅CNN作业的笔记-食物分类/1.png)]](https://img-blog.csdnimg.cn/20200411204915691.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

其中，**training**文件夹中有一万张图片，**validation**文件夹中有 3000 张验证图片。

![[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-Ii6WARFt-1586609341892)(/Users/wanglei/Desktop/Study Monitor/作业/简书/李宏毅CNN作业的笔记-食物分类/2.png)]](https://img-blog.csdnimg.cn/20200411204935627.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

他们的文件名数字开头就是他们的类型，我们需要分类图片中的食物类型，一万张图片，共 11 种食物

### 读取数据

```python
def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x
workspace_dir = './data'
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))
```

这一步是照搬的 ExampleCode 的函数式，不是本次的重点。

### 设计 CNN

先看代码：

```python
model = keras.models.Sequential()
# the raw shape (128,128,3)
# padding: same：对边缘补0 valid：对边缘不补0
model.add(keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu', input_shape=(128,128,3)))
# This time the shape is (32,128,128,3)
# MaxPooling
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# （32,64,64,3）
model.add(keras.layers.Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#  (64,32,32,3)
model.add(keras.layers.Flatten())
# Flatten and input Neural NetWOrk; set activation = selu(auto_batch_normalization)
# 20 layers; 100 nodes
for _ in range(20):
    model.add(keras.layers.Dense(100,activation='selu'))

model.add(keras.layers.Dense(11, activation='softmax'))
model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam',
              metrics = ['accuracy'])
```

**image data**先经过一次**Convolution**，我设计了 32 个**filter**，每个是（3，3）的矩阵，至于为什么选这个数，大家都喜欢用（具体原因我还在研究[这篇文章](https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/)），再经过一层**MaxPooling**，这里也用常用的（2，2），再来一组这样的操作，只不过这时候的**filter**的个数变成了原来的 2 倍。

然后对输出的数据进行**Flatten**之后输入到**Neural Network**里面去。网络一共 20 个**layer**，每个**layer**有 100 个**Node**。**Activation Function**选择 selu 是为了自动批归一化，提高准确率。

然后因为是分类问题，并且输入的分类 target 的值是 1、2、3、4…所以选了**sparse_categorical_crossentropy**作为**loss function**

### 训练网络

#### 第一次训练

```python
logdir = './dnn_callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,
                                "cnn_classfication_model.h5")
callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                   save_best_only=True),
]
history = model.fit(train_x, train_y, epochs=10,
                   callbacks=callbacks)
```

加入了两个 callback 是为了调试、在第一次训练的时候，我没有考虑 validation 数据集，我想用验证集来验证训练出来的网络的正确性，方便一些。

并且由于我用的设备 MacBookPro 没有支持 tensorflow-gpu 的外设，所以 epochs 只设置了 10 次，不过这也让我电脑运行了十几分钟。

在 trainning 过程中，表现优秀，正确率达到 0.87

![[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-NMDqQIoq-1586609341893)(/Users/wanglei/Desktop/Study Monitor/作业/简书/李宏毅CNN作业的笔记-食物分类/5.png)]](https://img-blog.csdnimg.cn/20200411205008409.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

但是在 valiadtion 数据集上，只有 0.2 的正确率。。。

![在这里插入图片描述](https://img-blog.csdnimg.cn/202004112112531.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

好吧，像这种在 training data 上有好的 performance，但是在 testing data 上的 performance 很差，根据李宏毅老师的课，大概有三种解决办法，还有救：

- Early Stoping

  ![[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-YPvxExUv-1586609341894)(/Users/wanglei/Desktop/Study Monitor/作业/简书/李宏毅CNN作业的笔记-食物分类/4.png)]](https://img-blog.csdnimg.cn/20200411205031693.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

- Dropout

- Regularization（正则化）

#### 第二次训练 - 加上 Dropout

在 tensorflow2.x 中加上 Dropout 的方法就是新增一层**layer**。

创建模型时候的代码变称下面这样：

```python
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu', input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(keras.layers.Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(keras.layers.Flatten())
for _ in range(20):
    model.add(keras.layers.Dense(100,activation='selu'))
model.add(keras.layers.AlphaDropout(rate=0.5))	###### 这里是新增的dropout
model.add(keras.layers.Dense(11, activation='softmax'))
model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam',
              metrics = ['accuracy'])
```

除了新增了一行**AlphaDropout**的 layer，其他都不变。AlphaDropout 比普通的 Dropout 有更好的性质。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200411210119821.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

擦，excuse me，这平均 15%的正确率是怎么回事？

心存侥幸，我还是在验证集上验证了一下。

![[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-cIlS7fgj-1586609341895)(/Users/wanglei/Desktop/Study Monitor/作业/简书/李宏毅CNN作业的笔记-食物分类/8.png)]](https://img-blog.csdnimg.cn/20200411205055237.png)

哎，又失败了，既然**Dropout**不行，那就再加上**EarlyStopping**。

#### 第三次训练 - 加上 Early Stopping

但这个时候，我们需要放弃验证集来做**evaluate**了，因为**EarlyStopping**需要验证集做训练，所以我们要手动看 test 的**performance**了。你可能要问我，为什么不从**training data**中分出一部分用作验证集？很简单、我比较懒，我能够想到用**numpy**的**random**方法从验证集中随机取出一部分用作**trainning data**，另一部分用作**validation data**，原本的**validation**文件夹里的图片用作**testing data**。

但这也得我们加上**validation data**后，**train**出了好的**performance**才可以，事实证明，我不用这么白忙活。因为加上了**Early Stopping**，又出了问题。

在 tensorflow2.x 中加上**Early Stopping**的途径是设置**callbacks**。这里我把之前加的 Dropout 也删去了，因为本来加上**Dropout**就有非常**poor**的**performance**，加上应该会更低吧。

```python
logdir = './dnn_callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,
                                "cnn_classfication_model.h5")
callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                   save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3) # 这里是新增的内容
]
history = model.fit(train_x, train_y, epochs=10,
                   callbacks=callbacks)
```

我们再运行。

![[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-e7rniJJm-1586609341895)(/Users/wanglei/Desktop/Study Monitor/作业/简书/李宏毅CNN作业的笔记-食物分类/9.png)]](https://img-blog.csdnimg.cn/20200411205108184.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

Early Stopping 并没有发挥作用，也就是说，即使我们不加 Early Stopping，仅加上验证集，也就会得到这样的结果，真是蛋疼。

但应付 train 的 poor performance，我们还有两种解决办法：

- New Activation Function

  我认为，这个不需要考虑了吧，还有什么比 selu 更适合的呢。

- Adapative Learning Rate

  我认为，这个也不许需要考虑了吧，我用的优化器是 adam，还不够 Adapative 么。

综上，现在还没找到好的解决方案，也许是 CNN 网络的设计问题？或许是 epoch 的次数不够多。初学者，希望各位路人大佬能教教我。

如果你是在我的[个人博客](http://leiblog.wang/)看到的这篇文章，推荐点击最上方的**前往 CSDN 阅读**进行留言，根据相关法律法规，博客暂未开通评论功能。

另外还可以体验我用 JavaScript 在前端编写的线性回归小游戏：[点击前往](http://leiblog.wang/game)

### 后记

问题解决了，还是自己太年轻。现在通用的 CNN 网络，大部分都是自 Conv 做手脚，比如很多层 conv 加上两到三层 fully connected 这样的。而我这个网络是两到三层 conv 加上二十层 fully conntected。建议大家把我当作反面教材。
特别感谢来自 TG 的不知名网友 Jieming Zhou 的指点迷津。
