---
title: Train AlexNet with PyTorch and Inference with TVM
top: 10
categories:
  - Technical
tags:
  - Pytorch
date: 2020-10-04 12:12:45
---

在这篇博客，我使用Pytorch构建了一个AlexNet网络，通过flower_data数据集训练AlexNet识别各种花朵，然后我分别使用pytorch直接推理网络，以及使用tvm编译过后的模型推理网络，对比了两者的输出，以及运行的速度。

Github Page:https://github.com/LeiWang1999/AlexNetTVM

![AlexNet Banner](http://leiblog.wang/static/image/2020/10/TL5kcp.jpg)

<!-- more -->

### AlexNet brief Introduction

AlexNet是2012年ImageNet竞赛冠军获得者Hinton和他的学生Alex Krizhevsky设计的。也是在那年之后，更多的更深的神经网络被提出，比如优秀的vgg，GoogLeNet。AlexNet中包含了几个比较新的技术点，也首次在CNN中成功应用了ReLU、Dropout和LRN等Trick。

有关网络的模型是咋样的，可以去读Alex的原论文，相比于现在的模型，AlexNet的模型还是比较简单的。我比较推荐一个B站宝藏UP主的讲解视频，训练代码也是抄的他的。

- [AlexNet网络结构详解与花分类数据集下载](https://www.bilibili.com/video/BV1p7411T7Pc)

这一小节主要就是为了凑字数，话说不会真的有人不知道AlexNet是啥吧？

### 训练AlexNet

#### 准备数据集

1. 下载flower_data: http://download.tensorflow.org/example_images/flower_photos.tgz

2. 解压到工程文件夹，注意文件夹名字是 flower_data
3. 运行"split_data.py"脚本自动将数据集划分成训练集train和验证集val

运行train.py就会进行AlexNet的训练了，模型定义在model.py里。

训练完成后，取在val集上表现最好的model保存成文件，这里我用了两种保存方式。

```python
torch.save(model.state_dict(), 'AlexNet_weights.pth')
torch.save(model, 'AlexNet.pth')
```

**有什么区别？**

在PyTorch中，模型的可学习参数(`model.parameters()`可以访问的）state_dict是一个简单的Python字典对象，每个层映射到其参数张量。只有具有可学习参数的层(卷积层，线性层等）和已注册的缓冲区(batchnorm的running_mean）才在模型的state_dict中具有条目。优化器对象(`torch.optim`）还具有state_dict，其中包含有关优化器状态以及所用超参数的信息。由于 state_dict 对象是Python词典，因此可以轻松地保存，更新，更改和还原它们，从而为PyTorch模型和优化器增加了很多模块化。

直接保存整个模型是保存/加载过程使用最直观的语法，并且涉及最少的代码。以这种方式保存模型将使用Python的pickle模块保存整个模块。这种方法的缺点是序列化的数据绑定到特定的类，并且在保存模型时使用确切的目录结构。这样做的原因是因为pickle不会保存模型类本身。而是将其保存到包含类的文件的路径，该路径在加载时使用。

总之，我喜欢用第一种，限制条件是导入的时候要给出模型。

### AlexNet推理

两种推理方式，没什么要说的。需要注意的是TVM的relay前端，接受的不是普通的torch.nn.Module对象，而是ScriptModules。转换函数在pytorch官方也给出了。

```python
scripted_model = torch.jit.trace(model, img).eval()
```

计时比较

```python
start = time.perf_counter()
# runing code
elapsed = (time.perf_counter() - start)
```

#### Pytorch推理

```zsh
Time used: 0.019670627999999857
tensor([-2.5184,  0.9804, -2.7127,  2.6251, -0.5793])
sunflowers 0.8036765456199646
```

#### TVM推理

```zsh
Time used: 0.011482629000000522
tensor([0.0047, 0.1552, 0.0039, 0.8037, 0.0326])
sunflowers tensor(0.8037)
```

速度果然提升了，但这个结果和我之前测的结果还不太一样，之前测的时候TVM的推理速度可能要快过Pytorch几十倍，也许是我记错了。。