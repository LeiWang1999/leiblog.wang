---
title: NVDLA int8 量化笔记
categories:
  - Technical
tags:
  - NVDLA
  - EEEE
date: 2021-02-02 16:50:26
---

前一段时间，在NVDLA上针对MNIST、CIFAR-10、IMAGENET2012这三个数据集上，训练了lenet5、resnet-18两个网络，并在NVDLA的vp环境下编译，运行，相关的模型可以在下文下载。

而NVDLA的文档描述里写到其是支持8bit的，但是如何将浮点模型在nvdla上量化成8bit并运行官方给出的说明很少，本文记述的内容有模型量化的方法、以及修复了官方代码中的一些问题。

<!-- more -->

## Caffe 预训练的网路模型

1. https://leiblog.wang/static/2020-12-29/caffe_lenet_mnist.zip
2. https://leiblog.wang/static/2021-01-23/resnet18-cifar10-caffe.zip
3. https://leiblog.wang/static/2021-01-23/resnet18-imagenet-caffe.zip

## NVDLA量化

在Github的仓库里，有给到量化的大体方向：[LowPrecision.md](https://github.com/nvdla/sw/blob/master/LowPrecision.md)

文中指出，我们需要使用TensorRT来进行模型的量化，并且最后得到一个calibration table。而如何使用TensorRT进行量化的内容可以参考TensorRT的INT8量化方法，简单的说就是需要求一个浮点数到定点数缩放因子，而这个求法的算法我们可以仅作了解，这里给出一个知乎链接：https://zhuanlan.zhihu.com/p/58182172

具体的步骤如下：

### 使用TensorRT生成Calibration Table

我推荐使用官方的docker镜像

```bash
docker pull nvcr.io/nvidia/tensorrt:20.12-py3
```

> 其实原本我以为，单为了生成calibration table文件装个TensorRT会不会太麻烦，首先尝试使用了圈圈虫的项目：https://github.com/BUG1989/caffe-int8-convert-tools 但遗憾的是不work。
>
> 1. ./nvdla_compiler其实读取的并不是生成的calibration table文件(这是一个text文件)，而是一个json文件，转换的脚本是[calib_txt_to_json.py](https://github.com/nvdla/sw/tree/master/umd/utils/calibdata/calib_txt_to_json.py)，而该脚本并不支持虫叔的仓库生成的calibration table文件，仅支持tensorRT
> 2. 经过实践发现、虫叔那一套方案没有TensorRT的高级，TensorRT量化的时候需要从验证集中选择一组数据来做辅助矫正，更高级，也可以得出矫正过后的精度，及量化之后的模型的精度如何，是很有用的。

最新版的TensorRT做int8的量化是支持Python API的，我们可以在`/opt/tensorrt/samples/python/int8_caffe_mnist`里找到官方的量化方法，当然与之对应的还有C++版本。我更推荐Python、可读性更好、重写起来更方便，而且C++其实也是调用的.so的库。

如果我们想要重构一份自己的网络量化方法，核心的部分是要自己定义一个Class、继承`trt.IInt8EntropyCalibrator2`，具体内容可以阅读MNIST量化的`calibrator.py`文件，我们需要提供自己训练的时候使用的数据集，deploy.prototxt，caffemodel这三个文件，然后自己编写解析的方法，例如我自己的CIFAR10，就写成这样：

```python
# Returns a numpy buffer of shape (num_images, 1, 32, 32)
def load_cifar10_data(filepath, scale=False):
    with open(filepath, "rb") as f:
        dic = pickle.load(f, encoding='bytes')
        test_images = np.array(dic[b"data"].reshape([-1, 3, 32, 32])).astype(np.uint8)
        # test_labels = np.array(dic[b"labels"])
    # Need to scale all values to the range of [0, 1]
    if scale:
        return np.ascontiguousarray((test_images / 255.0).astype(np.float32))
    else:
        return np.ascontiguousarray(test_images).astype(np.float32)
# Returns a numpy buffer of shape (num_images)
def load_cifar10_labels(filepath):
    with open(filepath, "rb") as f:
        dic = pickle.load(f, encoding='bytes')
        test_labels = np.array(dic[b"labels"])
    # Make sure the magic number is what we expect
    return test_labels
```

而自己定义的`EntropyCalibrator`如下：

```python
class Cifar10EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, batch_size=32):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = load_cifar10_data(training_data, scale=False)
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]


    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
```

这四个方法都会在build_engine阶段自动调用，需要特别注意的是get_batch方法，在获取数据的时候需要使用np.ascontiguousarray展平，否则会导致失真严重。

运行程序之后的输出：

```zsh
root@ea6ecad42f26:/opt/tensorrt# python samples/python/int8_caffe_cifar10/sample.py 
Calibrating batch 0, containing 32 images
Calibrating batch 10, containing 32 images
Calibrating batch 20, containing 32 images
Calibrating batch 30, containing 32 images
Calibrating batch 40, containing 32 images
Calibrating batch 50, containing 32 images
Calibrating batch 60, containing 32 images
Calibrating batch 70, containing 32 images
Calibrating batch 80, containing 32 images
Calibrating batch 90, containing 32 images
Calibrating batch 100, containing 32 images
Calibrating batch 110, containing 32 images
Calibrating batch 120, containing 32 images
Calibrating batch 130, containing 32 images
Calibrating batch 140, containing 32 images
Calibrating batch 150, containing 32 images
Calibrating batch 160, containing 32 images
Calibrating batch 170, containing 32 images
Calibrating batch 180, containing 32 images
Calibrating batch 190, containing 32 images
Calibrating batch 200, containing 32 images
Calibrating batch 210, containing 32 images
Calibrating batch 220, containing 32 images
Calibrating batch 230, containing 32 images
Calibrating batch 240, containing 32 images
Calibrating batch 250, containing 32 images
Calibrating batch 260, containing 32 images
Calibrating batch 270, containing 32 images
Calibrating batch 280, containing 32 images
Calibrating batch 290, containing 32 images
Calibrating batch 300, containing 32 images
Calibrating batch 310, containing 32 images
Validating batch 10
Validating batch 20
Validating batch 30
Validating batch 40
Validating batch 50
Validating batch 60
Validating batch 70
Validating batch 80
Validating batch 90
Validating batch 100
Validating batch 110
Validating batch 120
Validating batch 130
Validating batch 140
Validating batch 150
Validating batch 160
Validating batch 170
Validating batch 180
Validating batch 190
Validating batch 200
Validating batch 210
Validating batch 220
Validating batch 230
Validating batch 240
Validating batch 250
Validating batch 260
Validating batch 270
Validating batch 280
Validating batch 290
Validating batch 300
Validating batch 310
Total Accuracy: 88.37%
```

而在运行目录下生成的：cache文件就是calibration table了：

```zsh
TRT-7202-EntropyCalibration2
data: 4000890a
(Unnamed Layer* 0) [Convolution]_output: 411e6870
(Unnamed Layer* 1) [Scale]_output: 3daa12ee
(Unnamed Layer* 2) [Scale]_output: 3d8400aa
first_conv: 3d7e1308
(Unnamed Layer* 4) [Convolution]_output: 3e229280
(Unnamed Layer* 5) [Scale]_output: 3d9d69c8
(Unnamed Layer* 6) [Scale]_output: 3d6d057b
group0_block0_conv0: 3d49be67
(Unnamed Layer* 8) [Convolution]_output: 3de9d077
(Unnamed Layer* 9) [Scale]_output: 3d7fbd36
group0_block0_conv1: 3d6d6b9e
(Unnamed Layer* 11) [ElementWise]_output: 3d9d08d7
group0_block0_sum: 3d9d08d7
(Unnamed Layer* 13) [Convolution]_output: 3e838a54
(Unnamed Layer* 14) [Scale]_output: 3d693378
(Unnamed Layer* 15) [Scale]_output: 3d65b956
group0_block1_conv0: 3d38eaf7
(Unnamed Layer* 17) [Convolution]_output: 3da93e34
(Unnamed Layer* 18) [Scale]_output: 3d851a08
group0_block1_conv1: 3d6d458c
(Unnamed Layer* 20) [ElementWise]_output: 3db0ccce
group0_block1_sum: 3daffc1b
(Unnamed Layer* 22) [Convolution]_output: 3e6dcba7
(Unnamed Layer* 23) [Scale]_output: 3d4fc028
(Unnamed Layer* 24) [Scale]_output: 3d3d5aa8
group0_block2_conv0: 3cfbae80
(Unnamed Layer* 26) [Convolution]_output: 3d21c535
(Unnamed Layer* 27) [Scale]_output: 3d4dbefc
group0_block2_conv1: 3d1467d6
(Unnamed Layer* 29) [ElementWise]_output: 3dad9307
group0_block2_sum: 3dae4e9d
(Unnamed Layer* 31) [Convolution]_output: 3e82246b
(Unnamed Layer* 32) [Scale]_output: 3d5e11ea
(Unnamed Layer* 33) [Scale]_output: 3d31c955
group1_block0_conv0: 3d2c5625
(Unnamed Layer* 35) [Convolution]_output: 3dd0a5ed
(Unnamed Layer* 36) [Scale]_output: 3d45d5dc
group1_block0_conv1: 3d36f71d
(Unnamed Layer* 38) [Convolution]_output: 3dbba076
(Unnamed Layer* 39) [Scale]_output: 3d4dd5ae
group1_block0_proj: 3d09cfd0
(Unnamed Layer* 41) [ElementWise]_output: 3d6cefd9
group1_block0_sum: 3d6cefd9
(Unnamed Layer* 43) [Convolution]_output: 3e1b7a5a
(Unnamed Layer* 44) [Scale]_output: 3d3666e5
(Unnamed Layer* 45) [Scale]_output: 3d1df171
group1_block1_conv0: 3ce1b3e8
(Unnamed Layer* 47) [Convolution]_output: 3d0db714
(Unnamed Layer* 48) [Scale]_output: 3d2a1f7b
group1_block1_conv1: 3cfafb12
(Unnamed Layer* 50) [ElementWise]_output: 3d7a722e
group1_block1_sum: 3d83067b
(Unnamed Layer* 52) [Convolution]_output: 3e1e4ee2
(Unnamed Layer* 53) [Scale]_output: 3d324c5b
(Unnamed Layer* 54) [Scale]_output: 3d16b2ca
group1_block2_conv0: 3cfc9c36
(Unnamed Layer* 56) [Convolution]_output: 3d11f6e1
(Unnamed Layer* 57) [Scale]_output: 3d63d99f
group1_block2_conv1: 3d14e1c6
(Unnamed Layer* 59) [ElementWise]_output: 3d82154b
group1_block2_sum: 3d8acc00
(Unnamed Layer* 61) [Convolution]_output: 3e2b9d99
(Unnamed Layer* 62) [Scale]_output: 3d461d92
(Unnamed Layer* 63) [Scale]_output: 3d0753a3
group2_block0_conv0: 3d088b96
(Unnamed Layer* 65) [Convolution]_output: 3d8f6929
(Unnamed Layer* 66) [Scale]_output: 3d46f2cf
group2_block0_conv1: 3d401129
(Unnamed Layer* 68) [Convolution]_output: 3d3b470d
(Unnamed Layer* 69) [Scale]_output: 3d337907
group2_block0_proj: 3c8b7185
(Unnamed Layer* 71) [ElementWise]_output: 3d4aa35f
group2_block0_sum: 3d4c01a2
(Unnamed Layer* 73) [Convolution]_output: 3da85a1e
(Unnamed Layer* 74) [Scale]_output: 3d506b89
(Unnamed Layer* 75) [Scale]_output: 3d2390a0
group2_block1_conv0: 3d14723b
(Unnamed Layer* 77) [Convolution]_output: 3d10246f
(Unnamed Layer* 78) [Scale]_output: 3d8dd919
group2_block1_conv1: 3df2ed9d
(Unnamed Layer* 80) [ElementWise]_output: 3e01c7af
group2_block1_sum: 3e12079c
global_avg_pool: 3e12079c
fc: 3ea9f0fa
softmax: 3c010a14
```

里面有很多的Unnamed Layer :blonde_woman: 我们之后再解决。

### Calibration Table转JSON

而compiler需要接受的文件是json格式的，我们利用脚本转换(注意此时的环境不在刚才的tensorrt的docker containner中，切换到nvdla/vp环境下)：

```zsh
cd /usr/local/nvdla
git clone https://github.com/nvdla/sw
# cd 到存放calibration table文件的路径
python /usr/local/nvdla/sw/umd/utils/calibdata/calib_txt_to_json.py cifar10_calibration.cache resnet18-cifar10-int8.json 
```

我们就可以拿这个json文件做compiler的calibtable参数了，但因为之前生成的代码中有许多的Unnamed 

Layer，会导致有很多层找不到自己的scale是多少的错误，这个bug在2020年6月就有人提出来了：：https://github.com/nvdla/sw/issues/201

但至今楼下都没给出解决方案，NVDLA原本的维护团队应该已经放弃治疗了（ 我比对了一下官方提供的resnet50.json的History

![](http://leiblog.wang/static/image/2021/2/CpUrAm.png)

原来官方曾经也有这样的问题。。。。。但他这个解决方法也没公布，我发现其其实就是把scale的name换成了每个网络节点的name，于是自己写了一份python脚本解决:

```zsh
import json
from collections import OrderedDict
from google.protobuf import text_format
import caffe.proto.caffe_pb2 as caffe_pb2      # 载入caffe.proto编译生成的caffe_pb2文件


caffeprototxt_path = "./deploy.prototxt"
calibletable_json_path = "./resnet18-cifar10-int8.json"
# load deploy.prototxt
net = caffe_pb2.NetParameter()
text_format.Merge(open(caffeprototxt_path).read(), net)
# load jsonfile
with open(calibletable_json_path, "r") as f:
    calible = json.load(f, object_pairs_hook=OrderedDict)
_scales = []
_mins = []
_maxs = []
_offsets = []
_new = OrderedDict()
items = calible.items()
for key, value in items:
    _scales.append(value['scale'])
    _mins.append(value['min'])
    _maxs.append(value['max'])
    _offsets.append(value['offset'])
for idx, _layer in enumerate(net.layer):
    _tempDict =  OrderedDict({
        "scale": _scales[idx],
        "min": _mins[idx],
        "max": _maxs[idx],
        "offset": _offsets[idx],
    }) 
    _new[_layer.name] =_tempDict
with open('resnet18-cifar10-int8-fixed.json', 'w') as f:
    json.dump(_new, f)
```

如这段脚本，生成的文件是 resnet18-cifar10-int8-fixed.json ，注意运行这段脚本需要caffe环境，并且编译出了pycaffe接口。

### Compile and Runtime

