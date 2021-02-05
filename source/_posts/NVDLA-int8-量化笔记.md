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

如果我们想要重构一份自己的网络量化方法，核心的部分是要自己定义一个Class、继承`trt.IInt8EntropyCalibrator2`，具体内容可以阅读MNIST量化的`calibrator.py`文件，我们需要提供自己训练的时候使用的数据集，deploy.prototxt，caffemodel这三个文件，然后自己编写解析的方法，例如本文对针对ImageNet的resnet18网络做量化，就写成这样：

```python
# Returns a numpy buffer of shape (num_images, 1, 28, 28)
def load_data(filepath):
    test_imgs = []
    global fileList
    fileList = os.listdir(filepath)
    mean = np.ones([3, 224, 224], dtype=np.float)
    mean[0,:,:] = 104
    mean[1,:,:] = 117
    mean[2,:,:] = 123
    for img_path in fileList:
        img_path = filepath +'/'+ img_path
        img = cv.imread(img_path)
        img = crop_img(img, [224, 224])
        img = img.transpose((2, 0, 1))
        img = img - mean
        test_imgs.append(img)
    # Need to scale all values to the range of [0, 1]
    return np.ascontiguousarray(test_imgs).astype(np.float32)

# Returns a numpy buffer of shape (num_images)
def load_labels(filepath):
    global fileList
    test_labels = []
    labels_mapping = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for each in lines:
            imageName, labels = each.strip('\n').split(' ')
            labels_mapping[imageName] = int(labels)
    for each in fileList:
        test_labels.append(labels_mapping[each])
    return np.ascontiguousarray(test_labels)
```

而自己定义的`EntropyCalibrator`如下：

```python
class CustomEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, batch_size=64):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = load_data(training_data)
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
Calibrating batch 0, containing 64 images
Calibrating batch 10, containing 64 images
Validating batch 10
Validating batch 20
Validating batch 30
Total Accuracy: 58.54145854145854%
```

而在运行目录下生成的：cache文件就是calibration table了：

```zsh
TRT-7202-EntropyCalibration2
data: 3f9839e4
(Unnamed Layer* 0) [Convolution]_output: 404efcc8
(Unnamed Layer* 1) [Scale]_output: 3dad1056
(Unnamed Layer* 2) [Scale]_output: 3c612a39
conv1: 3c35ea42
pool1: 3c35ea42
(Unnamed Layer* 5) [Convolution]_output: 3c44ec17
(Unnamed Layer* 6) [Scale]_output: 3d79ddee
res2a_branch1: 3c71cc22
(Unnamed Layer* 8) [Convolution]_output: 3cbea256
(Unnamed Layer* 9) [Scale]_output: 3db5bbfc
(Unnamed Layer* 10) [Scale]_output: 3c893a25
res2a_branch2a: 3bfa7ac1
(Unnamed Layer* 12) [Convolution]_output: 3bb2c9bf
(Unnamed Layer* 13) [Scale]_output: 3d7d559f
res2a_branch2b: 3c431934
(Unnamed Layer* 15) [ElementWise]_output: 3caeb939
res2a: 3c53f5bb
(Unnamed Layer* 17) [Convolution]_output: 3c10df08
(Unnamed Layer* 18) [Scale]_output: 3d87d76c
(Unnamed Layer* 19) [Scale]_output: 3c5511f5
res2b_branch2a: 3c2a88bb
(Unnamed Layer* 21) [Convolution]_output: 3bb6625d
(Unnamed Layer* 22) [Scale]_output: 3d88ba43
res2b_branch2b: 3c6ad191
(Unnamed Layer* 24) [ElementWise]_output: 3c996872
res2b: 3c90d5e3
(Unnamed Layer* 26) [Convolution]_output: 3b7d9a5e
(Unnamed Layer* 27) [Scale]_output: 3d74307d
res3a_branch1: 3bfdfab7
(Unnamed Layer* 29) [Convolution]_output: 3c2fe025
(Unnamed Layer* 30) [Scale]_output: 3d7def72
(Unnamed Layer* 31) [Scale]_output: 3c31f779
res3a_branch2a: 3c2c8afe
(Unnamed Layer* 33) [Convolution]_output: 3bbf4a3a
(Unnamed Layer* 34) [Scale]_output: 3d83d04d
res3a_branch2b: 3c572369
(Unnamed Layer* 36) [ElementWise]_output: 3c69c835
res3a: 3c8064f2
(Unnamed Layer* 38) [Convolution]_output: 3c14239e
(Unnamed Layer* 39) [Scale]_output: 3d8554e6
(Unnamed Layer* 40) [Scale]_output: 3c31028f
res3b_branch2a: 3c169d63
(Unnamed Layer* 42) [Convolution]_output: 3b8624f8
(Unnamed Layer* 43) [Scale]_output: 3d9474e5
res3b_branch2b: 3c3b2d4d
(Unnamed Layer* 45) [ElementWise]_output: 3c62c909
res3b: 3c3576b1
(Unnamed Layer* 47) [Convolution]_output: 3b206632
(Unnamed Layer* 48) [Scale]_output: 3d814cf0
res4a_branch1: 3b9986ae
(Unnamed Layer* 50) [Convolution]_output: 3c17f128
(Unnamed Layer* 51) [Scale]_output: 3d827168
(Unnamed Layer* 52) [Scale]_output: 3c2b6742
res4a_branch2a: 3c1b4ed9
(Unnamed Layer* 54) [Convolution]_output: 3bab8f82
(Unnamed Layer* 55) [Scale]_output: 3d885e72
res4a_branch2b: 3c3ce0a1
(Unnamed Layer* 57) [ElementWise]_output: 3c4deb0b
res4a: 3c63cff2
(Unnamed Layer* 59) [Convolution]_output: 3bdc2dea
(Unnamed Layer* 60) [Scale]_output: 3d6cc90a
(Unnamed Layer* 61) [Scale]_output: 3c2cbe15
res4b_branch2a: 3c05ab17
(Unnamed Layer* 63) [Convolution]_output: 3b636e4c
(Unnamed Layer* 64) [Scale]_output: 3d857fd8
res4b_branch2b: 3c5bd401
(Unnamed Layer* 66) [ElementWise]_output: 3c6a9868
res4b: 3c5cbc54
(Unnamed Layer* 68) [Convolution]_output: 3ad16737
(Unnamed Layer* 69) [Scale]_output: 3d662897
res5a_branch1: 3bab30a7
(Unnamed Layer* 71) [Convolution]_output: 3bbe5321
(Unnamed Layer* 72) [Scale]_output: 3d82074f
(Unnamed Layer* 73) [Scale]_output: 3c2da30b
res5a_branch2a: 3c0fe297
(Unnamed Layer* 75) [Convolution]_output: 3b4dae0e
(Unnamed Layer* 76) [Scale]_output: 3d7a5e58
res5a_branch2b: 3c59f7ca
(Unnamed Layer* 78) [ElementWise]_output: 3c7a4a3e
res5a: 3c69fa84
(Unnamed Layer* 80) [Convolution]_output: 3bea45a4
(Unnamed Layer* 81) [Scale]_output: 3d891fa0
(Unnamed Layer* 82) [Scale]_output: 3c4d88a0
res5b_branch2a: 3bc901ec
(Unnamed Layer* 84) [Convolution]_output: 3af65712
(Unnamed Layer* 85) [Scale]_output: 3ddd4245
res5b_branch2b: 3e07db30
(Unnamed Layer* 87) [ElementWise]_output: 3e04055e
res5b: 3e1b53d4
pool5: 3e1b53d4
fc1000: 3e1f8999
prob: 3acc1705
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

原来官方曾经也有这样的问题。但他这个解决方法也没公布，我发现其其实就是把scale的name换成了每个网络节点的name，于是自己写了一份python脚本解决（本来想在原先的项目上提个pr，但方便解析prototxt需要安装额外的库所以就放下了:

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

如这段脚本，生成的文件是imagenet_resnet18_calibration.cache ，注意运行这段脚本需要caffe环境，并且编译出了pycaffe接口。

### Compile and Runtime

针对fp16的compile：

```zsh
./nvdla_compiler --prototxt resnet18-imagenet-caffe/deploy.prototxt --caffemodel resnet18-imagenet-caffe/resnet-18.caffemodel --profile imagenet-fp16
```

针对fp16的runtime：

```zsh
./nvdla_runtime --loadable imagenet-fp16.nvdla --image resnet18-imagenet-caffe/resized/0_tench.jpg   --mean 104,117,123  --rawdump
```

```zsh
# cat output.dimg 
0.97998 0.00159931 1.13845e-05 3.57628e-05 8.67844e-05 0.000154614 2.14577e-05 2.20537e-06 1.3113e-06 1.78814e-07 5.90086e-06 1.03116e-05 3.03984e-06 1.37091e-06 1.49012e-06 9.23872e-06 3.57628e-06 1.72853e-06 1.07288e-06 9.53674e-07 1.07288e-06 1.03712e-05 2.38419e-06 1.37091e-06 2.26498e-06 1.2517e-06 9.59635e-06 9.47714e-06 2.14577e-06 6.77109e-05 7.85589e-05 2.77758e-05 2.7895e-05 3.89218e-05 1.26958e-05 1.68681e-05 3.0458e-05 4.05312e-06 7.62939e-06 9.11951e-06 1.01924e-05 1.2517e-06 1.78814e-06 4.47035e-06 9.53674e-07 2.38419e-07 4.17233e-06 3.016e-05 7.7486e-07 1.60933e-06 3.39746e-06 2.56896e-05 1.3113e-06 1.01328e-06 4.76837e-07 1.96695e-06 5.36442e-07 2.98023e-07 2.26498e-06 1.66893e-06 1.84774e-06 1.3113e-06 1.90735e-06 2.44379e-06 8.16584e-06 3.8743e-06 1.78814e-06 4.76837e-07 9.53674e-07 1.96695e-06 6.55651e-07 1.43051e-06 1.13249e-06 2.86102e-06 8.34465e-07 1.60933e-06 1.13249e-06 2.98023e-07 1.43051e-06 4.17233e-07 2.20537e-06 5.36442e-07 1.84774e-06 3.8743e-06 1.2517e-06 7.15256e-07 4.29153e-06 5.30481e-06 1.68681e-05 9.89437e-06 1.60336e-05 4.05312e-06 1.90735e-05 8.64267e-06 1.10269e-05 9.53674e-06 1.2517e-05 8.9407e-06 2.47359e-05 6.79493e-06 1.78814e-06 1.3113e-06 2.86102e-06 0.000196218 8.34465e-07 2.20537e-06 2.5034e-06 6.4373e-06 9.59635e-06 3.15905e-06 3.21865e-06 1.54972e-06 1.35899e-05 5.26309e-05 3.09944e-05 5.1856e-06 5.60284e-06 8.34465e-06 1.92523e-05 4.41074e-06 3.69549e-06 1.03712e-05 1.64509e-05 3.57628e-06 2.43783e-05 2.14577e-06 7.62939e-06 5.96046e-07 2.02656e-06 4.23193e-06 3.8743e-06 1.13249e-06 2.6226e-06 1.70469e-05 3.8147e-06 1.60933e-06 3.21865e-06 1.07288e-06 8.9407e-07 4.76837e-07 1.37091e-06 1.78814e-06 1.43051e-06 1.07288e-06 8.22544e-06 1.01924e-05 1.78814e-06 3.39746e-06 3.03984e-06 3.9041e-05 1.36495e-05 6.31809e-06 8.34465e-07 5.96046e-07 2.80142e-06 5.36442e-07 1.13249e-06 2.32458e-06 4.11272e-06 1.49012e-06 1.01328e-06 2.86102e-06 2.98023e-06 9.53674e-06 1.01328e-06 2.08616e-06 2.38419e-06 2.14577e-06 4.29153e-06 7.15256e-07 4.76837e-07 1.96695e-06 1.07288e-06 1.66893e-06 1.13249e-06 3.63588e-06 1.37091e-06 2.98023e-07 9.53674e-07 2.14577e-06 3.27826e-06 4.76837e-07 2.32458e-06 8.34465e-07 2.02656e-06 2.26498e-06 3.33786e-06 5.1856e-06 1.19209e-06 1.54972e-06 6.55651e-07 2.74181e-06 1.37091e-06 5.24521e-06 2.02656e-06 1.84774e-06 3.57628e-07 2.38419e-07 8.9407e-07 8.9407e-07 9.53674e-07 1.72853e-06 1.2517e-06 5.96046e-07 1.01328e-06 7.7486e-07 9.53674e-07 5.84126e-06 3.27826e-06 2.6226e-06 1.01328e-06 4.11272e-06 1.07288e-06 1.13249e-06 1.13249e-06 1.37091e-06 2.02656e-06 2.38419e-07 1.84774e-06 5.96046e-07 2.86102e-06 6.55651e-07 7.7486e-07 4.23193e-06 1.54972e-06 8.16584e-06 1.96695e-06 2.26498e-06 7.7486e-07 2.98023e-07 1.72853e-06 1.84774e-06 5.96046e-07 4.76837e-07 2.86102e-06 1.12653e-05 1.07288e-06 7.689e-06 1.72853e-06 1.13249e-06 2.38419e-06 2.08616e-06 2.20537e-06 3.27826e-06 1.13249e-06 3.45707e-06 1.13249e-06 1.84774e-06 7.7486e-07 1.3113e-06 1.2517e-06 6.55651e-07 8.9407e-07 5.66244e-06 2.26498e-06 4.76837e-06 1.01328e-06 7.15256e-07 4.17233e-07 4.58956e-06 2.98023e-06 5.36442e-07 1.43647e-05 1.18613e-05 2.44379e-06 1.84774e-06 6.55651e-07 2.38419e-07 2.74181e-06 4.17233e-07 3.57628e-07 2.92063e-06 1.2517e-06 2.92063e-06 2.98023e-06 4.76837e-07 4.76837e-07 1.03116e-05 1.60933e-06 5.36442e-07 7.21216e-06 4.64916e-06 2.08616e-06 8.9407e-07 1.54972e-06 1.60933e-06 2.98023e-06 8.34465e-07 1.2517e-06 2.38419e-07 7.7486e-07 1.72853e-06 4.47035e-06 6.55651e-07 8.9407e-07 7.15256e-07 1.01328e-06 2.98023e-07 1.07288e-05 5.96046e-07 9.53674e-07 1.43051e-06 1.37091e-06 2.44379e-06 2.44379e-06 6.55651e-07 2.86102e-06 6.55651e-07 6.55651e-07 5.72205e-06 8.34465e-07 2.92063e-06 5.126e-06 1.2517e-06 2.86102e-06 2.44379e-06 9.53674e-07 2.563e-06 1.07288e-06 1.01328e-06 8.34465e-07 4.76837e-07 5.57899e-05 7.7486e-07 1.66893e-06 1.51992e-05 9.53674e-07 3.27826e-06 2.80142e-06 7.27177e-06 5.66244e-06 4.94719e-06 1.84774e-06 2.26498e-06 3.75509e-06 1.40071e-05 8.34465e-07 6.49691e-06 6.4373e-05 9.53674e-07 1.19209e-06 1.50204e-05 6.31809e-06 2.44379e-06 8.76188e-06 1.49012e-06 2.86102e-06 7.15256e-07 2.38419e-07 5.96046e-07 8.34465e-07 3.51667e-06 4.94719e-06 1.2517e-06 3.57628e-07 5.36442e-07 9.65595e-06 3.75509e-06 2.21133e-05 1.26362e-05 2.14577e-06 6.3777e-06 2.20537e-06 1.49608e-05 6.4373e-06 2.68221e-06 1.19209e-07 1.01328e-06 1.2517e-06 7.7486e-07 4.41074e-06 2.32458e-06 8.34465e-07 1.43051e-06 1.19209e-06 8.9407e-07 9.65595e-06 8.76188e-06 3.8147e-06 1.19805e-05 7.7486e-06 5.30481e-06 2.45571e-05 3.57628e-07 5.1856e-06 2.80142e-06 8.9407e-07 2.20537e-06 5.36442e-07 0.00881195 0.000128746 0.00135708 0.000327349 2.03252e-05 7.79033e-05 0.000499249 3.8147e-06 7.236e-05 4.17233e-07 1.3113e-06 1.01328e-06 5.36442e-07 5.36442e-07 7.7486e-07 5.36442e-07 5.96046e-07 2.98023e-07 7.15256e-07 1.3113e-06 1.72853e-06 2.92063e-06 4.17233e-07 5.36442e-07 2.80142e-06 5.96046e-07 1.3113e-06 1.2517e-06 2.20537e-06 5.36442e-07 2.55108e-05 7.7486e-07 4.17233e-07 2.80142e-06 8.34465e-07 5.36442e-07 4.76837e-07 1.3113e-06 2.14577e-06 1.43051e-06 1.72853e-06 7.7486e-07 9.53674e-07 7.15256e-07 2.38419e-06 1.96695e-06 2.02656e-06 7.15256e-07 1.37091e-06 1.43051e-06 1.37091e-06 7.15256e-07 1.43051e-06 8.34465e-07 1.66893e-06 1.01328e-06 8.34465e-07 9.53674e-07 2.02656e-06 1.07288e-06 2.74181e-06 8.9407e-07 2.20537e-06 1.96695e-06 1.3113e-06 4.76837e-07 3.57628e-07 5.60284e-06 1.37091e-06 3.57628e-07 1.07288e-06 4.17233e-07 2.20537e-06 4.11272e-06 2.38419e-06 5.1856e-06 3.51667e-06 6.55651e-07 3.45707e-06 6.55651e-07 2.98023e-06 1.90735e-06 1.13249e-06 2.08616e-06 8.70228e-06 4.76837e-06 1.60933e-06 3.45707e-06 1.37091e-06 7.15256e-07 8.34465e-07 5.96046e-07 4.35114e-06 6.61612e-06 9.53674e-07 1.01328e-06 1.96695e-06 6.55651e-07 2.5034e-06 1.49012e-06 9.53674e-07 1.72853e-06 5.72205e-06 6.55651e-07 5.96046e-07 2.563e-06 4.17233e-07 6.55651e-07 4.76837e-07 5.36442e-07 4.35114e-06 2.38419e-07 1.96695e-06 6.3777e-06 1.54972e-06 1.19209e-06 1.37091e-06 2.02656e-06 1.60933e-06 1.54972e-06 4.17233e-07 1.07288e-06 1.13249e-06 8.9407e-07 8.34465e-07 5.00679e-06 4.76837e-06 6.55651e-07 8.9407e-07 6.73532e-06 5.96046e-07 2.38419e-07 1.60933e-06 3.8147e-06 1.01328e-06 2.563e-06 1.54972e-06 4.76837e-07 1.01328e-06 1.96695e-06 2.74181e-06 1.07288e-06 1.43051e-06 3.57628e-07 8.34465e-07 8.9407e-07 3.39746e-05 1.19209e-06 5.96046e-07 5.36442e-07 5.54323e-06 1.72853e-06 1.78814e-06 1.37091e-06 2.98023e-06 1.43051e-06 2.32458e-06 8.9407e-07 5.96046e-07 3.57628e-07 7.15256e-07 1.2517e-06 2.74181e-06 1.84774e-06 5.36442e-07 1.37091e-06 4.17233e-07 6.55651e-07 1.37091e-06 5.36442e-07 1.66893e-06 1.19209e-06 3.57628e-07 1.3113e-06 8.34465e-07 2.38419e-07 5.36442e-07 3.75509e-06 4.23193e-06 7.15256e-07 7.7486e-07 3.33786e-06 8.34465e-07 1.84774e-06 4.76837e-07 2.26498e-06 1.3113e-06 1.49012e-06 6.67572e-06 4.17233e-07 1.78814e-07 2.32458e-06 6.55651e-07 9.53674e-07 8.34465e-07 1.39475e-05 5.96046e-07 3.57628e-07 1.54972e-06 1.78814e-06 2.26498e-06 4.11272e-06 1.60933e-06 2.92063e-06 3.57628e-07 2.98023e-07 6.55651e-07 2.80142e-06 5.66244e-06 4.17233e-07 2.26498e-06 5.30481e-06 7.15256e-07 2.6226e-06 2.38419e-07 1.19209e-06 1.54972e-06 2.26498e-06 3.45707e-06 2.86102e-06 5.36442e-07 5.54323e-06 1.60933e-06 8.9407e-07 3.03984e-06 1.72853e-06 2.23517e-05 1.13249e-06 2.68221e-06 1.2517e-06 7.7486e-07 6.55651e-07 2.20537e-06 8.9407e-07 1.2517e-06 5.36442e-07 9.53674e-07 6.55651e-07 4.17233e-07 5.96046e-07 8.34465e-07 2.563e-06 1.54972e-06 9.95398e-06 3.45707e-06 7.15256e-07 3.09944e-06 1.3113e-06 5.36442e-07 4.35114e-06 1.60933e-06 5.96046e-07 7.7486e-06 8.34465e-07 5.24521e-06 7.15256e-07 7.7486e-07 1.01328e-06 1.54972e-06 7.7486e-07 8.34465e-07 1.54972e-06 2.14577e-06 1.90735e-06 3.75509e-06 1.49012e-06 1.19209e-06 9.53674e-07 8.9407e-07 6.31809e-06 1.13249e-06 5.96046e-07 4.76837e-07 2.98023e-07 3.57628e-07 5.36442e-07 1.78814e-06 1.07288e-06 2.20537e-06 4.76837e-07 5.96046e-07 4.64916e-06 2.08616e-06 2.26498e-06 1.54972e-06 1.055e-05 7.15256e-07 8.16584e-06 9.53674e-07 3.39746e-06 5.96046e-07 4.82202e-05 4.76837e-07 2.98023e-07 3.57628e-07 1.39475e-05 1.37091e-06 3.75509e-06 2.98023e-07 4.17233e-07 8.34465e-07 1.49012e-06 5.66244e-06 3.03984e-06 5.24521e-06 1.2517e-06 2.26498e-06 1.78814e-05 2.68221e-06 4.76837e-07 1.13249e-06 3.15905e-06 3.21865e-06 1.49012e-06 7.15256e-07 7.15256e-07 2.98023e-07 4.17233e-07 5.96046e-07 6.55651e-07 8.9407e-07 6.4373e-06 6.55651e-07 5.06639e-06 1.37091e-06 5.42402e-06 5.06639e-06 1.01328e-06 4.76837e-07 1.19209e-06 8.07047e-05 2.74181e-06 1.13249e-06 3.63588e-06 3.57628e-06 1.90735e-06 1.78814e-06 1.5378e-05 1.37091e-06 6.19888e-06 1.60933e-06 7.7486e-07 1.96695e-06 1.2517e-06 6.55651e-06 2.98023e-07 1.64509e-05 8.9407e-07 6.55651e-07 2.14577e-06 5.60284e-06 2.08616e-06 9.53674e-07 2.26498e-06 5.96046e-07 1.84774e-06 8.9407e-07 7.7486e-07 4.76837e-07 1.2517e-06 8.04663e-06 1.13249e-06 7.7486e-07 1.96695e-06 1.3113e-06 7.7486e-07 5.96046e-07 4.76837e-07 6.55651e-07 0.00323105 2.68221e-06 5.96046e-07 1.78814e-06 7.7486e-07 4.35114e-06 1.96695e-06 1.3113e-06 8.04663e-06 2.02656e-06 1.54972e-06 5.96046e-07 1.07288e-06 1.2517e-06 2.02656e-06 1.72853e-06 1.90735e-06 3.75509e-06 4.76837e-07 6.07967e-06 3.57628e-07 2.98023e-07 4.41074e-06 2.98023e-07 8.34465e-07 1.37091e-06 2.32458e-06 8.40425e-06 8.9407e-07 2.44379e-06 1.01328e-06 1.19209e-06 6.55651e-07 4.76837e-07 2.68221e-06 2.68221e-06 1.60933e-06 3.45707e-06 5.06639e-06 3.03984e-06 1.19209e-06 5.96046e-07 5.96046e-07 1.18017e-05 1.49012e-06 4.17233e-07 4.47035e-06 2.20537e-06 1.66893e-06 2.08616e-06 5.00679e-06 1.19209e-06 5.96046e-07 3.57628e-06 1.2517e-06 2.08616e-06 2.02656e-06 1.2517e-06 7.689e-06 1.13249e-06 3.99351e-06 1.49012e-06 3.57628e-07 8.34465e-07 2.5034e-06 2.38419e-06 4.88758e-06 8.34465e-07 1.37091e-06 1.3113e-06 1.19209e-06 5.36442e-07 1.13249e-06 4.17233e-07 1.01328e-06 1.19209e-06 3.63588e-06 6.55651e-07 3.99351e-06 2.44379e-06 2.5034e-06 8.34465e-07 1.19209e-06 6.55651e-06 2.5034e-06 4.47035e-06 5.78165e-06 1.13249e-06 7.7486e-07 9.53674e-07 4.29153e-06 2.86102e-06 2.92063e-06 6.55651e-07 1.03116e-05 2.38419e-06 1.54972e-06 3.8147e-06 1.84774e-06 8.9407e-07 1.01328e-06 3.75509e-06 1.13249e-06 2.74181e-06 1.37091e-06 8.9407e-07 4.76837e-07 3.27826e-06 7.7486e-07 8.34465e-07 1.01328e-06 1.54972e-06 2.02656e-06 5.54323e-06 6.55651e-07 2.98023e-07 4.76837e-07 6.55651e-07 1.37091e-06 4.17233e-07 8.34465e-07 8.9407e-07 1.43051e-06 4.17233e-07 9.53674e-07 3.33786e-06 8.34465e-07 5.36442e-06 7.15256e-07 1.13249e-06 5.96046e-07 2.44379e-06 5.96046e-07 1.43051e-06 5.96046e-07 1.37091e-06 1.43051e-06 2.98023e-06 5.36442e-07 1.07288e-06 2.32458e-06 2.98023e-06 2.98023e-07 1.3113e-06 7.98702e-06 3.21865e-06 2.5034e-06 7.7486e-07 4.76837e-06 1.43051e-06 4.76837e-07 1.66893e-06 3.03984e-06 1.13845e-05 2.68221e-06 1.78814e-06 3.09944e-06 4.17233e-07 1.3113e-06 4.35114e-06 8.9407e-07 5.36442e-07 9.53674e-07 2.68221e-06 2.98023e-07 6.25849e-06 1.72853e-06 3.33786e-06 4.11272e-06 5.96046e-07 4.76837e-07 3.21865e-06 6.49691e-06 1.3113e-06 1.54972e-06 1.49012e-06 2.20537e-06 5.36442e-07 1.33514e-05 4.41074e-06 8.9407e-07 9.71556e-06 3.75509e-06 1.96695e-06 3.51667e-06 4.11272e-06 2.74181e-05 1.37091e-06 7.92742e-06 5.60284e-06 6.3777e-06 1.60933e-06 1.66893e-06 4.23193e-06 1.13249e-06 1.96695e-06 2.20537e-06 2.58684e-05 1.90735e-06 1.3113e-06 1.72853e-06 8.9407e-07 1.43051e-06 7.7486e-07 1.3113e-06 1.01328e-06 2.08616e-06 1.96695e-06 1.13249e-06 7.15256e-07 1.3113e-06 1.13249e-06 1.2517e-06 2.32458e-06 1.96695e-06 3.69549e-05 2.98023e-07 1.69277e-05 1.66893e-06 1.90735e-06 2.563e-06 7.56979e-06 2.38419e-06 1.84774e-06 3.57628e-07 1.50204e-05 1.78814e-06 1.96695e-06 1.54376e-05 7.45058e-06 2.5034e-06 8.34465e-07 1.96695e-06 2.38419e-06 1.54972e-06 9.53674e-07 3.33786e-06 9.23872e-06 8.10623e-06 4.35114e-05 1.38283e-05 5.96046e-07
```

看的太累，可以直接看top5:

```zsh
 cat output.dimg | sed "s#\ #\n#g" | cat -n | sort -gr -k2,2 | head -5 
     1  0.97998
   390  0.00881195
   759  0.00323105
     2  0.00159931
   392  0.00135708
```

int8的编译

```zsh
# compile per-kernel
./nvdla_compiler --prototxt resnet18-imagenet-caffe/deploy.prototxt --caffemodel resnet18-imagenet-caffe/resnet-18.caffemodel --profile imagenet-int8.kernel --cprecision int8 --calibtable resnet18-imagenet-caffe/resnet18.imagenet.fixed.int8.json 
# compile per-filter 主要是对比一下了两者有没有不同
./nvdla_compiler --prototxt resnet18-imagenet-caffe/deploy.prototxt --caffemodel resnet18-imagenet-caffe/resnet-18.caffemodel --profile imagenet-int8.filter --cprecision int8 --calibtable resnet18-imagenet-caffe/resnet18.imagenet.fixed.int8.json --quantizationMode per-filter

./nvdla_runtime --loadable imagenet-int8.filter.nvdla      --image resnet18-im
agenet-caffe/resized/0_tench.jpg   --rawdump
```

int8的runtime

```zsh
127 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

