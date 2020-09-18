---
title: TVM Notes｜一、前端导入ONNX模型
top: 10
categories:
  - Technical
tags:
  - tvm
date: 2020-09-15 11:22:02
---

![Banner](http://leiblog.wang/static/image/2020/9/Figure_1.png)

对于如何学习tvm源代码，我参考蓝大的在知乎上对有关问题的[回复](https://www.zhihu.com/question/268423574/answer/506008668)，从前端开始阅读，这篇文章记录的是relay层次导入onnx模型的笔记。

<!-- more -->

下面是我根据tutorial编写的程序，有关说明和运行结果可以在[tvm docs](https://tvm.apache.org/docs/tutorials/frontend/from_onnx.html#sphx-glr-tutorials-frontend-from-onnx-py)里找到，或者复制以下代码运行。

```python
import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from PIL import Image
from matplotlib import pyplot as plt
# Load pretrained onnx model
model_url = ''.join(['https://gist.github.com/zhreshold/',
                    'bcda4716699ac97ea44f791c24310193/raw/',
                    '93672b029103648953c4e5ad3ac3aadf346a4cdc/',
                    'super_resolution_0.2.onnx'])
model_path = download_testdata(model_url, 'super_resolution.onnx', module='onnx')
onnx_model = onnx.load(model_path)

# Load test image
image_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
image_path = download_testdata(image_url, 'cat.png', module='data')
image = Image.open(image_path).resize((224, 224))
image_ycbcr = image.convert("YCbCr")
img_y, img_cb, img_cr = image_ycbcr.split()
x = np.array(img_y)[np.newaxis, np.newaxis,:,:]

# Compile the model with relay
target = 'llvm'
input_name = '1'
shape_dict = {input_name: x.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=1):
    intrp = relay.build_module.create_executor('graph', mod, tvm.cpu(0), target)

# Execute on TVM
dtype = 'float32'
tvm_output = intrp.evaluate()(tvm.nd.array(x.astype(dtype)), **params).asnumpy()

# Display result
out_y = Image.fromarray(np.uint8((tvm_output[0, 0]).clip(0, 255)), mode='L')
out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
result = Image.merge('YCbCr', [out_y, out_cb, out_cr]).convert('RGB')
canvas = np.full((672, 672 * 2, 3), 255)
canvas[0:224, 0:224,:] = np.array(image)
canvas[:, 672:,:] = np.asarray(result)
plt.imshow(canvas.astype(np.uint8))
plt.show()
```

### super_resolution.onnx

关于ONNX格式的有关介绍，可以参考下面两篇博客：

- [开源一年多的模型交换格式ONNX，已经一统框架江湖了？](https://flashgene.com/archives/12034.html)
- [我们来谈谈ONNX的日常](https://flashgene.com/archives/30780.html)

超分辨率技术（Super-Resolution, SR）是指从观测到的低分辨率图像重建出相应的高分辨率图像，使用netron查看模型。

<div align="center">
<img src="http://leiblog.wang/static/image/2020/9/super_resolution.png" alt="Model" style="zoom:50%;" />  
</div>

接下来看`frontend.fromonnx`,可以看到relay前端提供了很多框架模型的接口

```python
from .mxnet import from_mxnet
from .mxnet_qnn_op_utils import quantize_conv_bias_mkldnn_from_var
from .keras import from_keras
from .onnx import from_onnx
from .tflite import from_tflite
from .coreml import from_coreml
from .caffe2 import from_caffe2
from .tensorflow import from_tensorflow
from .darknet import from_darknet
from .pytorch import from_pytorch
from .caffe import from_caffe
```

{% colorquote info %}

这里插个题外话，科普一下Python小知识。在框架源代码中经常使用from . import A 或者 from .A import B 的操作是什么意思？

首先，`.`的意思是当前目录，`..`的意思是上级目录。

当碰到from . import A，python回去找当前目录下的 `__init__.py`文件，从里面去找A，如果是..就是上级文件夹。

如果当前目录下没有`__init__.py`,则需要from .A import B,回到当前目录下的`A.py`里去寻找B，如果是..就是上级文件夹。

{% endcolorquote %}

```python
def from_onnx(model,
              shape=None,
              dtype="float32",
              opset=None):
    try:
        import onnx
        if hasattr(onnx.checker, 'check_model'):
            # try use onnx's own model checker before converting any model
            try:
                onnx.checker.check_model(model)
            except onnx.onnx_cpp2py_export.checker.ValidationError as e:
                import warnings
                # the checker is a bit violent about errors, so simply print warnings here
                warnings.warn(str(e))
    except ImportError:
        pass
    global g
    g = GraphProto(shape, dtype)
    graph = model.graph
    if opset is None:
        try:
            opset = model.opset_import[0].version if model.opset_import else 1
        except AttributeError:
            opset = 1
    mod, params = g.from_onnx(graph, opset)
    g = None
    return mod, params
```

刚开始是对onnx模型的检查，这样类似的检查在relay前端对接Keras模型的代码里也出现了，虽然编写的方式不同，然后将g声明为global，因为g这个变量在from_onnx.py的最开始就定义了，声明为全局变量才可以改变最外面的g，实现文件内的所有函数共享。这个变量为什么要命名为g，也许是应为是GraphProto对象的实例。

然后，`g = GraphProto(shape, dtype)`，该函数接受了两个参数，shape和dtype，在本程式中，shape是`shape_dict = {input_name: x.shape}`,也就是`{'1' : (1, 1, 224, 224)}`,这边的GraphProto是onnx的接口，实例化的时候初始化了一些参数。


