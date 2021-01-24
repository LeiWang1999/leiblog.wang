---
title: TVM Notes｜一、前端导入ONNX模型

categories:
  - Technical
tags:
  - tvm
date: 2020-09-15 11:22:02
---

![Banner](http://leiblog.wang/static/image/2020/9/Figure_1.png)

对于如何学习 tvm 源代码，我参考蓝大的在知乎上对有关问题的[回复](https://www.zhihu.com/question/268423574/answer/506008668)，从前端开始阅读，这篇文章记录的是 relay 层次导入 onnx 模型的笔记。

<!-- more -->

下面是我根据 tutorial 编写的程序，有关说明和运行结果可以在[tvm docs](https://tvm.apache.org/docs/tutorials/frontend/from_onnx.html#sphx-glr-tutorials-frontend-from-onnx-py)里找到，或者复制以下代码运行。

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

关于 ONNX 格式的有关介绍，可以参考下面两篇博客：

- [开源一年多的模型交换格式 ONNX，已经一统框架江湖了？](https://flashgene.com/archives/12034.html)
- [我们来谈谈 ONNX 的日常](https://flashgene.com/archives/30780.html)

超分辨率技术（Super-Resolution, SR）是指从观测到的低分辨率图像重建出相应的高分辨率图像，使用 netron 查看模型。

查看模型能让你知道一些关键的信息、比如模型的构成、还有更重要的一点是拿到 input_name 来塑造我们的 shape_dict，就比如用到的 SR 模型，输入的 name 是‘1’，那 shape_dict 就应该是{'1': x.shape}

<div align="center">
<img src="http://leiblog.wang/static/image/2020/9/super_resolution.png" alt="Model" style="zoom:50%;" />  
</div>

接下来看`frontend.fromonnx`,可以看到 relay 前端提供了很多框架模型的接口

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

这里插个题外话，科普一下 Python 小知识。在框架源代码中经常使用 from . import A 或者 from .A import B 的操作是什么意思？

首先，`.`的意思是当前目录，`..`的意思是上级目录。

当碰到 from . import A，python 回去找当前目录下的 `__init__.py`文件，从里面去找 A，如果是..就是上级文件夹。

如果当前目录下没有`__init__.py`,则需要 from .A import B,回到当前目录下的`A.py`里去寻找 B，如果是..就是上级文件夹。

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

刚开始是对 onnx 模型的检查，这样类似的检查在 relay 前端对接 Keras 模型的代码里也出现了，虽然编写的方式不同(因为这些前端框架的接口都是不同的开发者写的，所以风格有差别)，生命 GraphProto 实例 g，在模型转化过程中，g 实例会存储包括节点、参数等信息。

关于 GraphProto，定义在 onnx 的 repo 里，主要有以下结构:

> **nodes**:用 make_node 生成的节点列表 [类型:NodeProto 列表]
>
> 比如[node1,node2,node3,…]这种的
>
> **name**:graph 的名字 [类型:字符串]
>
> **inputs**:存放 graph 的输入数据信息 [类型:ValueInfoProto 列表]
>
> 输入数据的信息以 ValueInfoProto 的形式存储，会用到 make_tensor_value_info，来将输入数据的名字、数据类型、形状(维度)给记录下来。
>
> **outputs**:存放 graph 的输出数据信息 [类型:ValueInfoProto 列表]
>
> 与 inputs 相同。
>
> **initializer**:存放超参数 [类型:TensorProto 列表]

然后例化 GraphProto、并设置 opset，至于这个参数有什么用应该要参考 onnx 的文档，总之该参数默认为 1，记录 onnx 模型里的各种参数的值，包括 name、nodes 等.

#### g.from_onnx

**第一个 loop：**

```python
        # parse network inputs to relay, aka parameters
        for init_tensor in graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            self._params[init_tensor.name] = self._parse_array(init_tensor)
            self._nodes[init_tensor.name] = new_var(
                init_tensor.name,
                shape=self._params[init_tensor.name].shape,
                dtype=self._params[init_tensor.name].dtype,
            )
```

这里将 nodes 和 params 的信息都缓存到 g 实例中，具体是哪些信息呢？比如 Conv Node 的 weight 参数，ADD Node 的 B 参数，这些由我们训练出来的参数都是在这个阶段初始化的。`_nodes`里缓存的是节点的 name、shape 和 dtype，而`_params`里缓存的是具体的参数。

**第二个 loop:**

```python
        for i in graph.input:
            # from onnx v0.2, GraphProto.input has type ValueInfoProto,
            #  and the name is 'i.name'
            i_name = self._parse_value_proto(i)
            d_type = self._parse_dtype(i, "float32")
            if i_name in self._params:
                # i is a param instead of input
                self._num_param += 1
                self._params[i_name] = self._params.pop(i_name)
                self._nodes[i_name] = new_var(
                    i_name, shape=self._params[i_name].shape, dtype=self._params[i_name].dtype
                )
            else:
                self._num_input += 1
                if i_name in self._shape:
                    tshape = self._shape[i_name]
                else:
                    raise ValueError("Must provide an input shape for `{0}`.".format(i_name))
                if isinstance(self._dtype, dict):
                    dtype = self._dtype[i_name] if i_name in self._dtype else d_type
                else:
                    dtype = d_type
                self._nodes[i_name] = new_var(i_name, shape=tshape, dtype=dtype)
            self._inputs[i_name] = self._nodes[i_name]
```

遍历所有的 input，对于我们整个模型来讲 input 只有最开始出入的图像，而对于计算图而言余下的 input 还包括运算的权重信息。

然后就是最核心的`_get_convert_map`函数的实现，实例化了所有的 OP，追溯到实例对象的`__call__`方法，调用了函数：get_relay_op，这部分先放下（，毕竟现在还没调用呢不是。

**第三个 loop:**

```python
        for node in graph.node:
            op_name = node.op_type
            if (
                op_name not in convert_map
                and op_name != "Constant"
                and op_name not in _identity_list
            ):
                unsupported_ops.add(op_name)
```

在这里，遍历整个计算图的 node，查看是否有找不到映射的。

### Reference

1. Use TVM to compile onnx model: https://tvm.apache.org/docs/tutorials/frontend/from_onnx.html#sphx-glr-tutorials-frontend-from-onnx-py
2. TVM 代码走读（一）：https://zhuanlan.zhihu.com/p/145676823
