---
title: Pycaffe Transformer
categories:
  - Technical
tags:
  - Caffe
date: 2021-01-22 13:12:53
---

在使用Caffe训练网络的时候，往往会对图像有一些预处理操作、例如做resize、对训练数据减去均值等，在实际推理的时候我们还需把输入图像resize到网络输入的大小，必要的时候还需要做图像通道的transpose，这些操作可以caffe的transformer来实现。

<!-- more -->

通过查看Transformer对象的源代码，就可以知道有哪些方法。

```python
class Transformer:
"""
Transform input for feeding into a Net.
Note: this is mostly for illustrative purposes and it is likely better
to define your own input preprocessing routine for your needs.
Parameters
----------
net : a Net for which the input should be prepared
"""
def __init__(self, inputs):
    self.inputs = inputs #  inputs存的是一个name和shape的字典，比如 {'data':(1,3,256,340)}
    self.transpose = {}  # Transformer中的这些参数全部是字典dict
    self.channel_swap = {}
    self.raw_scale = {}
    self.mean = {}
    self.input_scale = {}
def __check_input(self, in_):
    if in_ not in self.inputs:
        raise Exception('{} is not one of the net inputs: {}'.format(
            in_, self.inputs))
def preprocess(self, in_, data):
    """
    Format input for Caffe:
    - convert to single
    - resize to input dimensions (preserving number of channels)
    - transpose dimensions to K x H x W
    - reorder channels (for instance color to BGR)
    - scale raw input (e.g. from [0, 1] to [0, 255] for ImageNet models)
    - subtract mean
    - scale feature
    Parameters
    ----------
    in_ : name of input blob to preprocess for
    data : (H' x W' x K) ndarray  ！！！！输入data必须是单个image，后面有限制
    Returns
    -------
    caffe_in : (K x H x W) ndarray for input to a Net
    """
    self.__check_input(in_)
    caffe_in = data.astype(np.float32, copy=False)
    transpose = self.transpose.get(in_)
    channel_swap = self.channel_swap.get(in_)
    raw_scale = self.raw_scale.get(in_)
    mean = self.mean.get(in_)
    input_scale = self.input_scale.get(in_)
    in_dims = self.inputs[in_][2:] # in_dims即inputs设定的shape的(H,W)
    # 检查输入data的(H,W)是否与in_dims一样,注意这里[:2]限制了输入data只能是单个iamge
    if caffe_in.shape[:2] != in_dims: 
        caffe_in = resize_image(caffe_in, in_dims)
    if transpose is not None:
        caffe_in = caffe_in.transpose(transpose)
    if channel_swap is not None:  # 如果没有transpose第一维不是通道，这里会报错吧。。。
        caffe_in = caffe_in[channel_swap, :, :]
    if raw_scale is not None:
        caffe_in *= raw_scale
    if mean is not None:
        caffe_in -= mean
    if input_scale is not None:
        caffe_in *= input_scale
    return caffe_in
def deprocess(self, in_, data):
    """
    Invert Caffe formatting; see preprocess().
    """
    self.__check_input(in_)
    decaf_in = data.copy().squeeze()
    transpose = self.transpose.get(in_)
    channel_swap = self.channel_swap.get(in_)
    raw_scale = self.raw_scale.get(in_)
    mean = self.mean.get(in_)
    input_scale = self.input_scale.get(in_)
    if input_scale is not None:
        decaf_in /= input_scale
    if mean is not None:
        decaf_in += mean
    if raw_scale is not None:
        decaf_in /= raw_scale
    if channel_swap is not None:
        decaf_in = decaf_in[np.argsort(channel_swap), :, :]
    if transpose is not None:
        decaf_in = decaf_in.transpose(np.argsort(transpose))
    return decaf_in
def set_transpose(self, in_, order):
    """
    Set the input channel order for e.g. RGB to BGR conversion
    as needed for the reference ImageNet model.
    Parameters
    ----------
    in_ : which input to assign this channel order
    order : the order to transpose the dimensions
    """
    self.__check_input(in_)
    if len(order) != len(self.inputs[in_]) - 1:  #注意，这里只比较设定shape的后3维
        raise Exception('Transpose order needs to have the same number of '
                        'dimensions as the input.')
    self.transpose[in_] = order
def set_channel_swap(self, in_, order):
    """
    Set the input channel order for e.g. RGB to BGR conversion
    as needed for the reference ImageNet model.
    N.B. this assumes the channels are the first dimension AFTER transpose.
    Parameters
    ----------
    in_ : which input to assign this channel order
    order : the order to take the channels.
        (2,1,0) maps RGB to BGR for example.
    """
    self.__check_input(in_)
    if len(order) != self.inputs[in_][1]:
        raise Exception('Channel swap needs to have the same number of '
                        'dimensions as the input channels.')
    self.channel_swap[in_] = order
def set_raw_scale(self, in_, scale):
    """
    Set the scale of raw features s.t. the input blob = input * scale.
    While Python represents images in [0, 1], certain Caffe models
    like CaffeNet and AlexNet represent images in [0, 255] so the raw_scale
    of these models must be 255.
    Parameters
    ----------
    in_ : which input to assign this scale factor
    scale : scale coefficient
    """
    self.__check_input(in_)
    self.raw_scale[in_] = scale
def set_mean(self, in_, mean):
    """
    Set the mean to subtract for centering the data.
    Parameters
    ----------
    in_ : which input to assign this mean.
    mean : mean ndarray (input dimensional or broadcastable)
    """
    self.__check_input(in_)
    ms = mean.shape
    if mean.ndim == 1:
        # broadcast channels
        if ms[0] != self.inputs[in_][1]:
            raise ValueError('Mean channels incompatible with input.')
        mean = mean[:, np.newaxis, np.newaxis]
    else:
        # elementwise mean
        if len(ms) == 2:
            ms = (1,) + ms
        if len(ms) != 3:
            raise ValueError('Mean shape invalid')
        if ms != self.inputs[in_][1:]:
            raise ValueError('Mean shape incompatible with input shape.')
    self.mean[in_] = mean
def set_input_scale(self, in_, scale):
    """
    Set the scale of preprocessed inputs s.t. the blob = blob * scale.
    N.B. input_scale is done AFTER mean subtraction and other preprocessing
    while raw_scale is done BEFORE.
    Parameters
    ----------
    in_ : which input to assign this scale factor
    scale : scale coefficient
    """
    self.__check_input(in_)
    self.input_scale[in_] = scale
```

