---
categories:
  - Technical
tags:
  - Tensorflow
date: 2020-04-09 22:58:34	
---

今天在学习 TensorFlow Basic Api 的时候，碰到了一个有趣的现象。

题目是计算以 2 为基数的等比数列的和, example code 是这样的：

```python
@tf.function
def converge_to_2(iters_n):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(iters_n):
        total += increment
        increment /= 2.0
    return total

result = converge_to_2(20)
print(result)
```

我起初以为这代码是错误的，为什么 total 已经被声明成了 tf.constant，但却依然可以被赋值？那 consant 的意义何在呢。可是运行了一下，却输出了正确的结果。

经过我的反复思考，我得到了自己的答案。

首先，Python 的变量类型是动态的，也许这个变量此时还是一个数组，下一个语句执行完了之后它就变成了字典。那既然是这样，Python 就不应该存在真正的 const 类型。

但 Python 是可以存在 const 的，对么，因为 Python 中的一切都是对象，而对象的魔术方法中有专门处理赋值的，我们可以在这里做手脚，实现对想声明为 const 对象的无法赋值操作。

But, Why Constant in Tensorflow do it?

其实我们只要想象一下，为什么 Tf 要定义 constant 这个变量，绝不是让你在平时的科学计算里使用的，当我们在 Train 一个 Neural Network 的时候，要保持一些量不随梯度所改变，而需要随梯度所改变的，我们使用 tf.Variable。所以 tf.constant 没必要对科学计算进行处理，只需要对 tf 内部的某些运算做处理即可。
