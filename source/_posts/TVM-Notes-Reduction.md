---
title: TVM Notes | Reduction Op
top: 10
categories:
  - Technical
tags:
  - TVM
date: 2020-11-03 14:52:30
---

### First. reduce_axis

`tvm.reduce_axis`是第一次接触到 reduce 有关的操作。

举个例子，我们实现一个矩阵乘法（参考自[tvm.d2l.ai](http://tvm.d2l.ai/chapter_common_operators/matmul.html)）：如下图

<div align="center"><image src="http://tvm.d2l.ai/_images/matmul_default.svg" /></div>

计算公式如下：

$$
C_{i,j} = \sum_{k=1}^l A_{i,k} B_{k,j}.
$$

<!-- more -->

tvm 的表达式应该要像下面这样写：

```python
def matmul(n, m, l):
    """Return the computing expression of matrix multiplication
    A : n x l matrix
    B : l x m matrix
    C : n x m matrix with C = A B
    """
    k = te.reduce_axis((0, l), name='k')
    A = te.placeholder((n, l), name='A')
    B = te.placeholder((l, m), name='B')
    C = te.compute((n, m),
                    lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
                    name='C')
    return A, B, C
```

如果我们不使用`reduce_axis`语法，则上述程序可能会编写成如下形式：

```python
def tvm_matrix_multi_noreduce():
    m, n, l = [te.var(name) for name in ('m','n','l')]
    A = te.placeholder((m, l), dtype='float32', name='a')
    B = te.placeholder((l, n), dtype='float32', name='b')
    def f(i, j):
        result=0
        for k in range(0, l):
            result += A[i,k] * B[i,j]
        return reuslt
    C = te.compute((m,n), f, name='c')
    return A, B, C
```

对与 compute 函数的第二个参数，我们很难使用一个匿名表达式表示出该计算过程，必须繁琐的自己创建一个函数。

但实际上第二段代码并不能正确执行，因为我们定义的 l 是`te.Var`对象，被定义在`~/tvm/python/tvm/tir/expr.py`里，而 range 需要接受的是两个整形。

```python
@tvm._ffi.register_object("tir.Var")
class Var(PrimExprWithOp):
    """Symbolic variable.

    Parameters
    ----------
    name : str
        The name

    dtype : Union[str, tvm.irType]
        The data type
    """

    def __init__(self, name, dtype):
        self.__init_handle_by_constructor__(_ffi_api.Var, name, dtype)
```

于是就有了对应的`reduce_axis`操作，`k = te.reduce_axis((0, l), name='k')`当碰到`A[x, k] * B[k, y]`的时候，会生成如下的代码：

```python
 for (k, 0, l) {
        a[((i) + (k))]*b[((k) + (j))]
 }
```

不难理解，对吧，但有趣的是`k = te.reduce_axis((1, l), name='k')`和`k = te.reduce_axis((0, l-1), name='k')`生成的代码是一样的：

```python
 for (k, 0, l-1) {
        a[((i) + (k))]*b[((k) + (j))]
 }
```

这里我们不探讨为什么生成的代码长这样（编译原理还没学到代码生成）。

这里给出使用 tvm 矩阵乘法的验证代码：

```python
A, B, C = tvm_matrix_multi()
s = te.create_schedule(C.op)
mod = tvm.build(s, [A,B,C])
def get_abc(shape, constructor=None):
    """Return random a, b and empty c with the same shape.
    """
    np.random.seed(0)
    a = np.random.normal(size=shape).astype(np.float32)
    b = np.random.normal(size=shape).astype(np.float32)
    c = np.empty_like(a)
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c
a, b, c = get_abc((2,2), tvm.nd.array)
mod(a, b, c)
```

以上是一维的 reduction，实际上 reduction 也可以是多维的，例如我们做多维矩阵的求和：

```python
i = te.reduce_axis((0, n), name='i')
j = te.reduce_axis((0, m), name='j')
B = te.compute((), lambda: te.sum(A[i, j], axis=(i, j)), name='b')
s = te.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
```

生成的代码如下：

```python
produce b {
  for (i, 0, n) {
    b[(i*stride)] = 0f
    for (j, 0, m) {
      b[(i*stride)] = (b[(i*stride)] + a[((i*stride) + (j*stride))])
    }
  }
}
```

### Second. Commutative Reduction

注意到，使用 reduce_axis 的时候，还需要配合一些方法，比如`te.sum`。我们可以写出自己的类似 sum 的函数，这里需要使用到`te.comm_reducer`方法。

在官方的 Tutorial 里：https://tvm.apache.org/docs/tutorials/language/reduction.html#general-reduction 给出了该方法的用法，但描述不是很人性化，根据我的研究，`te.comm_reducer`的第一个参数为比较函数，我们称之为**compare**，接受的是当前输入和上一个输出，第二个参数为初始化函数，我们称为**initial**，接受参数类型，初始化原始值。例如，当 compare 接受 reduction 轴上的第一个数据的时候，与他比较的数据还没产生，那我们输入的其实是初始化的值。

```python
comp = lambda a, b: a * b
init = lambda dtype: tvm.tir.const(1, dtype=dtype)
product = te.comm_reducer(comp, init)
```

生成的是一维向量每个元素相乘的操作：

```python
n = te.var('n')
m = te.var('m')
A = te.placeholder((n, m), name='a')
k = te.reduce_axis((0, m), name='k')
B = te.compute((n,), lambda i: product(A[i, k], axis=k), name='b')
s = te.create_schedule(B.op)
tvm.lower(s, [A, B], simple_mode=True)
```

从 k 轴上进行规约，就是把二维矩阵 A 的 i 轴上的数据相乘。

具体分析一下：

k 从 0->m 增加，当 k=0 的时候：compare 方法接受了两个参数,x 为初始化的 1，y 为 A[i,0]，然后两者相乘赋值给 B[i]；在第二个 loop，k=1, x 为 B[i], y 为 A[i,1] ......。

最后，我们就可以看懂官方手册里的 argmax 是如何实现的了，argmax 还是很有用的，目的是为了返回当前数组的最大值的下标，官方给的代码如下：

```python
# x and y are the operands of reduction, both of them is a tuple of index
# and value.
def fcombine(x, y):
    lhs = tvm.tir.Select((x[1] >= y[1]), x[0], y[0])
    rhs = tvm.tir.Select((x[1] >= y[1]), x[1], y[1])
    return lhs, rhs


# our identity element also need to be a tuple, so `fidentity` accepts
# two types as inputs.
def fidentity(t0, t1):
    return tvm.tir.const(-1, t0), tvm.te.min_value(t1)


argmax = te.comm_reducer(fcombine, fidentity, name="argmax")

# describe the reduction computation
m = te.var("m")
n = te.var("n")
idx = te.placeholder((m, n), name="idx", dtype="int32")
val = te.placeholder((m, n), name="val", dtype="int32")
k = te.reduce_axis((0, n), "k")
T0, T1 = te.compute((m,), lambda i: argmax((idx[i, k], val[i, k]), axis=k), name="T")

# the generated IR code would be:
s = te.create_schedule(T0.op)
print(tvm.lower(s, [idx, val, T0, T1], simple_mode=True))
```

argmax 接受了两个输入，一个是下标矩阵，一个是值矩阵。

当 k=0 时：输入是 idx[i, 0], val[i, 0]，fcombine 函数的输入 x 对应-1,min_value(t1)，y 对应 idx[i, 0], val[i, 0]，lhs 这一行的含义是，当 min_value(t1)比 val[i, 0]大，则 lhs 为-1，否则为 idx[i, 0],反之亦然。
