---
title: tilelang jit
categories:
  - Technical
tags:
  - MLSys
date: 2025-02-21 15:15:31
---

继上文 [运行时CUDA源代码转Torch Function的若干方法](https://zhuanlan.zhihu.com/p/22067744401) 评论区收集到了一些比较新颖的方案，例如pybind的作者新提出的bind方法nanobind, triton里生成了一个cpp使用PyObj来获得一个python对象的成员，从而获得torch的指针和shape等信息，以及mlc-python项目中使用的cython解决办法，花了一段时间在tilelang的jit部分添加了各种execution backend的实现，目前jit的execution backend包括`dlpack`,`cpp_extension`,`ctypes`以及`cython`这四个，我实际实现下来，cython的runtime overhead最小，实现最方便(熟练了cython之后)，这里总结一下给各位出主意的同学们交差，顺便介绍一下tilelang的jit.

<!-- more -->

在TileLang中，JIT负责把一个TileLang的Program生成生成一个可以输入Torch Tensor的Torch Function, 有两种主要的使用方法，一种是装饰器:

```python
@tilelang.jit(
  out_idx = -1, # output tensor index
  execution_backend = "dlpack", # execution backend
  target: Union[str, Target] = "cuda", # "cuda", "hip", "cpu
  ...
)
@T.prim_func
def main(
        A: T.Buffer(A_shape, in_dtype),
        B: T.Buffer(B_shape, in_dtype),
        C: T.Buffer((M, N), out_dtype),
):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared(A_shared_shape, in_dtype)
        B_shared = T.alloc_shared(B_shared_shape, in_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            if trans_A:
                T.copy(A[k * block_K, by * block_M], A_shared)
            else:
                T.copy(A[by * block_M, k * block_K], A_shared)
            if trans_B:
                T.copy(B[bx * block_N, k * block_K], B_shared)
            else:
                T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
        T.copy(C_local, C[by * block_M, bx * block_N])

return main
```

另一种是使用`tilelang.compile`:

```python
kernel = tilelang.compile(program, out_idx=-1, execution_backend="dlpack", target="cuda")
# 因为指定了-1是output tensor index，所以最后一个参数会在runtime的时候动态创建.
# execution backend 表示kernel使用dlpack来包装，默认的方法是cython.
C = kernel(A, B)
```

对于tilelang的jit来说，一般用户关注两部分overhead, 第一部分是`tilelang.compile`的时间，此处需要做一些代码编译的工作, 第二部分是runtime的overhead, 此处的runtime overhead不仅包括了前文中提到的ctypes转换指针，创建对象等overhead, 在这个基础上还包括了根据`out_idx`创建output tensor的overhead.

对于第一部分的代码编译开销，前文中提到，pybind的开销实在太大(dlpack, cython, ctypes)都只需要1-2s的时间就可以完成pack, 但是pybind加上torch的依赖实在需要太多文件，即使经过N多次的优化还是需要10s左右的时间，实在是太不又好了，评论区里有人提到了nanobind, 笔者尝试了一下, nanobind的出发点是pybind已经日积月累的变的非常臃肿，所以提出了一个更轻量级的方案，但是没有提供一个比较方便的类似`torch.cpp_extension.load`的接口，其次也没有一个比较方便的办法获取自动编译的nanobind的库, 为了compile还得codegen出一整个cmake project, 实在是太麻烦了，笔者尝试了很久，最后还是懒得弄了，其次Triton那个基于`PyObj`的方案，代码写起来非常ugly,懒得弄+1.

对于第二部分的开销，之前提到tilelang默认使用的是DLPack，在运行kernel的时候将torch的tensor再转换成DLPack的tensor，再转换成TVM Args的过程中会频繁调用tvm基于ctypes的ffi来call c++写的一些方法，开销实在太大，但是后来发现tvm的ffi其实有cython的实现，只不过因为当年的一些windows上支持不好的问题这个实现一直没有启动(甚至连个怎么enable的文档都没有)。尝试了之后发现dlpack+cython的runtime overhead也很小了，于是笔者就懒得再折腾了。直到有小伙伴提了issue才发现非常坑的一点是dlpack居然不能指定cuda stream, 于是笔者只能先把ctypes的backend实现了, 自己封装一个kernel的interface, 形态大概如下:

```cpp
// jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda", execution_backend="ctypes")
// print(jit_kernel.get_kernel_source())

#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void __launch_bounds__(128) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, int m) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    *(float2*)(C_local + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((i_1 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+((((((int)blockIdx.y) * 131072) + (i_1 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)), (((((((int)blockIdx.y) * 128) + (i_1 * 32)) + (((int)threadIdx.x) >> 2)) < m) && ((((((int)blockIdx.y) * 128) + (i_1 * 32)) + (((int)threadIdx.x) >> 2)) < m)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 8192), B+((((i_2 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 31; ++ko) {
    tl::cp_async_wait<0>();
    __syncthreads();
    tl::gemm_ss<128, 128, 32, 2, 2, 0, 0>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[4096])), (&(C_local[0])));
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((i_3 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), A+((((((((int)blockIdx.y) * 131072) + (i_3 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + (ko * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32), (((((((int)blockIdx.y) * 128) + (i_3 * 32)) + (((int)threadIdx.x) >> 2)) < m) && ((((((int)blockIdx.y) * 128) + (i_3 * 32)) + (((int)threadIdx.x) >> 2)) < m)));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 8192), B+((((((ko * 32768) + (i_4 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 32768));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<128, 128, 32, 2, 2, 0, 0>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[4096])), (&(C_local[0])));
  #pragma unroll
  for (int i_5 = 0; i_5 < 64; ++i_5) {
    if ((((((((int)blockIdx.y) * 128) + (((i_5 & 7) >> 1) * 32)) + (((((int)threadIdx.x) & 63) >> 5) * 16)) + ((i_5 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < m) {
      uint1 __1;
      float2 v_ = *(float2*)(C_local + (i_5 * 2));
      ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
      ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
      *(uint1*)(C + (((((((((((int)blockIdx.y) * 131072) + (((i_5 & 7) >> 1) * 32768)) + (((((int)threadIdx.x) & 63) >> 5) * 16384)) + ((i_5 & 1) * 8192)) + (((((int)threadIdx.x) & 31) >> 2) * 1024)) + (((int)blockIdx.x) * 128)) + ((i_5 >> 3) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
    }
  }
}


extern "C" void init() {
    
    cudaFuncSetAttribute(main_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384);

}

extern "C" void call(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, int m, cudaStream_t stream=cudaStreamDefault) {
if (m == 0) return; 
                main_kernel<<<dim3(8, (m + 127) / 128, 1), dim3(128, 1, 1), 16384, stream>>>(A, B, C, m);
}
```

代码主要有kernel本体，init(在实例化的时候调用，用来初始化一些dynamic smem的信息等), （因为如果n, k是dynamic的，需要用到tail split pass, 代码就会变得很长了，这里只列出m是dynamic的情况），call(实际调用kernel的函数)。

```python
def _forward_from_prebuild_lib(self, *args, stream: Optional[int] = None):
    """Low-level function to call the compiled CUDA kernel.
    
    Converts PyTorch tensor pointers to C void pointers for ctypes interface.
    """
    ctypes_args = [
        ctypes.c_void_p(arr.data_ptr()) if not isinstance(arr, int) else arr for arr in args
    ]
    ctypes_args.append(ctypes.c_void_p(stream))
    self.lib.call(*ctypes_args)

def _warp_forward_from_prebuild_lib(self,
                                    *ins: List[torch.Tensor],
                                    stream: Optional[int] = None):
    """High-level wrapper for kernel execution.
    
    Handles:
    1. Input validation
    2. Output tensor allocation
    3. Dynamic shape resolution
    4. CUDA stream management
    
    Args:
        ins: Input PyTorch tensors
        stream: Optional CUDA stream for asynchronous execution
    
    Returns:
        Single tensor or list of tensors containing the kernel results
    """
    if len(ins) + len(self.result_idx) != len(self.params):
        raise ValueError(
            f"Expected {len(self.params)} inputs, got {len(ins) + len(self.result_idx)} with {len(ins)} inputs and {len(self.result_idx)} outputs"
        )
    ins_idx = 0
    args = []

    # tensor pointers
    for i in range(len(self.params)):
        if i in self.result_idx:
            dtype = torch.__getattribute__(str(self.params[i].dtype))
            shape = list(map(int, self.params[i].shape))
            # use the device of the first input tensor if available
            device = ins[0].device if len(ins) > 0 else torch.cuda.current_device()
            tensor = torch.empty(*shape, dtype=dtype, device=device)
        else:
            tensor = ins[ins_idx]
            ins_idx += 1
        args.append(tensor)

    # dynamic symbolics
    for _, (buffer_idx, shape_idx) in self.dynamic_symbolic_map.items():
        args.append(ins[buffer_idx].shape[shape_idx])

    # if stream is not None, we need to pass the stream to the library
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream

    self._forward_from_prebuild_lib(*args, stream=stream)

    if len(self.result_idx) == 1:
        return args[self.result_idx[0]]
    else:
        return [args[i] for i in self.result_idx]
```

最核心就是把data_ptr转换成void_p, 然后调用lib.call, 这个lib.call就是前文提到的ctypes的call方法，当然外围还有一些初始化Stream，创建中间tensor，处理dynamic symbolics的逻辑，甚至这边最好还要做一些tensor attribute的check，这里交给python的来做解释执行总让我非常难受(总是担心解释器的性能很慢，最后拖累了tilelang的runtime，强迫症犯了)，于是试了一下把这部分代码变成提前compile的实现，当然就要用到cython!

Cython是python的静态编译器，可以用来把python代码编译成C代码，然后使用C的编译器编译成可执行文件，这个东西可以用 Python 的语法混合编写 Python 和 C/C++ 代码，提升 Python 速度，例如上述的代码，就可以被改造成cython的写法(老实说不是那么好写，折腾了我半天):

```python
# cython: language_level=3

import torch
cimport cython
import ctypes
from libc.stdint cimport int64_t, uintptr_t
from libc.stdlib cimport malloc, free

cdef class CythonKernelWrapper:
    # Class attributes to store kernel configuration and library reference
    cdef:
        object dynamic_symbolic_map  # Maps dynamic dimensions to their corresponding tensor indices
        list result_idx             # Indices of output tensors in the params list
        list params                 # List of parameter specifications (includes both inputs and outputs)
        object lib                  # Reference to the compiled library containing the kernel

    def __cinit__(self, dynamic_symbolic_map, result_idx, params, lib):
        # Initialize wrapper with kernel configuration
        self.dynamic_symbolic_map = dynamic_symbolic_map
        self.result_idx = result_idx
        self.params = params
        self.lib = lib

    cpdef forward(self, list inputs, int stream = -1):
        # Validate input dimensions and prepare for kernel execution
        cdef int total_params = len(self.params)
        cdef int total_inputs = len(inputs)
        cdef int total_result_idx = len(self.result_idx)
        cdef int total_dynamic_symbolics = len(self.dynamic_symbolic_map)

        # Ensure the number of inputs matches expected parameter count
        if total_params != total_inputs + total_result_idx:
            raise ValueError(
                f"Expected {len(self.params)} inputs, got {len(inputs) + len(self.result_idx)} with {len(inputs)} inputs and {len(self.result_idx)} outputs"
            )

        # Use current CUDA stream if none specified
        if stream == -1:
            stream = <uintptr_t>torch.cuda.current_stream().cuda_stream

        cdef int ins_idx = 0
        cdef list tensor_list = []
        cdef list call_args = []

        # Prepare input and output tensors
        for i in range(len(self.params)):
            if i in self.result_idx:
                # Create empty output tensor with specified dtype and shape
                dtype = torch.__getattribute__(str(self.params[i].dtype))
                shape = list(map(int, self.params[i].shape))
                device = inputs[0].device if len(inputs) > 0 else torch.cuda.current_device()
                tensor = torch.empty(*shape, dtype=dtype, device=device)
            else:
                # Use provided input tensor
                tensor = inputs[ins_idx]
                ins_idx += 1
            tensor_list.append(tensor)
        
        # Convert tensor pointers to C void pointers for kernel call
        call_args = [ctypes.c_void_p(tensor_list[i].data_ptr()) for i in range(len(tensor_list))]

        # Add dynamic dimension values to kernel arguments
        for _, (buffer_idx, shape_idx) in self.dynamic_symbolic_map.items():
            call_args.append(tensor_list[buffer_idx].shape[shape_idx])
        
        # Add CUDA stream to kernel arguments
        call_args.append(ctypes.c_void_p(stream))

        # Execute the kernel
        self.lib.call(*call_args)

        # Return output tensor(s)
        if len(self.result_idx) == 1:
            return tensor_list[self.result_idx[0]]
        else:
            return [tensor_list[i] for i in self.result_idx]
```

这个文件是一个`.pyx`为后缀的文件，我们需要先把这个文件编译成等效的`cython_wrapper.cpp`文件，然后使用C++的编译器编译成可执行文件，然后就可以在python中调用了.

```bash
# --cplus 表示使用C++的语法
cython cython_wrapper.pyx --cplus -o cython_wrapper.cpp

# 使用g++编译 -> 生成cython_wrapper.so
g++ -O3 -Wall -shared -std=c++11 -I /usr/include/python3.10 -fPIC cython_wrapper.cpp -o cython_wrapper.so
```

然后就可以在python中调用了:

```python
from cython_wrapper import CythonKernelWrapper
...
```

此时，forward内的内容都被静态编译优化过，速度会快了!(虽然没有快很多，但是治好了强迫症)。

但是cython亦有坑，不难发现cython的编译过程中包括`-I /usr/include/python3.10 `，和python的版本强绑定, 例如我在python3.10上compile了cython的so文件，那么python3.11上就无法使用, 为了解决这个问题, tilelang会在第一次初始化CythonKernelWrapper的时候，根据python的版本生成cython的so文件，然后做cache:

![image-20250221175614918](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/imgimage-20250221175614918.png)

如此，jit总算舒服很多了 :)
