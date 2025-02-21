---
title: CUDA CPP to Torch Executable
categories:
  - Technical
tags:
  - MLSys
date: 2025-02-07 16:26:23
---

很久之前笔者碰到过一个还挺棘手的问题: 怎么优雅的在运行时阶段自动把一个tvm生成的CUDA C文件封装成一个Torch Function? 作者大概尝试过三种方法，各有利弊：

1. PyBind Torch C++ Extension: 最常见的是通过torch基于pybind开发的cpp extension，也是大部分库对接torch的时候使用的方法，但是这种方法的编译时长太久，在runtime下(比如tilelang/bitblas)这种编译的overhead会让用户体验明显变差。
2. DLPack: DLPack应该是最直接的方式了，先将 Torch 张量转换为 DLPack 格式，然后再将其转换成 TVM 所需的参数。DLPack 的优点是使用时几乎无感，不需要像 PyBind 那样等待数十秒的编译时间。但它也存在一个不足：由于频繁通过 ctypes 进行调用，额外的运行时开销在小 kernel 场景下甚至可能超过 kernel 本身的执行时间。
3. 静态编译与指针传递: 最Hack的方法是将 CUDA 源码提前静态编译成库文件，然后在 Python 端直接传递指针进行调用。虽然这样可以省去运行时的编译时间，也是bitblas目前使用的方法，但是在cpp侧损失了张量信息，没有办法做高性能的张量属性check(如果把这些放到python侧来做，则又会引入新的overhead)。

本文接下来再详细分享一下这三种方法的利弊与笔者做出的一些尝试与改进，包括怎么尽可能的减少torch cpp extension的编译overhead，怎么分析dlpack的overhead等，希望能够帮助到之后踩坑的同学以及供大家讨论 :) 代码放在[CPPTorchExecutable](https://github.com/LeiWang1999/CPPTorchExecutable)

<!-- more -->

### CPP Extension的基础用法与优化

cpp extension是torch提供的一种用于将cpp代码封装成python模块的方法，其实现基于pybind11，使用cpp extension可以让用户在python中调用cpp代码，这样可以提高代码的运行效率，以下是本章节做出的一些优化以及结果:

| 编译方法                                    | CPP Extension 编译时间 |
| ------------------------------------------- | ---------------------- |
| 单文件大杂烩 (single file)                  | 144.90 s               |
| 指定cuda arch减少目标文件 (specify_arch)    | 45.28 s                |
| CUDA和CPP Extension分离 (separate_cuda_cpp) | 15.94 s                |
| 减少头文件依赖 (alleviate_dependency)       | 8.27 s                 |

#### 1. 单文件大杂烩版

最简单直观的情况，就是把CUDA Kernel以及pybind的部分都包装在一个文件内:

```c++
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

void square_cuda_forward(void* input, void* output, int size);

__global__ void square_kernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * input[idx];
    }
}

void square_cuda_forward(void* input, void* output, int size) {

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    square_kernel<<<blocks, threads>>>(
        static_cast<const float*>(input),
        static_cast<float*>(output),
        size
    );
}

void square_cuda_forward(void *input, void *output, int size);

torch::Tensor square_forward(torch::Tensor input)
{
    auto output = torch::zeros_like(input);

    square_cuda_forward(input.data_ptr(), output.data_ptr(), input.numel());

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("square_forward", &square_forward, "Square forward (CUDA)");
}
```

接着在python中如此进行pybind和编译:

```python
start = time.time()

square_cuda = load(
    name="square_cuda",
    sources=[f"{dir_path}/square_kernel.cu"],
    verbose=True,
    build_directory=build_dir
)

end = time.time()
print(f"Time taken: {end - start} seconds")
```

最后的时间开销为 144.90 s，开销是非常大. 通过`verbose=True`可以看到编译的过程:

```bash
[1/2] /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output square_kernel.cuda.o.d -DTORCH_EXTENSION_NAME=square_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /usr/local/lib/python3.10/dist-packages/torch/include -isystem /usr/local/lib/python3.10/dist-packages/torch/include/torch/csrc/api/include -isystem /usr/local/lib/python3.10/dist-packages/torch/include/TH -isystem /usr/local/lib/python3.10/dist-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /usr/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_90,code=compute_90 -gencode=arch=compute_90,code=sm_90 --compiler-options '-fPIC' -std=c++17 -c /root/CPPTorchExecutable/torch_cpp_extension/0.single_file/square_kernel.cu -o square_kernel.cuda.o 
[2/2] c++ square_kernel.cuda.o -shared -L/usr/local/lib/python3.10/dist-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o square_cuda.so
Loading extension module square_cuda...
Time taken: 149.87586188316345 seconds
```

#### 2. 指定cuda arch减少目标文件
不难发现，之前的编译过程中，nvcc会生成很多不同的目标文件，这些目标文件会被链接到最终的.so文件中，这样会导致编译时间过长。我们可以通过指定`TORCH_CUDA_ARCH_LIST`来减少目标文件的数量，从而减少编译时间:

```python
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

start = time.time()

square_cuda = load(
    name="square_cuda",
    sources=[f"{dir_path}/square_kernel.cu"],
    verbose=True,
    build_directory=build_dir
)

end = time.time()
print(f"Time taken: {end - start} seconds")
```

此时，程序的输出为:

```bash
[1/2] /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output square_kernel.cuda.o.d -DTORCH_EXTENSION_NAME=square_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /usr/local/lib/python3.10/dist-packages/torch/include -isystem /usr/local/lib/python3.10/dist-packages/torch/include/torch/csrc/api/include -isystem /usr/local/lib/python3.10/dist-packages/torch/include/TH -isystem /usr/local/lib/python3.10/dist-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /usr/include/python3.10 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_89,code=sm_89 --compiler-options '-fPIC' -std=c++17 -c /root/CPPTorchExecutable/torch_cpp_extension/1.specify_arch/square_kernel.cu -o square_kernel.cuda.o 
[2/2] c++ square_kernel.cuda.o -shared -L/usr/local/lib/python3.10/dist-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o square_cuda.so
Loading extension module square_cuda...
Time taken: 47.40068602561951 seconds
```

可以看到编译时间减少到了45.28 s，这是因为我们只指定了一个cuda arch，减少了目标文件的数量。

#### 3. CUDA和CPP Extension分离

经过进一步的实验，笔者发现，将CUDA和CPP Extension分离开来，可以进一步减少编译时间(也就是不让nvcc编译torch extension的内容)，我们将cuda kernel和cpp extension分开成为两个文件`square_kernel.cu`和`square.cpp`之后:

```python
square_cuda = load(
    name="square_cuda",
    sources=[f"{dir_path}/square_kernel.cu", f"{dir_path}/square.cpp"],
    verbose=True,
    build_directory=build_dir
)
```

此时，编译时间减少到了15.94 s，这是因为nvcc只需要编译CUDA的部分，而不需要编译torch extension的部分。

#### 4. 减少头文件依赖

最后，我们可以进一步减少头文件的依赖，这样可以减少编译时间，观察到`square.cpp`中通过`#include <torch/extension.h>`引入了完整的extension的内容，其实我们只需要张量部分(方便拿到`Tensor`信息，以及`data_ptr`等方法)，其次就是有关pybind的部分，于是我们可以将`square.cpp`改写为:

```c++
#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <torch/csrc/utils/pybind.h>

void square_cuda_forward(void* input, void* output, int size);

at::Tensor square_forward(const at::Tensor& input) {
    auto output = at::zeros_like(input);

    square_cuda_forward(input.data_ptr(), output.data_ptr(), input.numel());
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square_forward", &square_forward, "Square forward (CUDA)");
}
```

此时，编译时间减少到了8.27 s, 此时笔者已经优化不下去了，虽然这个时间相比于原来已经快了将近20倍，但是在运行时仍然会让用户明显的感受到程序的停顿(

### DLPack的使用与优化

DLPack 是一种开放的、轻量级的张量数据交换格式，用于在不同的深度学习和科学计算框架（如 PyTorch、TensorFlow、MXNet、JAX、CuPy 等）之间高效共享张量数据，而无需进行额外的数据拷贝, 在PyTorch中，我们可以通过`torch.utils.dlpack`来将张量转换为DLPack格式，然后再将其转换为TVM所需的参数。

```python
tvm_nd_array_tensors = [
    tvm.runtime.ndarray.from_dlpack(to_dlpack(torch_tensor))
    for torch_tensor in torch_tensors
]
rt_mod(*tvm_nd_array_tensors)
```

使用dlpack来完成张量的转换理论上是非常方便的，因为在这个过程中并没有实际的数据拷贝，而是根据指针和张量形状等重新构建TVM的NDArray对象，但是在实际的运行中，笔者发现这种方法的运行时间会比较长，这是因为在每次调用时都会通过ctypes来调用，这个过程会带来额外的运行时开销，这个开销在小kernel场景下甚至可能超过kernel本身的执行时间。

在论坛的帖子: https://discuss.tvm.apache.org/t/strange-overhead-of-tvm-runtime-ndarray-from-dlpack/16516 中，笔者在对一个仅耗时数微秒的小 Kernel（BitBLAS 1×1024×1024×fp16xint4b on A100）进行测试时，发现从 DLPack 构造 TVM NDArray 的过程和 Python 端的调用方式会带来不容忽视的运行时开销，这些开销在小 Kernel 场景下甚至可能超过 Kernel 本身的执行时间。为此，笔者对比了几种不同的调用方式及其耗时，结果如下：

1. **使用 `time_evaluator`**  
   首先，利用 `rt_mod.time_evaluator` 对 Kernel 的纯执行时间进行测试，结果显示在 A100 上平均只需 4 微秒左右。可见实际的计算部分非常快。

```python
time_evaluator = rt_mod.time_evaluator(rt_mod.entry_name, tvm.cuda(), number=1000000)
latency = time_evaluator(*tvm_nd_array_tensors).mean * 1e6
print("rt_mod time_evaluator Time: {:.2f} us".format(latency))
# rt_mod time_evaluator Time: 4.39 us
```

2. **直接在 Python 循环中调用（无动态 DLPack 转换）**  
   如果不在每次调用时都重新构造 TVM NDArray，而是将构造好的 TVM NDArray 对象传入 `rt_mod`，耗时大约为 13 微秒。

```python
# 先构造好 tvm_nd_array_tensors
for _ in range(1000):
    rt_mod(*tvm_nd_array_tensors)

start = time.time()
for _ in range(1000000):
    rt_mod(*tvm_nd_array_tensors)
end = time.time()
print("rt_mod only Time: {:.2f} us".format(float(end - start)))
# rt_mod only Time: 13.44 us
```
   可以看到，尽管 Kernel 本身只有 4 微秒左右，但实际完整调用却达到了 13 微秒上下，说明仅从 Python 层面调用（包括函数检索、参数打包等）也带来了额外的开销。

3. **在每次调用时重新构造 DLPack 并转换成 TVM NDArray**  
   若每次调用都动态执行
```python
dlpack_tensors = [to_dlpack(torch_tensor) for torch_tensor in torch_tensors]
tvm_nd_array_tensors = [
    tvm.runtime.ndarray.from_dlpack(dlpack_tensor)
    for dlpack_tensor in dlpack_tensors
]
```
   然后再执行 `rt_mod`，最终在测量中耗时达到了约 53 微秒：
```python
# warmup
for _ in range(1000):
    dlpack_tensors = [to_dlpack(torch_tensor) for torch_tensor in torch_tensors]
    tvm_nd_array_tensors = [
        tvm.runtime.ndarray.from_dlpack(dlpack_tensor)
        for dlpack_tensor in dlpack_tensors
    ]
    rt_mod(*tvm_nd_array_tensors)

start = time.time()
for _ in range(1000000):
    dlpack_tensors = [to_dlpack(torch_tensor) for torch_tensor in torch_tensors]
    tvm_nd_array_tensors = [
        tvm.runtime.ndarray.from_dlpack(dlpack_tensor)
        for dlpack_tensor in dlpack_tensors
    ]
    rt_mod(*tvm_nd_array_tensors)
end = time.time()
print("rt_mod with dlpack Time: {:.2f} us".format(float(end - start)))
# rt_mod with dlpack Time: 53.40 us
```
   这说明每次调用都需要多次的 Python 端封装、解封装以及底层的 CTypes 调用，导致在极小的 Kernel 上额外成本更为突出。

4. **直接从指针构造 TVM NDArray**  
   为了减少这部分开销，笔者尝试了直接从裸指针构造 TVM NDArray。这样做可以在一定程度上减轻每次通过 `to_dlpack -> from_dlpack` 的操作，但仍需要一些 Python→C 的调用过程。结果显示耗时大约为 20 微秒左右：
```python
# 省略前面获取 function_handle、TVMValue 等上下文的代码
time_arr = []
for _ in range(100):
    start = time.time()
    for _ in range(1000):
        for i, torch_tensor in enumerate(torch_tensors):
            attr_handle = TVMArrayHandle()
            data = ctypes.cast(torch_tensor.data_ptr(), ctypes.c_void_p)
            check_call(
                _LIB.TVMArrayFromDataPointerOnly(
                    data,
                    device,
                    ctypes.byref(attr_handle),
                )
            )
            values[i].v_handle = ctypes.cast(attr_handle, ctypes.c_void_p)

        check_call(
            _LIB.TVMFuncCall(
                function_handle,
                values,
                tcodes,
                ctypes.c_int(num_args),
                ctypes.byref(ret_val),
                ctypes.byref(ret_tcode),
            )
        )
    torch.cuda.synchronize()
    end = time.time()
    time_arr.append(end - start)
print("Overall Time: {:.2f} us".format(sum(time_arr) / len(time_arr) * 1e3))
# 结果大约在 20+ us 左右
```
   相比之前的 53 微秒已经大幅下降，但相比起单纯 Kernel 的 4 微秒仍有明显差距。

综合以上结果，可以得出以下结论：在非常小的 Kernel（几微秒级别）场景中，**任何来自 Python 和运行时的额外调用开销（例如通过ctypes调用一个cpp的函数）** 都可能比计算本身更大；  

其次，rt_mod本身也会有一些overhead, 这当然也是来源于python和cpp之间的调用开销，因为rt_mod需要进一步把tvm的NDArray转换为tvm args.

使用这种方法，其实就将Kernel托管给了tvm的runtime, 在tvm的代码生成中，其会在生成的host代码中生成关于tensor属性的check，这部分代码其实非常优雅和美好，只是他针对的输入是tvm自己的arg对象而不是torch的tensor..

### 静态编译与指针传递

为了进一步研究，我将其生成的CUDA源码通过nvcc编译成静态库，然后再python这使用ctypes直接传指针计算:

```python
import time
import ctypes

lib = warpper.load_lib()
torch.cuda.synchronize()
time_arrs = []
for __ in range(100):
    start = time.time()
    for _ in range(1000):
        lib.call(*[ctypes.c_void_p(arr.data_ptr()) for arr in torch_tensors])
    torch.cuda.synchronize()
    end = time.time()
    time_arrs.append(end - start)
print("time arrs: ", time_arrs)
print("lib only Time: ", sum(time_arrs) / len(time_arrs))
print(output_tensor)
# lib only Time:  4.469914436340332  us
```

可以观察到，此时基本没有overhead, 也就是cuda kernel的性能本身没有问题, 以及`time evaluator`的时间是可信的，目前BitBLAS使用该方法自动生成可以被包装成library的代码，然后在python侧直接传递指针进行调用，这样可以省去运行时的编译时间，但是在cpp侧损失了张量信息，没有办法做高性能的张量属性check。

### 总结

帮助大家总结一下该文，如果你是开发了一个静态的library，那么使用cpp_extension的方法当然是最好的，因为这部分编译的开销并不需要用户在运行时的时候来承担，但是在jit阶段使用会让用户感受到明显的卡顿。对于只关心kernel性能，暂时不关心和torch integrate的同学，那么dlpack是最方便的路径了，最后折中的方案就是ctypes+dll, 这样可以省去运行时的编译时间，但是在cpp侧损失了张量信息，没有办法做高性能的张量属性check。在BitBLAS里，我们选择了第三者，在tilelang中，我们选择了dlpack于torch extension的结合，在调优和一般调试过程中使用dlpack，当通过jit转换成一个torch function的时候使用torch extension，这样可以让用户在运行时不感受到编译的overhead，同时也可以保留张量的信息，做一些高性能的张量属性check（当然我觉得torch extension的编译时间可能还可以进一步优化)。
