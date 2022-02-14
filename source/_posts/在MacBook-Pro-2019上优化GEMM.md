---
title: 在MacBook Pro 2019上优化GEMM
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2022-02-12 17:23:39
---

<!-- more -->

### 1.浮点峰值计算

获取MacBook的CPU信息：

```bash
❯ sysctl machdep.cpu.brand_string
machdep.cpu.brand_string: Intel(R) Core(TM) i5-8257U CPU @ 1.40GHz
```

得到CPU的型号是`i5-8257U`，Google一下找到他的[Product页面](https://ark.intel.com/content/www/us/en/ark/products/191067/intel-core-i58257u-processor-6m-cache-up-to-3-90-ghz.html)。

首先关注一下Specfication，该CPU一共有四个核，八个线程，但是因为Intel的超线程技术（就是一个核当作两个核来用，我们这种两个线程都吃慢计算单元资源的程序吃不到这个超线程的福利，所以理论上两个线程的吞吐量和四个线程应该是一样的，所以GFLOPS要以四个线程来算）；这里的Boost Technology是指Intel的睿频技术，是指在程序对CPU的资源利用增加时会主动提升CPU运行的频率，并且在睿频加速期间，处理器的温度，功率等超过限制，则会处理器的时钟频率会下降，以保护处理器。我们在这里不关注睿频的频率，关注处理器的Base频率也就是1.4Ghz。

![image-20220212173146231](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20220212173146231.png)

其次，关注CPU支持的向量指令集（就是说一个指令可以操作向量，即很多个标量被一起操作，比如数组的乘法与加法），对于`i5-8257U`来说，其支持的扩展指令如下：
![image-20220212191541540](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20220212191541540.png)

关于CPU支持什么样的指令，也可以通过CPU_ID这个指令来检查，在代码里这样写，关于指令和CPUID的简单用法，可以参考[这篇文章](https://zhuanlan.zhihu.com/p/31271788)：

```c
static void cpuid_x86_exec(struct cpuid_t *cpuid) {
    asm volatile("pushq %%rbx\n"
                 "cpuid\n"
                 "movl %%ebx, %1\n"
                 "popq %%rbx\n"
                 : "=a"(cpuid->eax), "=r"(cpuid->ebx), "=c"(cpuid->ecx), "=d"(cpuid->edx)
                 : "a"(cpuid->ieax), "c"(cpuid->iecx)
                 : "cc");
}

    struct cpuid_t cpuid;

    feat = 0;

    cpuid.ieax = 0x1;
    cpuid.iecx = 0x0;
    cpuid_x86_exec(&cpuid);

    if (BIT_TEST(cpuid.edx, 25)) {
        SET_FEAT(_CPUID_X86_SSE_);
    }
    if (BIT_TEST(cpuid.ecx, 28)) {
        SET_FEAT(_CPUID_X86_AVX_);
    }
    if (BIT_TEST(cpuid.ecx, 12)) {
        SET_FEAT(_CPUID_X86_FMA_);
    }

    cpuid.ieax = 0x7;
    cpuid.iecx = 0x0;
    cpuid_x86_exec(&cpuid);

    if (BIT_TEST(cpuid.ebx, 16)) {
        SET_FEAT(_CPUID_X86_AVX512F_);
    }
    if (BIT_TEST(cpuid.ecx, 11)) {
        SET_FEAT(_CPUID_X86_AVX512_VNNI_);
    }
```

这样，就可以在程序中获得当前的X86 CPU支持的指令集，并分别调用不通的程序来运算，跑了一下这个程序发现他是支持FMA、AVX、SSE的。

为了计算浮点峰值还需要一个参数就是同一时间段能够处理多少个浮点计算指令，这需要研究一下处理器的微结构，但是我在官方的datasheet里没找到有关的描述。在果壳的体系结构课上领悟到的查chip的各种参数与他们之间的各种联系，还是去wikichip这个网站。

找到wikichip里关于i5-8250U的描述页面：https://en.wikichip.org/wiki/intel/core_i5/i5-8250u

我们的8257和这个8250在微架构上是一样的。

> 关于intel的命名规则：
>
> 第一位代表第几代CPU，一般越大，架构更优。i7-4770K>i7-3770K
>
> 第二位代表处理器等级，数字越大，性能越好。i7-4810mq>i7-4710mq
>
> 第三位代表核显，可忽略不比
>
> 第四位代表功耗可忽略不比
>
> H，M,U，表示功耗，字母越小，功耗越大，性能越好。所以后缀：H>M>U。比如：i5-5350H>i7-4610m,i5-4330m>i7-4558U

在页面描述上可以看到其微结构属于**Kaby Lake**！再进Kaby Lake的Wiki页面，可以看到单核的微结构图：

![skylake block diagram.svg](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/20220212233321.png)



PORT0和PORT1可以同时执行向量乘、向量加和FMA指令，通过查datasheet可以知道：

FMA指令集（floating-point fused multiply add）就是256位的融合乘加指令，对于32位的数据来说一个指令可以操作8个操作数，对于64位的数据来说，一个指令可以操作4个操作数，那么对于fp32的数据来说，我们的单个浮点峰值就是：
$$
2*(256/32)*2*1.4=44.8 \quad GFLOPS
$$
第一个2是两个port、第二个是操作数位fp32的时候的操作数个数，第三个2是乘法和加法两个运算，第三个1.4是没有睿频时的工作频率。

> Mac上需要使用到 Turbo Boost Switcher Pro这个软件来关闭睿频，不过我的mbp2019日常开启睿频，风扇疯狂起飞XDD

实际编写程序测试最大峰值：

```bash
gemm-optimizer/cmake-build-debug/tools/calc-cpu-flops 4
```

这里，分别对几个向量指令编写了测试的汇编程序：

```bash
❯ tree
.
├── CMakeLists.txt
├── calc-cpu-flops.c
├── cpufp_kernel_x86.h
├── cpufp_kernel_x86_avx.s
├── cpufp_kernel_x86_avx512_vnni.s
├── cpufp_kernel_x86_avx512f.s
├── cpufp_kernel_x86_fma.s
├── cpufp_kernel_x86_sse.s
├── smtl.c
└── smtl.h
```

关于汇编程序的内容，看一下之前分析的fma的程式：

```assembly
_cpufp_kernel_x86_fma_fp32:
    mov $0x40000000, %rax
    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm1, %ymm1, %ymm1
    vxorps %ymm2, %ymm2, %ymm2
    vxorps %ymm3, %ymm3, %ymm3
    vxorps %ymm4, %ymm4, %ymm4
    vxorps %ymm5, %ymm5, %ymm5
    vxorps %ymm6, %ymm6, %ymm6
    vxorps %ymm7, %ymm7, %ymm7
    vxorps %ymm8, %ymm8, %ymm8
    vxorps %ymm9, %ymm9, %ymm9
._cpufp.x86.fma.fp32.L1:
    vfmadd231ps %ymm0, %ymm0, %ymm0
    vfmadd231ps %ymm1, %ymm1, %ymm1
    vfmadd231ps %ymm2, %ymm2, %ymm2
    vfmadd231ps %ymm3, %ymm3, %ymm3
    vfmadd231ps %ymm4, %ymm4, %ymm4
    vfmadd231ps %ymm5, %ymm5, %ymm5
    vfmadd231ps %ymm6, %ymm6, %ymm6
    vfmadd231ps %ymm7, %ymm7, %ymm7
    vfmadd231ps %ymm8, %ymm8, %ymm8
    vfmadd231ps %ymm9, %ymm9, %ymm9
    sub $0x1, %rax
    jne ._cpufp.x86.fma.fp32.L1
    ret
```

`rax`存储的是循环的次数，循环初始化的时候执行的一系列异或指令是用来快速将寄存器的内容置零，接着循环内会执行十次乘累加运算，一共执行40000000次，之前分析得，对于32位的浮点数， 一次乘累加运算可以操作8个数，一共是 40000000 \* 8 * 10 * 2（乘法和加法）运算，共这么大的浮点运算量：

```c
#ifdef _FMA_
#define FMA_FP32_COMP (0x40000000L * 160)
#define FMA_FP64_COMP (0x40000000L * 80)
```

则gflops的计算如下式：

```c
perf = FMA_FP64_COMP * num_threads / time_used * 1e-9;
```

**这里有个容易忽略的点，在MacOS上，C/CPP 调用汇编的约定是要在汇编函数前面加个下划线，这样才能对C程序可见，大坑！**

因为mac上运行的程序有点多，所以这里的运算一般是吃不满cpu的资源的，我在这里给一下单核睿频关闭情况下跑出来的perf，这将作为我们调优的目标：

```bash
Thread(s): 1
binding to core 0
fma fp32 perf: 40.9189 gflops.
fma fp64 perf: 22.0921 gflops.
binding to core 0
avx fp32 perf: 21.9944 gflops.
avx fp64 perf: 11.0741 gflops.
binding to core 0
sse fp32 perf: 10.8526 gfops.
sse fp64 perf: 5.4786 gflops.
```

之前手算单核FMA的GFLOPS是44.8，差的也不是很多，基本上算是正确了。
