---
title: CMake Notes
categories:
  - Technical
tags:
  - CMake
date: 2022-02-11 14:34:43
---

从来没见过语法像CMake这么烂的DSL，构建项目的时候总是要去查文档，但是查了文档还是不知道该怎么办💢，这里记一下自己常用的一些写法。

<!-- more -->

### 都塞到lib里吧！

如果我们希望生成的lib能够在制定的目录出现，那么我们需要在最顶层的CMakeList里加上：

```cmake
SET (LIBRARY_OUTPUT_PATH_ROOT ${PROJECT_SOURCE_DIR}/lib)
```

这样，我们在项目中生成的lib文件就会自动的生成到项目主目录下的lib目录了，如下：

![image-20220211145901447](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20220211145901447.png)

如果在Cmake里进行了lib的创建：

```cmake
add_library(${PROJECT_NAME} SHARED
        src/Interface.cpp
        src/tools.cpp
        src/Parser.cpp)
```

则该lib文件会在lib目录下产生。

### 创建cmake目录

一些项目的主目录下会有cmake这个目录，具体来讲里面会存放一些cmake文件，大概存放的内容一般有：

1. 一些cmake的指令文件，例如一般来讲检查环境来设置变量的check.cmake，用来引入外部库的xxx.cmake，使用的方法也很简单，直接include绝对路径即可

   ```cmake
   # Do check list
   include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/check.cmake")
   ```

   或者提前制定好cmake的module path，这样可以直接用文件名完成任务：

   ```cmake
   set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
   include(lexy)
   ```

2. 另外一种很常见的是在这个目录下再放一个module文件夹，里面存放的是一些查找Pakcage的Find脚本，例如`FindCUDNN.cmake`之类的，在CMakeList里通过`FIND_PACKAGE(CUDNN REQUIRED)`就会自动的调用FindCUDNN来找到CUDNN并设置XXX_FOUND、XXX_INCLUDE等变量。

   ```cmake
   find_package(Threads)
   target_link_libraries(calc-cpu-flops fpkernel ${CMAKE_THREAD_LIBS_INIT})
   ```

### 用变量表示目录下的一堆文件

```cmake
file(GLOB LIB_SOURCE "${CMAKE_CURRENT_LIST_DIR}/cpufp_kernel_*.s")
add_library(fpkernel ${LIB_SOURCE})
```

### 添加header目录

```cmake
include_directories(${PROJECT_SOURCE_DIR})
include_directories(
        ${PROJECT_SOURCE_DIR}/include
)
```
