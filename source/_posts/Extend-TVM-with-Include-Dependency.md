---
title: 通过Include依赖扩展TVM的核心代码
categories:
  - Technical
tags:
  - MLSYS
  - TVM
date: 2024-10-11 23:18:53
---

之前在一篇文章中我提到过一句：`一千个基于TVM的项目，就有一千个被爆改过的TVM`，这是我对基于TVM开发项目现状的吐槽。理解TVM的代码对于开发者来说已经是一件不容易的事情，更不用说开发者们在面对一个当前TVM无法解决的场景，想要修改进行扩展的时候是怎样的困难。往往，基于TVM的项目都是Fork一份TVM的代码来修改，例如为TVM添加一个新的优化Pass，就在`src/tir/transformation`文件夹下面新建一个Pass文件，然后通过`ffi`绑定到python侧的代码，其他的需求，例如注册一个新的语法树节点，添加新的代码生成等，也都是如此来实现,我自己的github上fork的[LeiWang1999/tvm](https://github.com/LeiWang1999/tvm)就包含十几个分支，有为了[BitBLAS](https://github.com/microsoft/BitBLAS)扩展(引入了一些新的Node和Schedule来进行优化)的bitblas分支，有为了[Ladder/Welder](https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact)做高性能的算子融合而添加了一些优化Pass的ladder分支，有为给AMD上做代码生产的`amd_hip`分支。这些分支的关系已经非常错综复杂了，我以[BitBLAS](https://github.com/microsoft/BitBLAS)为例，探讨一下为什么这样的开发方式会导致困难，并且提供一种解决方法(参考自[MLC-LLM](https://github.com/mlc-ai/mlc-llm))，供大家一起讨论，代码放在[LeiWang1999/TVM.CMakeExtend](https://github.com/LeiWang1999/TVM.CMakeExtend)。

<!-- more -->

首先，[BitBLAS](https://github.com/microsoft/BitBLAS)以submodule(位于项目的`3rdparty/tvm`路径下)的形式使用tvm，当然这里的tvm已经是我自己魔改过的，他的实际地址[LeiWang1999/tvm](https://github.com/LeiWang1999/tvm)的`bitblas_tl`分支，这个分支的TVM包括但不限于做了以下修改:

1. 引入了两三个非常hack的新的Schedule原语，用来更改语法树以获得更好的性能:

```python
with_scaling = bool(weight_dequantize_info["with_scaling"])
  if with_scaling:
      sch.unsafe_rewrite_buffer_region(
          dequantize_block,
          ("read", offset + 1),
          get_param_indices(intra_index_map),
      )
  with_zeros = bool(weight_dequantize_info["with_zeros"])
  if with_zeros:
      sch.unsafe_rewrite_buffer_region(
          dequantize_block,
          ("read", offset + 2),
          get_param_indices(intra_index_map),
      )
```

2. 改进和添加了TVM的几个Pass，例如针对Hopper架构的PartialSync以及TMA的支持，修复了一些目前TVM上游存在的bug.
3. 增加了自己的Codegen后端，例如针对HIP的`CodegenHIP`,针对`cute`的codegen等。

这样的修改虽然能够满足BitBLAS的需求，但是也会阻碍自己项目本身的发展，从自己的项目上来说，TVM的版本就被这样固定死了，每次想要使用上游的最新的功能，就得自己update；这样的场景还包括开发者给自己的项目提pr的时候，还得去给你fork的tvm提pr，非常之麻烦，徒增了维护的成本。另一方面，这样的方法对于TVM社区来说也不是好事，首先这样的修改很难被合并到TVM的上游代码中，因为一些Schedule的实现以及Pass的实现，甚至是一些bug的修复是比较hack的，代码并不雅观，而TVM的代码非常经典美观，大家应该都不想看到这样的代码被合并到TVM的上游代码中。。而合并到上游的过程也比较痛苦，各种test case和ci都得过，花在这上面的精力相比于自己的项目来说收益实在太小。

除此之外，也是最重要的一点，大家基于TVM做了很多有意思的工作，但是却不能合并到TVM里，这不也是一种遗憾吗？

对于这个场景，我心里一直有一个理想的方案，那就是维持一套核心的核心的TVM代码，大家通过引入tvm的方式来扩展TVM的功能，并且这种扩展是不需要被合并到TVM的上游代码中，如图：

![冲突](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/202410120044457.png)

例如用户写了一个TensorIR的算子表达式，我们可以使用TVM自己的Tunner去对算子进行调优，也可以使用我们自己的第三方的框架进行调优，例如用户获得了一个整个模型的Relax计算图，我们可以使用TVM自己的Pass进行优化，也可以使用用户自己实现的Pass，例如Welder进行优化，大家相互分隔开，互不影响。但是为什么现在这种硬基于tvm开发的方法行不通呢？例如我之前想在MLC-LLM的基础上引入Welder的优化:


```python
import tvm  # upstream

relax_mod = relax_transform(relax_mod)

import welder
relax_mod = welder.tune(relax_mod)
# something bad happened
```

程序会Segment Fault，最后Trace的时候发现，当welder被import的时候, welder会引入自己的tvm，这个时候会load相关的library, 比如说`cutlass`,覆盖了之前的tvm的library(使用的cutlass commit不一致)，导致了segment fault。

在TVM社区里的[Phasing out Legacy Components](https://discuss.tvm.apache.org/t/phasing-out-legacy-components/17703),tianqi对这个问题的看法是: 开发者应该避免在不同项目中直接修改 TVM 的源码，而是要确保所有下游项目（如 ProjectA 和 ProjectB）依赖于同一个核心 TVM 库。这样一来，大家可以在同一个 TVM 版本的基础上进行扩展，而不会因为各自的修改而导致兼容性问题。为此，他建议重构项目的组织方式，像 [MLC-LLM](https://github.com/mlc-ai/mlc-llm) 那样，将自定义的 Pass 和优化功能作为独立模块附加在 TVM 上，而不是直接对 TVM 做内联修改。这种做法能够通过模块化的方式来提升开发的灵活性和可维护性。未来，随着上游 API 逐渐变得更加模块化，社区希望不同项目能够更自然地基于相同的核心 TVM 进行各种变换和优化，从而在整个生态中形成更加有机的协作。

![tianqi response](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/202410120054359.png)


因此，我在这里学习了一下MLC-LLM的项目结构，并且借鉴和简化了一下，把一个基于TVM实现的小项目使用这种方式分离成一个单独的小项目: [LeiWang1999/TVM.CMakeExtend](https://github.com/LeiWang1999/TVM.CMakeExtend).


### 一种解决方案：通过CMake模块化构建TVM扩展

基于之前讨论的种种痛点，本文接下来介绍[LeiWang1999/TVM.CMakeExtend]里的CMake的构建方式，可以作为TVM模块化开发的参考方案。具体来说，我们将TVM视为一个独立的子模块，并且引入自定义的扩展逻辑，如TileLang（这里是一个示例的扩展模块），通过CMake的模块管理能力，将这些扩展与核心的TVM库解耦，从而实现灵活的功能组合与模块管理。

在设计模块化扩展方案时，我们遵循以下几个原则：

1.	核心代码与扩展代码分离：通过CMake的配置，我们能够将TVM核心库（tvm 和 tvm_runtime）与扩展模块（如 tilelang）的构建和管理分开。在CMake中，这种配置可以通过 add_subdirectory 或 IMPORTED 目标来实现，核心库与扩展模块的编译流程独立，彼此不会互相影响。
2.	灵活的路径管理：使用 TVM_SOURCE_DIR 和 TVM_PREBUILD_PATH 环境变量来定位TVM的核心代码，支持从源码构建和预构建二进制两种模式。这样做的好处是能够灵活地切换开发模式（例如，当上游TVM代码更新时，只需替换 TVM_PREBUILD_PATH 的路径即可）。
3.	最小化外部依赖冲突：特别是针对CUDA和其他第三方依赖项（如 cutlass），在项目中进行精细的依赖路径管理（如 target_include_directories），确保每个模块使用自己明确的依赖路径，避免动态链接时出现库覆盖的问题。

6.2 详细分析 CMake 配置文件

以下是 TileLang 项目的 CMake 配置文件的核心部分，并且对重要的代码片段进行了解释：

```cmake
if (DEFINED TVM_PREBUILD_PATH)
  message(STATUS "TVM_PREBUILD_PATH: ${TVM_PREBUILD_PATH}")
  # 当指定了 TVM 预构建路径时，直接将其作为共享库引入，而不是从源码构建
  add_library(tvm SHARED IMPORTED)
  set_target_properties(tvm PROPERTIES
    IMPORTED_LOCATION "${TVM_PREBUILD_PATH}/libtvm.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TVM_PREBUILD_PATH}/../include"
  )
  add_library(tvm_runtime SHARED IMPORTED)
  set_target_properties(tvm_runtime PROPERTIES
    IMPORTED_LOCATION "${TVM_PREBUILD_PATH}/libtvm_runtime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TVM_PREBUILD_PATH}/../include"
  )
else()
  # 当没有指定预构建路径时，使用源码进行编译
  message(STATUS "TVM_PREBUILD_PATH NOT SET, will build TVM from source")
  message(STATUS "TVM_SOURCE_DIR: ${TVM_SOURCE_DIR}")
  add_subdirectory(${TVM_SOURCE_DIR} tvm EXCLUDE_FROM_ALL)
endif()
```

这里的配置通过检查 TVM_PREBUILD_PATH 变量，判断是否需要从源码进行编译，还是直接使用预编译好的TVM二进制文件。对于开发者来说，这种方式能够极大地简化扩展模块与核心模块的构建流程，也能避免在不同版本之间切换时出现的编译问题。

例如，当系统已经存在一个编译好了的上游tvm,可能在python已经安装好的包里，可以找到`libtvm.so`, `libtvmruntime.so`等库文件，通过设置TVM_PREBUILD_PATH的方式直接链接到指定的库的位置，而不需要重新编译TVM。

```bash
git clone --recursive https://github.com/LeiWang1999/TVM.CMakeExtend

cd https://github.com/LeiWang1999/TVM.CMakeExtend

cmake .. -DTVM_PREBUILD_PATH=/your/path/to/tvm/build  # e.g., /workspace/tvm/build

make -j 16  # only build the source under src folder
```

这样，只有自己的源代码会被编译，不仅节省了编译时间，而且避免了与TVM上游代码的冲突。

其次, 用户也可以指定一个特定的TVM项目的位置进行编译:

```bash
git clone --recursive https://github.com/LeiWang1999/TVM.CMakeExtend

cd https://github.com/LeiWang1999/TVM.CMakeExtend

cp 3rdparty/tvm/cmake/config.cmake build

cd build

echo "set(USE_LLVM ON)" >> config.cmake

echo "set(USE_CUDA ON)" >> config.cmake

cmake ..

make -j 16 # build the source under src folder and tvm
# which may take a while
```

```cmake
# 定义 TileLang 模块的源文件
tilelang_file_glob(GLOB_RECURSE TILE_LANG_SRCS src/*.cc)
message(STATUS "TILE_LANG_SRCS: ${TILE_LANG_SRCS}")
add_library(tilelang_objs OBJECT ${TILE_LANG_SRCS})

# 设置 TileLang 模块的 include 路径
set(
  TILE_LANG_INCLUDES
  ${TVM_SOURCE_DIR}/include
  ${TVM_SOURCE_DIR}/src
  ${TVM_SOURCE_DIR}/3rdparty/dlpack/include
  ${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include
)
```

这一段代码将 TileLang 扩展模块的所有源文件（.cc 文件）添加到对象库 tilelang_objs 中，并设置了与TVM共享的 include 路径。对象库的使用能够让我们在不同的目标中重复利用相同的源文件，而不需要重新编译。

```cmake
# 生成共享库 `tilelang`
add_library(tilelang SHARED $<TARGET_OBJECTS:tilelang_objs>)
add_library(tilelang_static STATIC $<TARGET_OBJECTS:tilelang_objs>)
add_dependencies(tilelang_static tvm_runtime)
target_link_libraries(tilelang PUBLIC tvm_runtime)
```

通过生成共享库 tilelang 以及静态库 tilelang_static，用户可以根据需要灵活地选择不同的链接方式（静态链接或动态链接），并且 tilelang 模块会自动依赖 tvm_runtime，保证在编译时能够正确地找到相关符号和依赖项。

### 在cpp通过TVM的核心API编写自己的扩展模块

在[src](https://github.com/LeiWang1999/TVM.CMakeExtend/tree/main/src)下面, 我放置了对tvm扩展的例子，包括优化的pass，新的codegen等，在自己写的代码中，我们引入TVM的头文件，编写自己的Pass:

```cpp
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>
```

在最后，我们注册一个全局接口，以供python端进行调用:

```cpp
tvm::transform::Pass Simplify() {
    auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
        arith::Analyzer analyzer;
        auto cfg = ctx->GetConfig<SimplifyConfig>("tl.Simplify");
        return StmtSimplifier::Apply(f, &analyzer, cfg);
    };
    return CreatePrimFuncPass(pass_func, 0, "tl.Simplify", {});
}

TVM_REGISTER_GLOBAL("tl.transform.Simplify").set_body_typed(Simplify);
```

这样在python端就可以通过`tvm.tl.transform.Simplify`来调用这个pass优化tensorir的语法树了。

### 在项目init的时候为TVM扩展模块动态查找与加载自定义库文件

在基于 TVM 的扩展模块中，我们经常需要将自定义的动态链接库（DLL）或共享库（如 .so、.dll 或 .dylib）与现有的 TVM 运行时进行集成。这种情况下，一个有效的全局入口点机制能够帮助我们动态查找这些库文件，并确保它们能够被正确加载到系统中。这篇博客将详细讲解一个典型的库查找与加载的实现——包括路径的动态定位、多平台的兼容处理、以及如何优雅地处理环境变量，确保我们在复杂的开发环境中能够稳定地找到所需的库文件，在这个例子中，做这段时期的代码是[libinfo.py](https://github.com/LeiWang1999/TVM.CMakeExtend/blob/tilelang/python/tilelang/libinfo.py), 这段代码的核心功能是实现自定义库(如刚刚编译出来的本项目的libxxx.so)的自动查找与路径定位，当然也是大部分借鉴自[MLC-LLM](https://github.com/mlc-ai/mlc-llm)，供大家一起讨论。

**Note: 这里load lib的过程要发生在import tvm之前, 因为实际的全局函数的注册过程发生在tvm的import的时候，这个时候lib必须已经被load进来了，否则自己写的全局函数就无法被tvm的import过程注册到tvm的全局函数表中。**


例如，直接`import tvm`:

```python
from tvm._ffi.registry import list_global_func_names
print(list_global_func_names())
'''
'tir.analysis.find_anchor_block', ... , 'relay._transform.ToBasicBlockNormalForm', 'relay._transform.to_cps', 'relay._transform.ToCPS', 'topi.bitwise_or', 'relay._transform.InferTypeLocal', 'relay.backend.aot.CreateExecutorMetadata', 'relay.build_module._AOTExecutorCodegen', 'relay.build_module._BuildModule', 'tvm_callback_cuda_compile', 'relay.backend.CreateExecutor']
'''
```

先load lib再`import tvm`:

```python
import bitblas
from bitblas import tvm as tvm
from tvm._ffi.registry import list_global_func_names
print(list_global_func_names())
'''
'tl.transform.Simplify', ..., 'tir.analysis.find_anchor_block', 'tir.analysis.find_anchor_block', ...'relay._transform.InferTypeLocal', 'relay.backend.aot.CreateExecutorMetadata', 'relay.build_module._AOTExecutorCodegen', 'relay.build_module._BuildModule', 'tvm_callback_cuda_compile', 'relay.backend.CreateExecutor']
'''
```

这里还有一个细节, 在整个项目的`__init__.py`中，我们还需要设置好环境变量，以便在运行时能够正确地找到tvm python interface的地址。

```python
# Handle TVM_IMPORT_PYTHON_PATH to import tvm from the specified path
TVM_IMPORT_PYTHON_PATH = os.environ.get("TVM_IMPORT_PYTHON_PATH", None)

if TVM_IMPORT_PYTHON_PATH is not None:
    os.environ["PYTHONPATH"] = TVM_IMPORT_PYTHON_PATH + ":" + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, TVM_IMPORT_PYTHON_PATH + "/python")
else:
    install_tvm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3rdparty", "tvm")
    if os.path.exists(install_tvm_path) and install_tvm_path not in sys.path:
        os.environ["PYTHONPATH"] = install_tvm_path + "/python:" + os.environ.get("PYTHONPATH", "")
        sys.path.insert(0, install_tvm_path + "/python")

    develop_tvm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "3rdparty", "tvm")
    tvm_library_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build", "tvm")
    if os.path.exists(develop_tvm_path) and develop_tvm_path not in sys.path:
        os.environ["PYTHONPATH"] = develop_tvm_path + "/python:" + os.environ.get("PYTHONPATH", "")
        sys.path.insert(0, develop_tvm_path + "/python")
    if os.environ.get("TVM_LIBRARY_PATH") is None:
        os.environ["TVM_LIBRARY_PATH"] = tvm_library_path

import tvm as tvm
```

所以在之前的例子上, 我们使用了`from bitblas import tvm as tvm`来确保我们的tvm的python接口和我们link的lib是一致的，不过这个是为了谨慎，一般而言应该不需要。


### 总结

本文介绍了一种基于 CMake 的模块化开发方案，例子参考[LeiWang1999/TVM.CMakeExtend](https://github.com/LeiWang1999/TVM.CMakeExtend)，用于构建 TVM 的扩展模块，帮助和我一样基于TVM开发项目的同学们一点小启发。通过将 TVM 核心库与自定义扩展模块相分离，我们不仅能够更加灵活地管理和组合功能，还避免了直接修改 TVM 源码所带来的种种问题。试想一下，当我们安装基于 TVM 的项目时，不需要单独编译一份 TVM，只需通过 `pip install apache-tvm` 安装最上游版本的 TVM，然后再使用 `pip install xxx` 安装自己的扩展模块。从源码安装时，也无需构建完整的 TVM 代码，只需先用 `pip install apache-tvm` 下载最新的 TVM 发行版，扩展模块的库文件就能自动链接到已安装的 TVM 包中，大家如果都这么做，tvm的体验应该会变得美好一些 (
