As discussed in [Phasing out Legacy Components](https://discuss.tvm.apache.org/t/phasing-out-legacy-components/17703), Third-party developers often choose to directly apply inplace modification to TVM rather than contributing their changes upstream for several reasons. First, TVM’s codebase is complex, and understanding or modifying it requires significant effort. Developers frequently face scenarios where TVM’s existing capabilities cannot meet their specific optimization needs, such as adding custom schedules, transformation passes, or backends for certain hardware architectures. These custom modifications are often too specific or “hacky” to meet the high code quality and design standards required by the TVM community, making it difficult for such changes to be merged upstream. Furthermore, the process of contributing upstream can be cumbersome and time-consuming, requiring rigorous testing and CI checks, which may outweigh the benefits for individual projects. Additionally, developers often lock their forks to specific versions of TVM to stabilize their custom modifications, making it harder to keep up with upstream updates. As a result, it is easier and faster for developers to maintain their own fork rather than engage in the lengthy and complex process of merging code upstream. Finally, the diverse nature of TVM-based projects means that different forks often have highly specialized code, which is not always applicable to the broader community, further reducing the motivation to merge changes back into TVM’s mainline codebase.

As @tianqi mentioned in the discussion, developers are recommended to avoid directly modifying TVM’s core code in their individual projects. Instead, the goal is to ensure that all downstream projects (such as ProjectA and ProjectB) can rely on a shared, unmodified core TVM library. This approach prevents compatibility issues that arise when each project forks and customizes TVM independently. To achieve this, @tianqi suggests taking inspiration from projects like [MLC-LLM](https://github.com/mlc-ai/mlc-llm), where custom passes and optimizations are added as separate, modular extensions rather than inline modifications to TVM itself (a perfect example). 

In this thread, let me summarize the key points of this approach and share a workflow that I have developed to extend TVM without altering its core code [LeiWang1999/TVM.CMakeExtend](https://github.com/LeiWang1999/TVM.CMakeExtend), and some important considerations that we should aware of.

## Solution Overview

This project demonstrates how to:

- **Keep TVM as an Independent Module**: Treat TVM as an external dependency, either as a submodule or by linking to a prebuilt version.
- **Use CMake for Modular Builds**: Utilize CMake to build your custom code separately, linking against the TVM libraries without integrating your code into TVM's source tree.
- **Avoid Code Duplication and Conflicts**: By not modifying TVM directly, you avoid merge conflicts and can benefit from the latest updates in TVM without additional overhead.
- **Facilitate Collaboration**: Other developers can contribute to your project without needing to navigate a custom version of TVM.

## Repository Structure

```
TVM.CMakeExtend/
├── 3rdparty/
│   └── tvm/                 # Submodule pointing to TVM
├── build/                   # Build directory
├── include/                 # Custom header files
├── src/                     # Custom source files (passes, codegens, etc.)
├── python/
│   └── your_project/        # Python bindings and extensions
├── CMakeLists.txt           # Main CMake configuration
└── README.md                # This README file
```

### CMake Modular Build

The key to this approach is the CMake configuration that allows you to build your project separately while linking against TVM.

**Using Prebuilt TVM Libraries**

```cmake
if (DEFINED TVM_PREBUILD_PATH)
    message(STATUS "Using prebuilt TVM from ${TVM_PREBUILD_PATH}")
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
    message(STATUS "Building TVM from source")
    add_subdirectory(${TVM_SOURCE_DIR} tvm EXCLUDE_FROM_ALL)
endif()
```

This configuration checks if `TVM_PREBUILD_PATH` is defined:

- If it is, it treats TVM as a prebuilt library and links against it.
- If not, it adds TVM as a subdirectory to build it from source.

**Building Your Custom Extensions**

```cmake
file(GLOB_RECURSE CUSTOM_SRCS src/*.cc)
add_library(custom_objs OBJECT ${CUSTOM_SRCS})
set(CUSTOM_INCLUDES
    ${TVM_SOURCE_DIR}/include
    ${TVM_SOURCE_DIR}/src
    ${TVM_SOURCE_DIR}/3rdparty/dlpack/include
    ${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include
)
target_include_directories(custom_objs PRIVATE ${CUSTOM_INCLUDES})
```

This sets up your custom source files and includes TVM's headers for compilation.

You have two options:

1. **Use a Prebuilt TVM**

   If you already have TVM installed or built elsewhere (e.g., via `pip install apache-tvm`), you can link against it.

   ```bash
   mkdir build && cd build
   cmake .. -DTVM_PREBUILD_PATH=/path/to/tvm/build
   make -j$(nproc)
   ```

   Replace `/path/to/tvm/build` with the actual path to your TVM build directory containing `libtvm.so` and `libtvm_runtime.so`.

2. **Build TVM from Source**

   If you prefer to build TVM from source along with your project:

   ```bash
   mkdir build && cd build
   cp ../3rdparty/tvm/cmake/config.cmake .
   # Edit config.cmake to enable desired features
   echo "set(USE_LLVM ON)" >> config.cmake
   echo "set(USE_CUDA ON)" >> config.cmake
   cmake ..
   make -j$(nproc)
   ```

   This will build both TVM and your custom extensions.

### Handling Python Bindings

To ensure that your custom extensions are properly registered with TVM's Python API:

- **Load Custom Libraries Before Importing TVM**: Load your custom shared libraries before importing `tvm` in Python to ensure that the global functions are registered.

- **Modify `__init__.py`**: In your Python package's `__init__.py`, handle environment variables and library loading:

  ```python
  import os
  import sys

  # Set up environment variables
  os.environ['TVM_LIBRARY_PATH'] = '/path/to/your/libs'

  # Load custom libraries
  from .libinfo import find_lib_path
  _LIBS = find_lib_path()
  for lib in _LIBS:
      tvm.lib.load_library(lib)

  import tvm
  ```

### Custom Library Loader (`libinfo.py`)

Implement a custom library finder that locates your shared libraries at runtime.

```python
import os

def find_lib_path():
    curr_path = os.path.dirname(os.path.abspath(__file__))
    lib_path = []
    for lib in ['libyour_project.so', 'libyour_project.dylib', 'your_project.dll']:
        full_path = os.path.join(curr_path, lib)
        if os.path.exists(full_path):
            lib_path.append(full_path)
    if not lib_path:
        raise RuntimeError("Cannot find your_project library")
    return lib_path
```


Note: The process of loading the library must happen before importing TVM because the registration of global functions occurs during TVM’s import process. At that point, the library must already be loaded; otherwise, any custom global functions you’ve written won’t be registered in TVM’s global function table during the import process.

For example, directly running import tvm:

```python
from tvm._ffi.registry import list_global_func_names
print(list_global_func_names())
'''
'tir.analysis.find_anchor_block', ... , 'relay._transform.ToBasicBlockNormalForm', 'relay._transform.to_cps', 'relay._transform.ToCPS', 'topi.bitwise_or', 'relay._transform.InferTypeLocal', 'relay.backend.aot.CreateExecutorMetadata', 'relay.build_module._AOTExecutorCodegen', 'relay.build_module._BuildModule', 'tvm_callback_cuda_compile', 'relay.backend.CreateExecutor']
'''
```

Loading the library before importing TVM:

```python
import tilelang
from tilelang import tvm as tvm
from tvm._ffi.registry import list_global_func_names
print(list_global_func_names())
'''
tl* is our own pass
'tl.transform.LowerHopperIntrin', ..., 'tir.analysis.find_anchor_block', 'tir.analysis.find_anchor_block', ... 'relay._transform.InferTypeLocal', 'relay.backend.aot.CreateExecutorMetadata', 'relay.build_module._AOTExecutorCodegen', 'relay.build_module._BuildModule', 'tvm_callback_cuda_compile', 'relay.backend.CreateExecutor']
'''
```

## Examples

### Adding a Custom Pass

**C++ Implementation (`src/my_pass.cc`):**

```cpp
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {
namespace transform {

tvm::transform::Pass MyCustomPass() {
    auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
        // Implement your pass logic here
        return f;
    };
    return tvm::transform::CreatePrimFuncPass(pass_func, 0, "MyCustomPass", {});
}

TVM_REGISTER_GLOBAL("my_project.transform.MyCustomPass")
.set_body_typed(MyCustomPass);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
```

**Python Usage:**

```python
import tvm
import your_project.transform

mod = ...  # your IRModule
mod = your_project.transform.MyCustomPass()(mod)
```


I think the final blueprint of this approach: when installing a project based on TVM, there’s no need to compile a separate version of TVM. You can simply install the upstream version of TVM via `pip install apache-tvm` and then install your custom extension module with `pip install xxx`. Even when building from source, there’s no need to compile the entire TVM codebase. You can just download the latest TVM release using `pip install apache-tvm`, and your extension module’s libraries will automatically link to the installed TVM package. If everyone adopts this approach, the overall experience of working with TVM should improve significantly.

