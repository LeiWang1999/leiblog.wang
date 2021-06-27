---
title: NVDLA Compiler Analysis
categories:
  - Technical
tags:
  - NVDLA
date: 2021-06-24 16:42:15
---

这篇文章记录一下笔者剖析 NVDLA Compiler 工作机制的一些经过，在 NVDLA 的软件仓库中，Compiler与Runtime的源代码被放置在umd目录下，由同一个 Makefile 组织。

对于sw部分文档十分缺乏，在 NVDLA的页面里只有零星的介绍，而关于软件设计的细节、以及如何拓展设计都没有介绍。并且，Compiler这部分的代码充满了智慧，例如：

![image-20210624203707954](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210624203707954.png)

这是实习生写的吧？？？ 还有一些突然停止维护了而没开发完的feature。

但阅读其代码，理解其设计与工作的思路，学习其抽象语法树的设计还是一个比较有意义的工作。

<!-- more -->

### 1. Clion 结合 Docker 调试 Makefile Project

由于 Compiler 需要一些 Linux 的头文件，所以在 Mac 和 Windows 上不方便直接编译（笔者是 MacOS），为了不污染环境，我推荐使用 docker 新建一个容器，当然为了方便验证，我还是推荐直接使用之前文章里提到过的 image ：

```bash
docker pull farhanaslam/nvdla 
# 做一个端口映射，毕竟22端口还是很容易冲突的
docker run -it -v /Users/wanglei/:/home -p 2222:22 --name "nvdlakit" wanglei/nvdla-kit
```

然后为其添加sshd服务，具体可以参考这篇[博客](https://www.cnblogs.com/mengw/p/11413461.html)：

```bash
apt-get upgrade
apt-get install openssh-client
apt-get install openssh-server
/etc/init.d/ssh start
```

开启 root 的访问权限：

```bash
vim /etc/ssh/sshd_config
```

找到 `PermitRootLogin`将其设置为 yes 。

再用`passwd `命令设置一下Root的密码就好了。

大概一年之前，Clion 并不支持 Makefile 项目，在这之后其在插件里提供了部分对 Makefile 的支持，但是不能够进行远程调试，而在笔者尝试使用 Clion 远程调试 Makefile 项目的时候，JB的官方刚好发布了这个问题的解决方案，时机非常巧妙，官方的post title叫做 [Full remote mode](https://www.jetbrains.com/help/clion/remote-projects-support.html)，根据配置就可以远程调试了，但是对于 Compiler，其在编译的过程中依赖环境变量`TOP`、其在编译之后的执行阶段又依赖环境变量`LD_LIBRARY_PATH`，所以我们需要对配置做一点小改动。

在`Preference|Build,Execution,Deployment｜Makefile`页面，为 options 加上 TOP 的路径。

![image-20210624181927692](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210624181927692.png)

在`Configurations`页面，加上 Environment variables 。

![image-20210624182055447](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20210624182055447.png)

再来，apps 的 Makefile 里没有编译出带调试信息的可执行文件，为`MODULE_COMPILEFLAGS`加上`-g`选项。这样，就可以单步调试了。

### 2. Compiler

首先，umd目录下面一共有若干个文件夹：

- apps：compiler和runtime的外层逻辑，一般这里放的都是解析命令行参数之类的程序。
- core：runtime和compiler的主要实现逻辑放在这里，是需要着重阅读的部分
- externel：这个目录存放了umd用到的第三方库，需要说明的是protobuf主要用于caffe模型的解析，因为caffe的blob格式原始存储方式是使用了google开发的protobuf、用来压缩生成 Loadable 文件的 FlatBuffers等等。
- make：若干通用的makefile规则。
- port：主要是runtime的底层访问API，为了实现OS无关，做了这个隔离层，目前只实现了Linux下的。这层API包括内存访问，用户态和内核态的交互IOCTL，内存分配等。需要注意的是NVDLA的runtime部分用户态和内核态交互使用了Linux用于显卡抽线的DRM接口
- utils：这个文件夹放了几个通用的模块，包括BitBinaryTree以及伙伴内存管理等。其中伙伴内存管理模块在compiler的tensor内存分配推导部分有用到

#### 2.1 NVDLA Compiler 工作流程

下图是我整理的，Compiler工作的主要路线图，其中蓝色标注的部分是编译部分的主体，在正式进行编译之前，编译器会把 CaffeModel 转换成 Network 对象；把输入的profileName 转换成 Profile 对象，例如指定 fast-math 那么在profile 对象里就会把一些优化的开关打开，把targetname，就是nv_small/nv_large 这种，会转换成TargetConfig对象，内有硬件相关的配置信息。

之后，通过 Network，生成一个标准的抽象语法树，再转换成与硬件相关的 engine_ast，engine_ast 就是编译器最核心的 IR，接下来的各种优化都是对该 IR 做变换。

![WorkFlow of  NVDLA Compiler](/Users/wanglei/Downloads/WorkFlow of  NVDLA Compiler.png)

在阅读代码的时候，可以打开官方已经写了的部分调试信息输出开关，在源代码里可以找到类似这样的声明：

```c++
inline bool debugGraphDump() const { return false; }
inline bool debugClone() const { return false; }
inline bool debugOps() const { return false; }
inline bool debugGroupOps() const { return false; }
inline bool debugMathOptz() const { return false; }
inline bool debugWeights() const { return false; }
inline bool debugQuantization() const { return false; }
inline bool debugFuseSubEngineOps() const { return false; }
inline bool debugSurfaces() const { return false; }
inline bool debugBuffers() const { return false; }
inline bool debugCopyOutDebug() const { return false; }
inline bool debugMemoryLayout() const { return false; }
inline bool debugBinding() const { return false; }
inline bool debugDepGraph() const { return false; }
inline bool debugMemHazards() const { return false; }
inline bool debugRelocs() const { return false; }
```

将`false`调成`true`，就可以输出一些信息，例如 debugGraphs 开关可以把几个关键的抽象语法树保存为 Json 格式的文件。

#### 2.2 main.c

在 main 函数里，编译器首先做的事情是对命令行的参数进行解析，不得不感叹这部分的逻辑用的居然是if、else语句来解析参数：

```c++
if (ii >= argc)
  break;
const char* arg = argv[ii];

if (std::strcmp(arg, "-h") == 0) // help
{
  // Print usage
  showHelp = true;
  break;
}
```

我觉得，改成类似 Caffe 的，使用 gtest 的思路不会更好吗？

```c++
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
    "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0,
    "Optional; network level.");
```

解析参数的过程中，会初始化TestAppArgs这个结构体，而在程序的开始，已经给其赋了初始值：

```c++
static TestAppArgs defaultTestAppArgs =
{
    /* .project = */ "OpenDLA",
    /* .inputPath = */ "./",
    /* .inputName = */ "",
    /* .outputPath = */ "./",
    /* .testname = */ "",
    /* .testArgs = */ "",
    /* .prototxt = */ "",
    /* .caffemodel = */ "",
    /* .cachemodel = */ "",
    /* .profileName = */ "fast-math",
    /* .profileFile = */ "",
    /* .configtarget = */ TARGET_CONFIG_NAME,
    /* .calibtable = */ "",
    /* .quantizationMode = */ DEFAULT_QUANT_MODE,
    /* .numBatches = */ DEFAULT_BATCH_SIZE,
    /* .inDataFormat = */ DEFAULT_DATA_FMT,
    /* .computePrecision = */ nvdla::DataType::INT8
};
```

初始化完成之后，再调用同样写在该文件的 LaunchTest 函数:

```C++
NvDlaError launchTest(const TestAppArgs* appArgs)
{
    NvDlaError e = NvDlaSuccess;
    TestInfo testInfo;

    PROPAGATE_ERROR_FAIL(testSetup(appArgs, &testInfo));

    PROPAGATE_ERROR_FAIL(parseAndCompile(appArgs, &testInfo));

    return NvDlaSuccess;

fail:
    return e;
}
```

在 testSetup 函数里，主要是检验一下对应的文件是否真的存在等。

其中又有一个非常重要的结构体：

```C++
struct TestInfo
{
    // runtime
    nvdla::IRuntime* runtime;
    std::string inputLoadablePath;
    NvU8 *inputHandle;
    NvU8 *outputHandle;
    NvU8 *pData;
    bool dlaServerRunning;
    NvS32 dlaRemoteSock;
    NvS32 dlaServerSock;
    NvU32 numInputs;
    NvU32 numOutputs;
    NvDlaImage* inputImage;
    NvDlaImage* outputImage;
};
```

在该函数里，还有一个让人窒息的操作，给大家康康：

```c++
    // Clear wisdomPath if any exist
    removeCmd += "rm -rf " + wisdomPath;
    ii = std::system(removeCmd.c_str()); // This is pretty awful
    if (ii != 0)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "system command failed: \"%s\"", removeCmd.c_str());
```

#### 2.3 工厂设计模式

在 NVDLA 的软件栈代码里，用了很多工厂模式的设计，例如创建一个 Network 对象，需要调用 NetworkFactory 的 newNetwork 方法，其他的，像是 Tensor、Node、Layer 都有对应的工厂，来生产不同的 Tensor、 Node、Layer。

此外，针对 Network、Tensor、Node、Layer 等等对象，其都定义了对应的虚接口类，如 INetwork、ITensor、INode、ILayer、这些接口类里通过纯虚函数定义了继承的子类必须实现的函数。

比如举个 Network 的例子：

```C++
class Network : public INetwork
{
public: // externally facing

    virtual ITensor* addInput(const char* name, Dims4 dimensions);

    //	virtual void markChanged(const ILayer*);
    virtual bool markInput(ITensor * tensor);
    virtual void markOutput(ITensor* tensor);

    virtual IConvolutionLayer *    addConvolution(ITensor* input, int numOutputs, int paddingValue,
                                                  Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation,
                                                  Weights kernelWeights, Weights biasWeights, BiasMode biasmode, int numGroups);
    virtual IFullyConnectedLayer * addFullyConnected(ITensor* input, int outputSize, Weights kernelWeights, Weights biasWeights, BiasMode biasMode);
    virtual IActivationLayer *     addActivation(ITensor* input, ActivationType type);
    virtual IPoolingLayer *        addPooling(ITensor* input, PoolingType type,
                                              Dims2 windowSize, Dims2 stride, Dims2 tlPadding, Dims2 brPadding);
    virtual ILRNLayer *            addLRN(ITensor* input, int window, float alpha, float beta, float k);
    virtual IScaleLayer *          addScale(ITensor* input, ScaleMode mode, Weights shift, Weights scale, Weights power);
    virtual IBatchNormLayer *      addBatchNorm(ITensor* input, BatchNormMode mode, Weights mean, Weights variance, float epsilon);
    virtual ISoftMaxLayer *        addSoftMax(ITensor* input);
    virtual IConcatenationLayer *  addConcatenation(ITensor * const * inputs, int numInputs);
    virtual ISliceLayer *          addSlice(ITensor* input, int numOutputs);
    virtual IDeconvolutionLayer *  addDeconvolution(ITensor* input, int numOutputs, int paddingValue,
                                                    Dims2 kernelSize, Dims2 tlPadding, Dims2 brPadding, Dims2 stride, Dims2 dilation,
                                                    Weights kernelWeights, Weights biasWeights, BiasMode biasMode, int numGroups);
    virtual IElementWiseLayer *    addElementWise(ITensor* input0, ITensor* input1, ElementWiseOperation op);

    virtual int  getNumInputs() const;
    virtual int  getNumOutputs() const;
    virtual int  getNumLayers() const ;

    virtual ILayer  * getLayer(int index)  const;
    virtual ITensor * getOutput(int index) const;
    virtual ITensor * getInput(int index)  const;

    virtual void setPoolingOutputDimensionsFormula      (OutputDimensionsFormula* callback);
    virtual void setConvolutionOutputDimensionsFormula  (OutputDimensionsFormula* callback);
    virtual void setDeconvolutionOutputDimensionsFormula(OutputDimensionsFormula* callback);

    virtual OutputDimensionsFormula& getPoolingOutputDimensionsFormula()       const;
    virtual OutputDimensionsFormula& getConvolutionOutputDimensionsFormula()   const;
    virtual OutputDimensionsFormula& getDeconvolutionOutputDimensionsFormula() const;

    virtual const std::vector<ITensor *>& getInputs()  const;
    virtual const std::vector<ILayer * >& getLayers()  const;
    virtual const std::vector<ITensor *>& getOutputs() const;

    virtual NvU16 getFactoryType() const;


public: // internally facing
    Network();
    virtual ~Network();
    virtual bool serializeTo(WisdomContainerEntry *) const;
    virtual bool deserializeFrom(WisdomContainerEntry *);
    virtual bool assignSymbols(Wisdom *);

protected:
    friend class Wisdom;
    friend class NetworkFactory;

    void destroy();

private:

    std::string newLayerName() const;
    std::string newTensorName() const;

    ITensor* addTensor(const std::string & s);
    const ILayer* findLayer(const std::string& name) const;
    bool checkNames(const char* name);

    std::vector<ITensor *> mTensors;
    std::vector<ILayer *>  mLayers;
    std::vector<ITensor *> mInputs;
    std::vector<ITensor *> mOutputs;

    // provides layer dimension caching. Layers can be mutated in any order and dimensions queried at any point.
    // So mutating a layer trims this, and querying always refills the cache up to the queried layer
    //	mutable std::vector<Dims3> mDimensions;

    // internal flags used by the builder that are not accessible through the API
    // int mInternalBuildFlags{ InternalBuildFlags::kENABLE_GRAPH_OPTIMIZATIONS };
    OutputDimensionsFormula* mConvDims, *mDeconvDims, *mPoolDims;


};
```

