---
categories:
  - Technical
tags:
  - Upic
  - Figure Bed
date: 2020-04-27 00:39:10	
---

这篇文章给大家分享的是个人用自建图床搭建过程。

图床，也就是专门提供存储图片的地方，我们只要通过图床提供的 API 接口，把图片上传上去，就可以通过外链访问了，我们在 CSDN 发表文章，上传图片，其实就是用的 CSDN 的图床，但 CSDN 的图床有时候也挺不方便的。

比如，在我刚开始写博客的时候，我喜欢先在本地写，博客中的图片我都存储在本地的文件夹里。
![](https://imgconvert.csdnimg.cn/aHR0cDovL2xlaWJsb2cud2FuZy9zdGF0aWMvaW1hZ2UvMjAyMC80L2NwdG5IUi5wbmc?x-oss-process=image/format,png)

<!-- more -->

然后在本地写完了，保存为.md 文件。

再打开 CSDN、上传 Markdown，然后就会出现这样的错误：

![](https://imgconvert.csdnimg.cn/aHR0cDovL2xlaWJsb2cud2FuZy9zdGF0aWMvaW1hZ2UvMjAyMC80L21ZSzZ2Ni5wbmc?x-oss-process=image/format,png)

如果一篇文章的图片多起来，事情就麻烦了，有时候还会漏掉一些图片。

如果我们自己搭建图床，直接把这些图片存储在云端，图片的来源是 http 服务，就没有这些问题了，本文就是以该思路来提供一个解决方案。

最终效果如下：

![](https://imgconvert.csdnimg.cn/aHR0cDovL2xlaWJsb2cud2FuZy9zdGF0aWMvaW1hZ2UvMjAyMC80LyVFNSVCMSU4RiVFNSVCOSU5NSVFNSVCRCU5NSVFNSU4OCVCNjIwMjAtMDQtMjYlRTQlQjglOEIlRTUlOEQlODgxMC40MC40OS5naWY)

为了实现这样的效果，你需要：

- 一点 Node.js 的知识
- 一台 VPS 服务器，Linux、Windows 都可以（本文使用的是 Win Server、坑比 Linux 要多得多）
- 本文仅用作 Mac 用户作参考，因为用到了 upic（开源图床客户端）

### 下载并且安装 uPic

uPic(upload Picture) 是一款 Mac 端的图床(文件)上传客户端，可以将图片、各种文件上传到配置好的指定提供商的对象存储中。
然后快速获取可供互联网访问的文件 URL

不要去 APP Store 下载。

点击这个[链接](https://github.com/gee1k/uPic/releases)，前往 Github 下载 dmg 文件。

或者使用 HomeBrew 安装

```bash
brew cask install upic
```

### 在 VPS 服务器上安装 minio

minio 是可以部署在本地的对象存储服务。

根据 VPS 的系统不同，在[官方文档](https://docs.min.io/cn/minio-quickstart-guide.html)里都能够找到对应的安装方法。

Mac 和 Linux 安装都很方便，Windows 下载下来就是一个 exe，都不带安装过程的。

在 CMD 里启动服务：

```powershell
./minio.exe server ./data
```

其中./data 是指定的对象存储的文件所在位置。

打开之后默认会侦听 9000 端口，访问该端口可以进入服务。

![](https://imgconvert.csdnimg.cn/aHR0cDovL2xlaWJsb2cud2FuZy9zdGF0aWMvaW1hZ2UvMjAyMC80L2V6M2o4ci5wbmc?x-oss-process=image/format,png)

然后，Win 比较头疼的是如何将这句话变成一个服务，毕竟你总不能一直挂个命令行在上面吧。

先创建一个 minio.bat 文件，然后写入如下内容

```bat
@echo off
set MINIO_ACCESS_KEY=access_key
set MINIO_SECRET_KEY=secret_key
C:\Users\Administrator\Desktop\个人网站\minio.exe server C:\Users\Administrator\Desktop\个人网站\文件共享
```

将第四行的两个绝对路径分别替换成 minio 的路径和对象存储的路径即可。

然后可以运行一下./minio.bat 看看是否能够成功运行。

要注意**MINIO_ACCESS_KEY**和**MINIO_SECRET_KEY**这两个参数，相当于 minio 的用户名和密码。

然后我们将这个脚本注册成服务，使用 cmd 的 sc 命令，格式如下：

```powershell
sc create minioServer binpath=C:\Users\Administrator\Desktop\个人网站\minio.bat start=auto
```

同样，把脚本的绝对路径替换成自己的。

这样我们就注册完服务，在服务管理页面能够看到该服务了，可以设置为开机自启动。

![](https://imgconvert.csdnimg.cn/aHR0cDovL2xlaWJsb2cud2FuZy9zdGF0aWMvaW1hZ2UvMjAyMC80L0hPaUZxMS5wbmc?x-oss-process=image/format,png)

### 编写 Nodejs 后端

代码很简单，使用的 Expressjs，抄的[学长的代码](http://online.njtech.edu.cn/blog/2020/04/05/upic-figure-bed/)。

```javascript
const express = require("express");
const multer = require("multer");
const Minio = require("minio");
const path = require("path");
const fs = require("fs");

const upload = multer({ dest: "./tmp/" });
upload.limits = {
  fileSize: 10 * 1024 * 1024, // 最大 10MB
};

const app = express();

// minio 客户端
const minioClient = new Minio.Client({
  endPoint: "xxxxxx", // 替换 minio 的访问地址
  port: 9000,
  useSSL: false,
  accessKey: "xxxxx", // 替换 accessKey
  secretKey: "xxxxxx", // 替换 secretKey
});

const handleError = (err, res) => {
  res.status(500).contentType("text/plain").end("Oops! Something went wrong!");
};

app.post("/", upload.single("file"), function (req, res, next) {
  // 检查 token
  const token = req.get("token");
  if (token !== "xxxxxxx") {
    // 替换鉴权 token
    return handleError("", res);
  }
  // 文件临时存储文件夹
  const { originalname, path: tmpPath } = req.file;
  // 文件后缀
  const extname = path.extname(originalname).toLowerCase();

  if ([".png", ".jpg", ".jpeg", ".gif"].indexOf(extname) !== -1) {
    // minio 中保存的文件名
    const filePath = `${tmpPath}.${extname}`;
    // 获取月份
    const today = new Date();
    const month = today.getMonth() + 1;
    const year = today.getFullYear();
    // 重命名文件
    fs.rename(tmpPath, filePath, (err) => {
      if (err) return handleError(err, res);
      // 上传指 minio 中
      minioClient.fPutObject(
        "resource",
        `img/${year}/${month}/${originalname}`,
        filePath,
        {
          "Content-Type": `image/${extname.split(".")[1]}`,
        },
        async () => {
          await fs.unlink(filePath, function () {});
          res.json({
            data: `http://ip/resource/img/${year}/${month}/${originalname}`, // 替换 ip
          });
        }
      );
    });
  } else {
    // 删除文件
    fs.unlink(tmpPath, (err) => {
      if (err) return handleError(err, res);
    });
    res.status(403).contentType("text/plain").end("Only img files are allowed!");
  }
});

app.listen(3002, () => {
  console.log("app listening on port 3002");
});
```

然后用 pm2 部署一下就 ok 了。

我们侦听的是 3002 端口，实际上有域名的话可以做个转发，这里我就不多讲了。

### uPic 客户端配置

![image-20200427002819589](https://imgconvert.csdnimg.cn/aHR0cDovL2xlaWJsb2cud2FuZy9zdGF0aWMvaW1hZ2UvMjAyMC80L0wxRVVOeC5wbmc?x-oss-process=image/format,png)

点击其他字段新增一个`Header` ：

![](https://imgconvert.csdnimg.cn/aHR0cDovL2xlaWJsb2cud2FuZy9zdGF0aWMvaW1hZ2UvMjAyMC80L0p0VFlRNC5wbmc?x-oss-process=image/format,png)

**注意**：这里的 token 字段是在 nodejs 代码中设置的。

然后验证一下：

![](https://imgconvert.csdnimg.cn/aHR0cDovL2xlaWJsb2cud2FuZy9zdGF0aWMvaW1hZ2UvMjAyMC80L2NSUVFNZS5wbmc?x-oss-process=image/format,png)

并且能够支持截图、剪贴板、文件上传，很完美！
