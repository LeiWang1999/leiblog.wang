---
title: 正方教务爬虫JavaScript版
categories:
  - Technical
tags:
  - Javascript
  - Reptile
date: 2020-03-23 01:14:38	
---

人到大三、也不想考研。在留在大学有限的时间里，想做一些有意义的事情。第一个想法是开发一个应用，可以查课表、成绩、空教室等等，毕竟在我的学校大多数人使用的超级课程表，面对那么多用户，那么多高校的教务系统,所以难免会有些高延迟。如果我们缩小用户范围，仅仅面对我们学校，那速度肯定会快很多的，但是询问过学校的信息中心，发现高校用的教务系统是外包给外面的公司管理的，并没有 API 接口。于是首要任务变成了**提供 API 接口**。

<!-- more -->

很显然，教务官网没有提供 API 接口给我们，于是只能自己模拟登陆写爬虫了。在搜索引擎上进行搜索，发现有类似想法的人还是很多的，但是我逛了一大圈、居然只有 Java、Python 两个版本，没有别的版本了，但好在思路很清晰。

为了后期开发方便，提供接口，我用 JavaScript 重构一下。我相信开发网站的全干工程师们，比起 Java 和 Python，后端用 Nodejs 会更加顺手一点吧，也在此文中记录一些踩过的坑。

> 虽然网络上已经对爬取的过程有了原理性的介绍了，但为了避免读者还要反复阅读别的文章的麻烦、这里还是要详细的介绍一下流程。

#### 本次开发使用的环境

###### 操作系统：MacOS

###### 开发框架：Egg.js（阿里为了规范提出的框架， 熟悉 nodejs 的 es6 语法和 koa2 框架的同学一个多小时就可以上手）

###### 浏览器和抓包工具：Chrome/Safari、Charles

###### 开源仓库地址：[戳我前往](https://github.com/NjtechPrinceling/SchoolApi)

### 模拟登陆

#### 第一步、分析登陆表单

先贴出本次实验的正方教务系统的界面，省的不同的看官白费力气，但就算界面不同，思路肯定也是相同的，看看也无妨。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200323005925321.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
按 F12 打开开发者工具，调到**Network**这栏（不知道为啥写文章的时候我的 chrome 有点抽风，所以我打开了 charles，该工具类似于 Windows 上的 Fiddler），我们首先模拟一次失败的登陆，我这里键入虚假的用户名和密码:1543140220/123456。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200323005954986.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
前端给后端发了两个请求，我们来逐一分析。

##### 第一个请求

其实从请求的名字就能看得出来，是获取加密的公钥的，可以大胆的猜测这是一个 RSA 加密了，等会儿我们分析 js 代码的时候会证明这一点，所以我们要记住 modulus 和 exponent 这两个参数。

##### 第二个请求

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200323010016424.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
第二个请求携带的参数就多了
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200323010038706.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

###### csrftoken:这个我们一般是用来防止网站遭受 xss 跨站请求脚本攻击的时候弄的玩意

如何获取这个令牌呢？
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200323010053737.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

搜索一下网页的 html 可以发现，这玩意藏在表单的一个隐藏标签里。在我的项目中，使用了**cheerio**这个库来解析 HTML。相关代码如下：

```javascript
 async get_csrf_token(session, time) {
    const ctx = this.ctx;
    let headers = {
      'Host': 'jwgl.njtech.edu.cn',
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:58.0) Gecko/20100101 Firefox/58.0',
      'Accept-Encoding': 'gzip, deflate',
      'Accept': '*/*',
      'Connection': 'keep-alive',
      'Cookie': session
    }
    const options = {
      headers,
    }
    const url = await this.service.common.get_login_url();
    const res = await ctx.curl(url + time, options);
    const resultHtml = (res.data.toString());
    const cheerioModel = cheerio.load(resultHtml);
    const csrf_token = cheerioModel('#csrftoken')[0].attribs.value;

    return { token: csrf_token, session }
  }
```

###### yhm：这个是用户名

###### mm：密码，明显是加密过的。But 为什么要写两次？我觉得很迷惑，反正请求的时候发一个过去就行了。

#### 第二步、获取加密密码

##### 分析网页端的加密算法

还是分析 Web 前端的 js 脚本，看看他是怎么加密的。在开发者工具里搜索何时给“mm”赋值、找到了这一段，果然是 rsa 加密。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200323010109270.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
但 nodejs 的 rsa 加密真是坑了我一天，我本来使用的是**node-rsa**这一个库、似乎除了这个库 nodejs 没有别的用来 rsa 加密的库了。而网页上的 rsa 加密算法，貌似是自己撸出来的、代码奇丑无比。但是明明生成的 RSA 对象的 key 跟相同参数生成的网页端 debug 生成的 key 是一样的，验证就是不通过。

最后很无奈，只好把网页上的 rsa 加密用的 js 代码拷贝下来，封装成 utils 的一个接口了，事实证明这是一个好办法。

以上是我叨逼叨、只是为了介绍了我项目中 utils 下 rsa.js 文件的由来。有了密码的明文、RSA 算法、公钥对、我们就能够正确求出加密后的算法了。

#### 第三步、判断是否登录成功

总之，我们有了密码、csrf_token、外加模仿他正常请求的 header，就能够正确登陆了。但我们如何判断是否登录成功？

当登陆失败的时候，服务器会给我们返回 HTML、有点 low。但返回的 html 中包含字段"用户名或密码不正确"。只需要判断返回的文本中中是否包含这句话，就可以判断是否登录成功了。总之，登录的流程代码如下：

```javascript
  async login(username, password, time) {
    let { modulus, exponent, session } = await this.service.common.get_public_key(time);
    let { token } = await this.service.common.get_csrf_token(session, time);
    let enpassword = await this.service.common.process_public(password, modulus, exponent);
    let data = {
      'csrftoken': token,
      'mm': enpassword,
      'yhm': username
    };
    const url = await this.service.common.get_login_url();
    let headers = {
      'Host': 'jwgl.njtech.edu.cn',
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:58.0) Gecko/20100101 Firefox/58.0',
      'Accept': 'text/html, */*; q=0.01',
      'Accept-Encoding': 'gzip, deflate',
      'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
      'Referer': url + time,
      'Upgrade-Insecure-Requests': '1',
      'Cookie': session,
      'Connection': 'keep-alive',
      "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
    }

    const options = {
      method: 'POST',
      headers,
      data,
    }
    const ctx = this.ctx;
    const result = await ctx.curl(url, options);
    const regValue = '用户名或密码不正确'
    if (result.data.toString().indexOf(regValue) > 0) {
      return {
        success: false,
        message: regValue
      }
    }
    else {
      return {
        success: true,
        message: '登陆成功',
        session: result.headers['set-cookie']
      }
    }
  }
```

当登陆成功之后，服务器给我们返回的响应里包含"Set-Cookie"字段。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200323014340476.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
拿到这个**JSESSIONID**把它放到以后请求的 Cookie 里，就可以为所欲为啦。

到这里，模拟登陆的思路介绍完了。

### 演示：获取成绩

在上一节，模拟登陆过后。我们拿到了 JSessionID，如果你还是比较迷茫，那我再掩饰一下如何获取成绩？

#### 第一步、模拟获取成绩

在这里插入图片描述
现在网页端模拟一下获得成绩、再看看抓到的包。

首先看 Cookie、是 JSESSIONID 字段、第二个字段测试了一下不加也没事。
![在这里插入图片描述](https://img-blog.csdnimg.cn/202003230138146.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
在看发送的表单内容

![[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-3gYhYF1U-1584896295432)(/Users/wanglei/Library/Application Support/typora-user-images/image-20200323004010062.png)]](https://img-blog.csdnimg.cn/20200323010314744.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

第一栏是学年、第二栏是学期。这里的学期比较坑、好像是加密过的，大家只要记住第一学期是 3、第二学期是 12 就好了、这个可以自己试出来的，nd 是时间戳。其他照抄就好。

然后看他返回的数据：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200323010326178.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)

**PS**：丑的一批

依稀能分辨出来，这个是根据拼音来命名的。

所以获取成绩的 API 的代码是这样的：

```JavaScript
    async grade() {
        const { ctx } = this;
        const { username, password, year, term } = ctx.request.body;
        const time = await this.service.common.get_time();
        const loginInfo = await this.service.login.login(username, password, time);
        if (loginInfo.success) {
            const gradeInfo = await this.service.grade.post_grade_data(year, term, loginInfo.session)
            ctx.body = gradeInfo
        } else {
            ctx.body = {
                success: false,
                message: loginInfo.message
            }
        }
    }
```

```javascript
    async post_grade_data(year, term, session) {
        // 校验
        if (!parseInt(year) || parseInt(year) > (new Date().getFullYear())) {
            return {
                success: false,
                message: "请求课程年份出错"
            }
        }

        //   默认第一学期
        let form_term = '3';
        if (parseInt(term) === 1) {
            form_term = '3'
        } else if (parseInt(term) === 2) {
            form_term = '12'
        }
        const url = await this.service.common.get_grade_url();
        const data = {
            '_search': 'false',
            'nd': this.service.common.get_time(),
            'queryModel.currentPage': '1',
            'queryModel.showCount': '15',
            'queryModel.sortName': '',
            'queryModel.sortOrder': 'asc',
            'time': '0',
            'xnm': year,
            'xqm': form_term
        }
        let headers = {
            'Host': 'jwgl.njtech.edu.cn',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:58.0) Gecko/20100101 Firefox/58.0',
            'Accept': 'text/html, */*; q=0.01',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': url,
            'Upgrade-Insecure-Requests': '1',
            'Cookie': session,
            'Connection': 'keep-alive',
            "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
        }
        const options = {
            method: 'POST',
            headers,
            data,
        }
        const ctx = this.ctx;
        const result = await ctx.curl(url, options);

        const response_data = JSON.parse(result.data.toString());
        const courseitems = response_data.items;
        const grade = courseitems.map(currentValue => {
            return {
                name: currentValue.kcmc,
                credit: currentValue.xf,
                grade: currentValue.bfzcj,
                point: currentValue.jd,
                teacher: currentValue.jsxm
            }
        })

        return {
            success: true,
            message: "请求课程成绩成功",
            grade: grade
        };
    }
```

### 结果演示

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200323010341532.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NDk4NzAx,size_16,color_FFFFFF,t_70)
如果你觉得这个过程写的不够好，也可以参考 [这位博主的 Python 版本](https://blog.csdn.net/Koevas/article/details/88384604)

另外，还可以去我的个人博客阅读，虽然没有开启评论功能哈哈

博客地址：[点击前往](http://www.leiblog.wang/home)

另外再次贴上 Github 仓库链接：[戳我前往](https://github.com/NjtechPrinceling/SchoolApi)
