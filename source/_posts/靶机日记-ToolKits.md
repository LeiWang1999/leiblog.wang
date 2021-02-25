---
title: 靶机日记 | ToolKits
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2021-02-25 14:14:24
---

### 1. Crack the Passwd of Zip Package

```bash
zip2john backup.zip > passwd.txt
john passwd.txt
```

### 2. Find

1. Find file with suid

```bash
find / -perm -u=s 2>/dev/null
```

If we got it ，we can go to [GTFobins](https://gtfobins.github.io) to check out whether it is possible for Privilege Escalation

2. Find file for special group

``` bash
find / -type f -group bugtracker 2>/dev/null
```

3. Find file

```bash
find / -name "user.txt"
```

<!-- more -->

### 3. smb

If the target machine with port 445 opened, we can check out the file under directory if no auth：

```bash
smbclient -N -L ////ip
```

and then dump file from directory:

```bash
smbclient  \\\\10.10.10.27\\backup/
get filename
```

### 4. Upgrade Shell to tty

```bash
SHELL=/bin/bash script -q /dev/null
python3 -c "import pty;pty.spawn('/bin/bash')"
```

### 5. Sudo

view the special permissions of current role:

```bash
sudo -l
```

When a command can be executed with sudo , we can use this command to open a new bash shell to get root privilege .

For example,  `! /bim/bash` in Vim.