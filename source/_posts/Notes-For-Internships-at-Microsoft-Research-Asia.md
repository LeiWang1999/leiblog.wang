---
title: Notes For Internships at Microsoft Research Asia
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2024-05-04 14:19:21
---

Happy to share some notes for internships at Microsoft Research Asia. Which includes the basic understanding of the fancy Infrastructures (e.g. GCR, Blob Storage, etc.).

<!-- more -->

## 1. Notes for Working

### 1.1. Connect the GCR Dev Node

There already exists a great github tutorial from another intern to help us connect the GCR Dev Node through vscode. The tutorial is [here](https://github.com/Timmhxw/GCR_Bastion_guide).

Beyound that, if you want to connect the GCR Dev Node through MacOS, you replace the scripts `gdl.ps1` with the `gdl.sh` from [GCR-Bastion-Auto-Connect-script-with-Bash](https://dev.azure.com/msresearch/GCR/_wiki/wikis/GCR.wiki/6651/GCR-Bastion-Auto-Connect-script-with-Bash).

And you don't need to create another connect_gdl.bat script to forward the port, with `/gdl_1180.sh -n 1180` you can directly connect to the GCR Dev Node, with `./gdl_1180.sh -t -n 1180` you can connect to the GCR Dev Node with the tunnel.

Moreover, if you want to connect several different GCR Dev Nodes, you should dispatch different ports for them. For example, you can modify the contents of `gdl.sh` as follows:

```bash
BASTIONPARAMS="--resource-port 22 --port 2222"
# to
BASTIONPARAMS="--resource-port 22 --port 21180"
```

then you can connect to the GCR Dev Node with `./gdl_1180.sh -n 1180` and connect to the GCR Dev Node with the tunnel with `./gdl_1180.sh -t -n 1180`.

### 1.2. Disable docker reload

```bash
vim /mnt/DATALOSS_WARNING_README.txt
```

```text
WARNING: THIS IS A TEMPORARY DISK.

Any data stored on this drive is SUBJECT TO LOSS and THERE IS NO WAY TO
RECOVER IT.

Please do not use this disk for storing any personal or application data.

For additional details to please refer to the MSDN documentation at:
http://msdn.microsoft.com/en-us/library/windowsazure/jj672979.aspx

To remove this warning run:
    sudo chattr -i /mnt/DATALOSS_WARNING_README.txt
    sudo rm /mnt/DATALOSS_WARNING_README.txt

This warning is written each boot; to disable it:
    echo "manual" | sudo tee /etc/init/ephemeral-disk-warning.override
    sudo systemctl disable ephemeral-disk-warning.service
```


so we can disable the warning with the following commands:

```bash
echo "manual" | sudo tee /etc/init/ephemeral-disk-warning.override
sudo systemctl disable ephemeral-disk-warning.service
```

Moreover, the root directory of the docker service is under the `/mnt` directory, the storage of this temp storage is very limited, we can change the root directory of the docker service to the `/home` directory, which has more space.

```bash 
vim /etc/docker/daemon.json
# add the following contents
{
  "data-root": "/home/docker-data"
}

systemctl restart docker
```

Connect the docker from ssh is also a good idea, you can use the following command to connect the docker service from ssh.

### 1.3. Connect the Blob Storage

Because the GCR Dev Node can only be hold for at most six months, we need to backup our data to some space when we apply for another node, the space is the Blob Storage. But connect a blob is hard. 

First of all, you should have a blob account, usually the intern doesn't have the permission to create a blob account, so you should ask your mentor to share his blob account with you. 

```bash

wget https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb

dpkg -i packages-microsoft-prod.deb

apt-get update

apt-get install -y blobfuse2 libfuse2

echo "accountName $ACCOUNT_NAME" >> ~/fuse_connection.cfg
echo "containerName $sasToken" >> ~/fuse_connection.cfg
echo "sasToken $containerName" >> ~/fuse_connection.cfg

mkdir /home/alias/blob
blobfuse2 mountv1 /home/alias/blob --tmp-path=/home/alias/blobfusetmp --config-file=/home/alias/fuse_connection.cfg --use-https=true --file-cache-timeout-in-seconds=120

fusermount -u /home/alias/blob
```
