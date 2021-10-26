# Linux CRIU功能学习

CRIU(Checkpoint Restore In Username)背景忘记了，大概是一个很牛逼的Linux开发者写的一个功能?程序?，作用是手动将正在运行的进程冻结下来，然后在需要的时候手动恢复进程。
大概原理是将运行中的进程在CPU、内存中的运行中的内容全部以文件的形式保存在硬盘，然后在恢复的时候，通过这些保存的文件恢复。这个很符合Linux上万物皆文件的理念。也可以将这些文件放在另一台支持CRIU的电脑上恢复进程，相当于将进程迁移了。其实说起来很简单，其实复杂的很，反正需要操作很多东西。以下是简单demo：

1. CRIU需要Linux的内核依赖，具体哪个版本我不记得了，但是一般比较新的系统应该都支持。

2. 首先安装CRIU。
ubuntu的话，直接sudo apt-get install criu就可以了。
centos应该可以sudo yum install criu。如果不行的话，可以yum search criu找一下，我没试过centos的。

3. criu官方的demo是用shell脚本来的，这里我们用python来试一下。新建run.py文件。
```python
import time
import datetime

while True:
	time.sleep(3)
	with open("log.txt","a") as f:
		f.write(str(datetime.datetime.now()),"\n")

```

 代码很简单，一个简单的server，每隔3秒向log.txt中输入当前时间。
4. 现在启动这个server。
```shell
nohup python run.py &
```

5. 待代码跑起一会，然后查看日志。应该可以看到输出了很多时间，每个时间三秒间隔。
```shell
cat log.txt
```

6. 现在通过ps命令找到我们运行的进程。
```shell
ps -aef | grep run.py
```

7. 通过上面找到的进程id，冻结进程。$PID是找到的进程ID
```shell
criu dump -t $PID --shell-job
```

8. 执行完后，ls一下，会发现出现了很多新的文件，这些文件应该就是保存的进程运行在CPU，内存中的信息。

9. 现在通过CRIU恢复进程。
```shell
nohup criu restore --shell-job &
```

10. 等运行一段时间，我们再看日志。应该可以看到日志中间断开了一段时间，然后又重新每隔3秒记录当前时间。
```shell
cat log.txt
```

11. 最后找到这个进程然后kill掉，不然一直跑
```shell
ps -aef | grep run.py
kill $PID
```

以上就是CRIU最简单的应用了，也可以将A主机的进程冻结掉，然后迁移到B主机上运行。Google搜索CRIU可以找到criu的官方教程。
我个人的感觉，普通应用CRIU用不到什么地方去。可能我接触的只有web项目，你把web项目都暂停了，何必恢复呢，直接重启也可以，迁移到另一个主机上恢复，和在另一个主机上启动没什么差别吧。除了web项目中有大量的缓存需要保留，这种情况还比较有用，但是缓存放在web服务里，万一web服务崩掉了岂不是很危险。所以我到现在还没想到CRIU在哪种情况下比较有用。
但是对于docker来说比较有用，docker如果用上CRIU可以直接将整个容器直接暂停然后迁移到另一个主机上，这种情况还比较有用，因为docker本身就是一个环境，保存了很多应用运行环境信息，直接开一个不一定能满足要求，直接迁移过去很方便。（但是docker也可以将运行的容器打成镜像然后再另一个地方运行。）这种比较适合有很多容器的项目整体迁移。

了解到CRIU是在学习docker的时候，看到docker代码里面有criu的文件，然后搜索到的。
其实用criu --help还可以看到很多功能，但是因为英文不好而且Linux基础差，所以很多都不知道。还要继续学习。喵喵喵～

参考：

# Telefork 学习笔记
Telefork 是用 Rust 写的 700 行的类似于 CRIU 功能的 example 项目。

1. 主要分为两部分 telefork() 和 telepad()
2. telefork() 核心的部分是在 fork 的时候，创建子进程，然后将子进程 frozen ，通过 /proc/$child_pid/maps 记录下虚拟内存中的 `地址起始值`。这里只记录了地址的值，并没有传递地址中的数据。做的主要工作是区分出来虚拟内存中哪些是系统调用，哪些是应用的内存。
3. telepad() 核心的部分是在恢复的时候，创建子进程，然后将 fork 时候记录下的虚拟内存的地址值通过系统调用 process_vm_writev 恢复。

虚拟内存中的数据值只传递了 `地址起始值`，并没有传递地址中的数据，恢复地址中的值使用的是系统调用 process_vm_writev, 将一个虚拟地址的值写到另一个虚拟地址中，意味着 telefork 只能在一台主机上使用。
