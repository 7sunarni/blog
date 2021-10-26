# libcontainer学习

Libcontainer是Golang实现的一种用来创建带有namespcaes, cgroups, capabilities和文件系统控制的容器。Libcontainer允许使用者在创建容器之后可以管理容器的生命周期。

1. 使用的时候，先创建factory。用libcontainer.New("")创建一个工厂

2. 创建一个容器包括了两部分，一部分是容器的配置参数config,再通过factory创建出一个容器container；另一部分是容器进程运行的参数Process。Process中有容器运行的CMD命令等，是进程实际运行的方法。


容器启动的时候通过container.run(process)新开一个容器进程。

CRIU(Checkpoing/Restore In Userspace)技术：Linux上实现checkpoint/restore功能的技术。

linux_factory.go 是构建容器的工厂代码
linux_container.go 是构建容器代码
