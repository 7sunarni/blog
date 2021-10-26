## docker源码学习笔记

### daemon网络的创建
daemon.go中 start()代表函数的启动
docker/docker/daemon daemon.go中 NewDaemon()初始化网络环境


### container网络
config配置 在/usr/tsunami/go_test_project/src/github.com/docker/docker/api/types/container/host_config.go文件中

container 在 /usr/tsunami/go_test_project/src/moby/container/container.go 文件中

networksetting 在 /usr/tsunami/go_test_project/src/github.com/docker/docker/api/types/types.go 文件中

Docker Daemon紧接着创建container对象 /usr/tsunami/go_test_project/src/moby/daemon/create.go containerCreate()函数
这个函数中实现了大部分container的代码 重要

//TODO LIST
找到run部分的代码


Command类型 是docker重要的概念 在

container start在/usr/tsunami/go_test_project/src/moby/daemon/start.go 中	

client的实现代码在/usr/tsunami/go_test_project/src/github.com/docker/docker/libcontainerd/remote/client.go中
process的实现代码在/usr/tsunami/go_test_project/src/github.com/docker/docker/vendor/github.com/containerd/containerd/process.go中

daemon有关container的操作在/usr/tsunami/go_test_project/src/moby/daemon/container_operations.go中

Spec　这个结构体描述的是镜像的基础信息＇

一些地方通过gRPC方式启动