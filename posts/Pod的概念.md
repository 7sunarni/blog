# Pod的概念

A Pod is a Kubernetes abstraction that represents a group of one or more application containers (such as Docker or rkt), and some shared resources for those containers. Those resources include:
> * Shared storage, as Volumes
> * Networking, as a unique cluster IP 
> * address Information about how to run each container, such as the container image version or specific ports to use

Pod是Kubernetes中的一个抽象概念，代表了一个或多个容器（比如Docker或者rkt）以及这些容器共享的资源，这些资源包括：
> * 共享的储存，比如挂载的卷
> * 网络资源，比如一个唯一的集群IP
> * 如何启动一个容器的地址信息，比如容器镜像的位置或者容器使用到的端口。
