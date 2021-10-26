# k8s 中设计到的网络知识

# iptables, ipvs 
iptables 和 ipvs 用来作为 service 的负载。
iptables 实现 service 是通过 nat 地址装换。

# 容器跨节点通信
1. vxlan 在每个节点上创建 vtep, 将 vxlan 的网络设备和容器的网桥连接起来。
```shell
ip link add vxlan_test type vxlan id 1 remote $remote_ip dstport 4789 dev ens192
ip link set vxlan_test up
ip addr add 10.0.10.1/24 dev vxlan_test 

ip netns add ns1
ip link add br1 type bridge 
ip link set br1 up
ip addr add 192.168.10.1/24 br1 broadcast 192.168.10.255
ip link add veth1 type veth peer veth0
ip link set veth0 netns ns1
ip -n ns1 link set veth0 up
ip -n ns1 link set lo up
ip netns exec ns1 ifconfig veth0 192.168.10.10

ip link set veth1 up
ip link set veth1 master br1

## 在另一个节点中做一样的动作

```
2. ipip calico TODO:

# isito 流量劫持
使用 iptables 转发端口，主要在 iptables nat 表中实现。 使用 --uid-owner 来表示 envoy 的流量不处理。 iptables 的文档中没有 --uid-owner?
```shell
# 最简单的转发出口流量示例
iptables -t nat -A POSTROUTING -p tcp --destination xx.xx.xx.xx/xx --destination-port xx -j REDIRECT --to-destination $port
```
