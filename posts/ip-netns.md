# netns example
```shell
# !/bin/bash
# add netns
sudo ip netns add ns1
# add veth pair
sudo ip link add veth1 type veth peer veth2
# set veth2 to netns 
sudo ip link set dev veth2 netns ns1
# add bridge
sudo ip link add br1 type bridge
# link veth1 to bridge
sudo ip link set veth1 master br1
# set veth1 up
sudo ip link set veth1 up
# set bridge up
sudo ip link set br1 up
# set netns loopback up
sudo ip -n ns1 link set lo up 
# set netns veth address
sudo ip address add 10.0.20.1/24 dev br1
# set netns veth2 up
sudo ip -n ns1 link set veth2 up
# set route
sudo ip -n ns1 address add 10.0.20.2/24 dev veth2
# set default route
sudo ip -n ns1 route add default dev veth2
# nat to netns
sudo iptables -t nat -A POSTROUTING -s 10.0.20.1/24 ! -o br1 -j MASQUERADE
sudo iptables -t filter -A FORWARD -o br1 -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
sudo iptables -t filter -A FORWARD -i br1 ! -o br1 -j ACCEPT
sudo iptables -t filter -A FORWARD -i br1 -o br1 -j ACCEPT
```
