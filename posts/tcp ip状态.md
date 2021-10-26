# TCP/IP协议中的状态

TCP协议是有状态的协议，TCP共有11种状态
1. CLOSE 关闭状态，表示这个端口没有使用
2. LISTENING 监听，表示服务正在监听这个端口
3. SYNC SEND 对应TCP3次握手，客户端发出第一条消息后的状态
4. SYNC RECV 对应TCP3次握手，服务端接收到第一条消息后的状态
5. ESTABLISHED 对应TCP3次握手，客户端第二条消息后的状态；服务端收到第三条消息后的状态。通信时候的状态也是ESTABLISHED。
6. FIN WAIT 1 客户端结束对话，进入FIN WAIT 1状态，发出TCP4次挥手第一条消息
7. CLOSE WAIT 服务端收到第一条消息，进入CLOSE WAIT状态，然后发出TCP4次挥手第二条消息
8. FIN WAIT 2 客户端收到第二条消息，进入FIN WAIT 2状态，然后发出TCP4次挥手第三条消息，这个时候TCP已经断开双向了，客户端只能接收不能发消息了
9. LAST ACK 客户端收到第三条消息，然后发出第四条消息，这个时候客户端也断开了，等待进入TIME WAIT
10. TIMED WAIT 客户端服务端都进入 TIMES WAIT状态，等待资源被回收
11. CLOSING

一般来说，在服务器上通过netstat看到的最多的状态应该是CLOSE LISTENING ESTABLISHED 和TIMED WAIT
对于HTTP服务来说TIMED WAIT可能会出现很多，应为一般来说HTTP的服务，持续时间不是很长，大量的TIMED WAIT状态会浪费一定的资源，可以通过配置减少TIMED WAIT状态的持续时间。