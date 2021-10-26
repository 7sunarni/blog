# Raft协议算法学习笔记

参考一个动画演示地址[http://thesecretlivesofdata.com/raft/](http://thesecretlivesofdata.com/raft/)
Raft的目的是为了维持分布式一致性（TODO：了解分布式一致性的三个不能同时满足的条件）

## Node的状态
每个Node有三种状态
1. Leader State
2. Candicate State
3. Follower State

所有的Node在初始化的时候都是Follower状态。
如果一个Node没有接受到从Follower的消息，那么这个Node变为Candicate状态。
Candicate状态的节点接受来自于其他节点的vote。
Node会回复来自于其他节点的vote。
Candicate的节点变成Leader状态当其受到大多数的节点的vote。
这个过程称为Leader Election。

## Leader Election
Election控制中有两个超时(timeout)
1. 选举超时： 指的是candicate状态的节点变成leader状态的等待时间。选举超时的时间一般控制在150ms-300ms。
超过了选举超时时间，follower状态的节点变成candicate状态并开始新一轮选举。
先vote自己，然后发送requestVote 给其他节点。如果收到的节点在这一轮选举中还没有投票，那么他投票给发来的节点。然后节点重置选举超时时间。
然后得到最多的节点变成leader节点。leader开始发送Append Entries给follower。这些消息发送给节点保持心跳超时。follower回复Append Entries
选举周期将会结束，当起收到心跳超时的时候


## 日志备份

go的ticker删掉后会出bug？
