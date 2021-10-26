# Golang Random生成真随机数

在写Raft的demo的时候，需要每个follower设置一个心跳超时时间，如果超过了这个心跳超时还没收到leader的心跳，就开始进行下一轮选举。

如果大家同时开始选举，都会先投自己一票，投了自己之后就没法投别人了，这个时候整个系统就会陷入没人投对方的状态，就选不出来leader了。这时候有个选举超时，如果过了选举超时没选出来leader，就再次进入下一轮选举。

为了保证每次每个candidate的开始选举时间不同（如果又相同，每个都又投了自己一票，又陷入了不能投对方的状态，又选不出来leader，进入死循环了)，每个节点的选举超时时间都会在一个基准时间上随机一段时间。

（我写的demo超时时间是20s，然后随机时间是5s。时间大是为了系统请求慢，方便debug排除问题，真实的超时应该在几百ms左右）。我用的随机直接就是random.Int63n(5)。结果跑起来的时候发现每个节点的超时时间都相同，然后节点都同时开始发起竞选，导致每次都没办法选出来新leader。

通过打印超时的时间，发现每个节点（我设置了5个节点）的每次的随机时间都相同，我才想起这个应该是生成了伪随机数，就是每次生成的随机数其实在系统里面已经是固定的了。然后查了一下，可以给random设置随机种子生成真正的随机数，然后再跑起来，每个节点的随机时间都相同了。

生成真随机数需要在调用随机函数前设置随机种子，一般传unix时间戳就可以了
``` go
rand.Seed(time.Now().UnixNano())
// 生成随机数
realRandom = rand.Int63n(RANDOM_SIZE)
```

参考链接：
[https://stackoverflow.com/questions/45753397/why-does-golang-repeats-the-same-random-number](https://stackoverflow.com/questions/45753397/why-does-golang-repeats-the-same-random-number)