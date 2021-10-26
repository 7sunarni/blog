# Redis开发与运维 读书笔记

Redis查询快速的原因：
1. Redis的数据存在内存里面， 读取较快
2. Redis采用单线程设计（采用单线程设计的原因： 在内存中读取很快， 减少了很多线程切换的开销和锁的开销， 为什么不用CAS）
3. Redis用c语言写， 与操作系统交互更快
4. IO多路复用技术（TODO：什么是IO多路复用技术）

## 第二章

### 第一节 Api
可以通过object encoding命令查看value在内部以什么格式存在

### 第二节 字符串
设置字符串用set命令， set命令后面可选选项: ex表示秒级过期时间； px表示毫秒级过期时间； nx表示只有key不存在才能添加成功（官方给出了用set nx来作为分布式锁的实现，TODO 实现demo）； xx表示只有key存在才能设置成功，用于更新。
获得字符串用get命令， get用于获得一个字符串， mget用于获得多个字符串。 文中指出如果要获得多个字符串的时候， 用mget比较好， 原因是因为redis执行一条命令用的时间（在内存中操作）远远小于网络开销的时间（网络传输数据）。如果redis和调用者在同一台电脑上呢。（TODO 通过ping的时间测试get和mget）。
字符串内部编码格式： 
1. int(8字节)， 
2. embstr(embed string?小于39个字节)， 
3. raw(大于39个字节)， redis会根据值的类型和长度实现编码。
TODO： 如果不是int类型的话，就不能使用incr自增命令？如果key从8字节->8+字节，或39->39+字节是否会切换。测试结果是类型会向上向下自动转换的。占用更大的value向下切换的时候也会转换类型。(TODO 基准测试，看类型转换是否影响效率，猜测应该不会)。

### 第三节 hash
hmget用于获取key下的多个field， hmset用于设置key下的多个field， hkeys获取key下所有的field，hvals获取所有的values。hgetall获取所有的field和value，开发建议不要用hgetall，会造成阻塞，建议用hscan命令。（想起来我之前用hash的时候就是用的hgetall拿到数据后在代码中遍历，T_T）。
hash内部编码：
1. ziplist（压缩列表），当list小于hash-max-ziplist-entries（默认512）且所有所有值小于hash-max-ziplist-value（默认64）时候采用。ziplist比hashtable更加节省空间(TODO ziplist的golang实现demo)。
2. hashtable， 当条件无法满足ziplist的时候，采用hashtable实现（TODO：1.也会动态改变数据结构？2.hashtable采用field的hash值来存储？hash值冲突怎么比较是不是同一个元素，毕竟没有equals方法？3.hash冲突之后采用什么数据结构储存）1.会动态改变数据结构
使用场景， 如果说字符串用来做缓存的话， 那么hash更像是数据库了， 看了文中的例子， 其实一个hash key就相当于一张表的一条数据， 只不过这张表的列是动态的， 之前没有意识到， 现在觉得hash这种数据结构更体现出来了Redis作为NoSQL中的SQL部分， 可以用来存放复杂的关系型的数据。

### 第四节 list
list会阻塞，可以用来实现一个简单的pub/sub？ TODO: list内部阻塞是怎么实现的
list内部编码：
1. ziplist 同hash的ziplist
2. linkedlist
3. redis 3.2之后用了quickllist，todo：quicklist的实现

使用场景：可以用list实现需要分页的数据，疑问：既然都放在redis里面了为何不全部拿出来给前端还要分页呢。 其他还可以实现基本的数据结构stack，queue，MessageQueue，Redis的pub/sub使用list吗？

### 第五节 set
集合可以用来去交集sinter并集sunion差集sdiff等操作， 有点像SQL里面的join，这个应该很有用，之前不知道这种操作。

set的内部编码
1. intset，当元素都是整数且数量小于set-max-inset-entries时候用intset这种数据结构？TODO intset数据结构的实现，是不是用的bitmap这种类型？
2. hashtable

使用场景：存放用户的标签，习惯，生成随机数抽奖

### 第六节 zset
zset的zadd命令也又指定参数xx, nx，同string类型一样。
zset集合之间也支持交集并集操作。

内部编码：
1. ziplist，同hash。
2. skiplist，TODO：skiplist的实现。

### 第七节 建管理
redis不支持二级key的过期管理， redis的keys支持正则。TODO: 如果string或者hash通过key查找，是否支持正则？如果支持的话，可能会有安全隐患。get的时候不支持正则。
用scan命令来遍历键可以减少阻塞。



## 第三章 
### 第四节 事务
redis事务不支持回滚？ TODO 完成事务的demo

### 第五节 Bitmap
bitmap可以用来取and，or，not, xor操作。
只提供了Bitmap的api，具体还是要将字符串转换成一个整数来实现。感觉这个用redis和用Java，Golang的实现差别不大。
使用Bitmap的时候要考虑内存的使用，很符合之前公司cto的教导。
HyperLogLog，翻译为超级日志日志？很节省空间，但是有0.81%的误差率，猜测里面的实现用到了Hash,会不会和bloomfilter一样？

GEO业务开发应该用不到

