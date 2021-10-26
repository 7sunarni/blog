# SpringBoot   com.netflix.client.ClientException 问题， 非Load balancer does not have available server问题

## 先说结论， 计算机名最好不要用符号， 用字母表示就行了。在SpringCloud中， 如果计算机名有下划线会导致消费者调用生产者失败。
原因可以参考SpringBoot微服务名不能有下划线。//TODO 确认为什么URL中不能有下划线

运行状况： SpringCloud + SpringBoot项目本地可以好好跑起来没问题。但是在测试服务器上， 消费者调用生产者失败， 消费者代码运行到了熔断器中，没有进到生产者代码， 而直接curl请求生产者是可以拿到结果的。 
打开注册中心， 可以看到注册中心正确注册到了生成者， 消费者。
启动顺序也是先启动生成者， 消费者， 应该没问题。这就奇怪了， 生产者好好的运行， 就是运行不到。

排查问题， 首先在消费者代码中用HttpClient原生请求生产者， 可以拿到结果， 说明服务器端口这些都没问题。
继续排查， 在网上找到了熔断器打印异常代码的方法， 运行后， 打印的异常是 com.netflix.client.ClientException。注意这个异常后面没有其他信息了。
然后Google去用关键字搜， 发现出现提示这个异常的都是 Load balancer does not have available server， 都是说的是没找到生成者， 而我这个没有提示没找到生产者， 说明消费者已经找到了生产者， 而且如果没找到生产者的话本地应该也会报错的。

继续执行， 只打印到了LoadBalancerFeignClient.execute()的调用栈， 点进去看， 后面还有很多很多代码。
而且这个本地可以好好跑， 到了测试服务器就出问题。WTF！
问题来了， 按照这个意思是， 出现这个问题是找到了消费者， 但是为什么调用失败呢。

然后找到了一段代码可以打印 FeignClient的Http请求的代码， 执行下来， 有个惊人发现。
FeignClient执行的http请求， url是 **http://服务名/方法名** 这样的形式， 而之前的手写的HttpClient使用的 **http://ip:port/方法名** 的形式。
很有可能对服务名的url有问题。而直接用curl请求 **http://服务名/方法名** 这个接口是不行的， 说明FeignClient做了处理。 其实这里很好理解为什么要用服务名， 因为多个生产者用一个服务名， 用服务名可以做负载均衡。 （//TODO 负载均衡的计算逻辑在消费者中计算而不是注册中心算？）

现在在FeignClient调用的地方断点debug。因为消费者从注册中心拿了生产者信息会保存在本地， 因为用服务名的url是不能请求到数据的， 所以肯定有地方把服务名进行转换的。
后面就是繁琐的debug了， 最后找到LoadBalancerCommand这个类， 负载中心， 会将服务名转化成注册中心发给消费者的计算机名。注意是计算机名不是IP。（这里可以通过eureka配置为IP）

然后用curl **http://计算机名/方法名**， 可以调用。
下一步， 到了测试服务器， 打开注册中心一看， 卧槽， 计算机名是Sql_server（Windows的测试服务器）， 哇！ 有个下划线， 我立马想到服务名不能有下划线， 然后立马整个流程就想通了， 然后改了计算机名重启后运行OK~

哇， 这次debug费时半天， 差不多5H， SpringBoot的源码debug起来好长一串。通过这次debug， 对FeignClient有了更深入的了解。最后命名不要乱命名， 中划线， 下划线不能随便用。
我以前以为下划线够稳了， 结果还有这种问题， 看来只能驼峰命名了。


[主机名规则参考](https://www.ibm.com/support/knowledgecenter/zh/SSAW57_8.5.5/com.ibm.websphere.nd.multiplatform.doc/ae/rins_hostname.html)