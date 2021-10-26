# MyBatis类型问题

今天同事用SpringCloud的时候发现一个很奇怪的问题，首先问题描述如下：
微服务A和微服务B之间有个方法，A调用B返回的类型为MyObjectA，简化如下：
```java
class MyObjectA{
	private String Title;
	private String Other;
	private List<MyObjectB> list;
	// 省略get set方法
}

class MyObjectB{
	private String Name;
	private String ID;
	private String BOther;
	// 省略get set方法
}

// C在A中没有用到
class MyObjectC{
	private String Name;
	private Strign ID;
	private String COther;;
}
```

其中MyObjectA, MyObjectA, MyObjectA对应的都有数据库表。
然后调用的时候发现，在服务A中拿到的MyObjectA的实例中，list中的MyObjectB的变量全为null，而MyObjectA的其他变量比如Title，Other这些不为空。但是在服务B中打断点看到的返回的MyObjectA中的list的MyObjectB的变量是有值的，这就很奇怪了。

看到问题，我首先想到的是MyObjectA,MyObjectB是不是没有写get set方法导致序列化出问题了，然后检查了一下，MyObjectA,MyObjectB都有写，然后在A的熔断器里面打了日志，发现也没有进入熔断器。代码里面没有加ControllerAdvice，如果进入熔断器了，应该返回的是null，而不是部分数据为null。这种问题很奇怪，引起了我的兴趣。

接下来排查问题，直接curl调用B的服务，发现能够正确的返回json数据，数据也是完全的，这就说明A服务HTTP调用B的时候，拿到返回的数据解析成对象出了问题。说明MyObjectA对象的JSON序列化的一部分除了问题。

于是在A服务中加入代码测试序列化：
```java
MyObjectA a;
// ...
// 中间是从数据库查出来的a
// ...
// JSON库是com.alibaba.fastjson.json的，随便找了一个
String string = JSON.toJSONString(a); // 此行报错 
MyObjectA parseA = (MyObjectA) JSON.parse(string);
```

执行的时候发现报错，然后报错的原因是
```java
java.lang.ClassCastException: *.*.*.MyObjectC cannot be cast to *.*.*.MyObjectB
```

然后检查MyObjectA里面都没有MyObjectC的东西啊
接着检查MyObjectA的来源，MyObjectA从MyBatis里面查出来的，然后查看XML。
```xml
 <resultMap id="baseResult" type="*.*.*MyObjectA">
        <id column="TITLE" jdbcType="VARCHAR" property="Title"/>
        <result column="OTHER" jdbcType="VARCHAR" property="Other"/>
        <collection property="list" column="TITLE" select="*.*.*.MyObjectCMapper.select"/>
    </resultMap>
```

注意到 collection返回的MyObjectCMapper的select方法，返回的是List里的类型是MyObjectC而不是MyObjectB
之后我试了一下，确实能够set上，如果MyObjectC和MyObjectB之间有相同的字段的时候。

这样可以解释成MyBatis返回的MyObjectA中的list其实是List<MyObjectC>而不是List<MyObjectB>，所以在序列化的时候报错了。
但是为什么直接调用B的时候，同样的序列化就没报错呢。

TODO：
1. 深入MyBatis转换对象时候的映射策略
2. 查看SpringBoot为什么直接返回Json序列化没报错
3. 不同的Json包对上述情况进行测试

写得有点乱，orz，后期整理。

继续整理 FeignClient关于Json的解析
SpringBoot默认的Json的库是Jackson，可以通过在配置文件里面修改为Google的gson和阿里的fastjson。
MyObjectA里面的List如果是MyObjectC的话，意思是如果List出错的话，在SpringBoot返回的时候会出现问题，会导致返回的数据有部分会没有。
SpringCloud的FeignClient默认是Apache的HttpClient，也可以通过配置文件修改为OKHttp的Client。
默认情况下，FeignClient里面封装了一些HttpClient，有很多层猜到Apache的HttpClient，之前有断点进去大概看过。
FeignClient相关的类在org.springframework.cloud:spring-cloud:netflix-core这个jar包下。
Http请求后的Response会由SpringDecoder（实现了Decoder接口，这个接口的唯一方法是decode()，将Feign封装的Response（来自于HttpResponse）解析成JavaType）这个类来解析HttpResponse。意思就是这个类是用来将HttpResponse解析成JavaType的，相当于我们平时在代码里面写的
```java
HttpResponse response;
String jsonString = EntityUtils.toString(response.getEntity());
ObjectType object = (ObjectType)JSON.parse(jsonString);
```

在SpringDecoder的decode()函数中，会有HttpMessageConverter接口的实现类，查看实现类，可以看到有很多第三方库都实现了这个接口，包括Jackson，Gson，fastjson等，还有一些没见过的库。意思如果你自己实现了一个解析json的库，要想让FeignClient用你的库来解析http饭后的json，你的库中就应该实现这个接口。接下来在Jackson的实现类里面看到了具体的实现，没什么特别的，就将HttpResponse里面的body读出来，然后用ObjectMapper转换成目标类，HttpResponse读出来的body，前面已经说到因为类型问题，会导致一些数据没有，然后转成目标类的的时候，又有些数据没有，这就导致了最后服务里面的问题。

总结一下前面所说的问题：主要是因为MyBatis写SQL的时候，类型出错的问题，然后SpringBoot的http调用的时候，类经过了json序列化和反序列化两个步骤，因为类型的问题，两个步骤都丢失了一部分数据，最后导致数据缺失了。

如果以后SpringBoot返回的数据有缺失的情况，或者SpringBoot微服务调用之间数据有缺失的情况，可以通过看传递的类的json序列化和反序列化是否有问题来确认。

写的太乱了，后面再深入了解之后整理一下。

TODO: 关于Jackson，fastjson，gson的工作原理还不清楚，需要学习的地方还太多了