## Spring中的有状态对象和无状态对象
在Spring中bean按生命周期分为5种类
1.singleton：这种类型的bean全局只有一个,单例模式
2.prototype：这种类型的bean每次使用的时候都会初始化一个
3.request：每次请求的时候都会创建一个
4.session：生命周期为会话
5.global session：生命周期为整个会话
注意到singleton不会每次都初始化，多个请求都共用，这种类型的bean在Spring中称为无状态对象，没有实例变量的对象。而prototype每次都会初始化一个实例，这样的bean在Spring中称为有状态对象，是有实例变量的。
Spring中@Service，@Controller，@Dao都是默认的singleton，这就要在使用中注意，不要在singleton的类中使用成员变量，因为一次请求使用了成员变量之后，下一次调用这个bean的时候，这个成员变量依旧会存在，这样就会发生“窜数据”的情况的发生。具体描述不清楚，用代码来演示。下面的代码是一个Controller。

```java
@Controller
public class TestController{
	public StringBuilder trans_data=new StringBuilder("init");

	@RequestMapping("/first")
	public String first(ModelMap map){
		trans_data=trans_data.append("append");
		return first;
	}
	@RequestMapping("/second")
	public String second(){
		map.add("trans_data",trans_data);
		return second;
	}
}
```

运行上述代码，如果直接访问"***/second",在second.jsp中得到的trans_data的值为##"init"##；而如果先访问
"***/first"，然后在访问"***/second"，在second.jsp中得到的trans_data的值就为##initappend##，因为在访问/first的时候，修改了trans_data，而下一次访问的时候，不会重新创建一个TestController，因此数据流窜到了下一次访问当中。
对于request,session,global session，没有用代码尝试，可以猜测的是与他们的生命周期密切相关。