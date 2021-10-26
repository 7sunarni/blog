# 浏览器通过文件形式打开HTML不能设置cookie
## 结论是：如果HTML通过文件形式在浏览器中打开，是不能设置浏览器的cookie的。

前几天同事在把代码从测试服务器部署到正式服务器的时候，部署上去了服务端代码之后，然后前端调用的时候提示登录失败。
通过返回的信息，刚开始我们以为是服务端的问题。依次检查了注册中心，生产者，消费者，网关服务之后发现都没问题，然后检查了 缓存服务，发现也没问题，最后一步一步看HTTP请求，发现是前端没有传token。

我在我本地用Chrome打开登录的HTML开始Debug。前端使用vue开发的，前端的同事说他是用npm的一个叫**js-cookie**的来帮助存取token的，**js-cookie**的api大概是
``` JavaScript
from js-cookie import Cookies
Cookies.set("COOKIE_KEY","COOKIE_VALUE");
let cookieValue = Cookies.get("COOKIE_KEY");
```

这个库应该是封装了原生的cookie操作，那为什么不行存进去了之后取不到值了呢。
开始网页debug，从入口进去，慢慢debug终于找到了一段代码，果不其然是封装的原生代码
```JavaScript
// 关于存的代码
document.cookie=****
```

发现取的时候取到的是空。
然后我在Chrome控制台试了一下
```JavaScript
document.cookie = '123' // 这里提示可以成功
document.cookie // 这里发现没有设置进去
```

之后试了很多次，发现都是这个问题，中途还给Cookie设置了超时时间，也是不行。
最后问了一下之前的前端同事，发现是浏览器限制了，如果是以文件协议的形式打开HTML的话，是不能用document.cookie的。
最后在本地开了一个静态服务，然后访问这个HTML，发现cookie是可以用的。

到此结束，但是调试了半天之后，前端同事以为是库出了问题，就改了一种放token的方式。orz~