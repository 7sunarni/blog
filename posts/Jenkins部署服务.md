# Jenkins使用

现在公司的服务部署在测试环境下是登录测试环境的电脑（Windows），然后在手动去更新svn，然后在Eclipse中启动。对，你没看错，不是用命令行启动，而是在Eclipse中启动。其实我觉得用Windows作为服务器就很奇怪了。

最近事情做完了，便想着在测试服上搭建个Jenkins作为持续集成的，不用每次都远程到测试环境部署了。

之前也只是使用过Jenkins，并没有搭建过Jenkins的环境，在测试服务上下载后jenkins.war后就直接启动了，然后配置好账号密码这些。

我们公司开发是SpringBoot的微服务，一般来说，开发的时候需要用到三个项目，最后测试环境启动的时候需要启动两个项目（其中一个项目是类似公共类的）大概来说是C依赖B，B依赖A，最后启动的时候只用启动C，B。

先建A项目，配置好项目SVN账号后，每次只需要更新一下A项目，然后把A放入Maven中供B依赖就行了。
```
mvn clean install -Dmaven.test.skip=true
```

然后建B项目，B项目首先需要杀死B之前的进程，否则mvn的时候Windows会提示jar包正在被使用，Windows就是这么坑，fxxk
写Windows的cmd脚本也是坑，一点没shell脚本方便，可能是用的比较少吧。
根据端口杀进程的cmd脚本：
```
@echo off
setlocal enabledelayedexpansion
for /f "tokens=1-5" %%a in ('netstat -ano ^| findstr "0.0.0.0:8001"') do (
 taskkill /f /pid %%e
)

```

8001是端口，正则查找的时候，加上0.0.0.0:比较好，否则会查出来0进程和其他进程，杀不掉或者重复，导致脚本构建失败，Jenkins不能往下走。

杀死端口后，然后开始mvn构建
```
mvn clean install -Dmaven.test.skip=true
```

构建成功后，再在后台启动jar，Windows用start /b，Linux上就直接用nohup &启动就OK了
```
start /b javaw -jar ***.jar
```

然后在B项目的项目中设置触发，在A项目构建成功且稳定后触发。

A项目和B项目的结构差不多，只是设置由B项目触发，其实A，B不用先后顺序，只要A，B都成功构建启动了，注册中心会自动依赖的
设置A由B触发是方便整个项目能快速运行，不然只有等A，B在注册中心重新拿到消息才行。

Jenkins会出现一个问题，在Jenkins中后台启动进程的时候，会提示你文件泄漏，然后Jenkins把自己衍生的进程杀死，应该是防止资源泄漏吧，但是我们需要Jenkins帮我们启动后台进程。
StackoverFlow上有三种解决办法，
1. 配置全局变量BUILD_ID
2. 启动Jenkins时增加命令号参数 java -Dhudson.util.ProcessTree.disable=true
3. 后台启动时候加环境变量

我用第二种方法可以，配置全局环境变量好像没有成功。

TODO LIST
1. cmd的脚本用起真的坑，赋值cmd变量还要在for循环里才能赋值？一点没shell方便
2. Jenkins启动占了很大内存，就开了两个服务就1g内存了？后面优化一下jvm参数看看效果
3. 每次启动的时候没有加jvm参数，可以优化一下。
4. 服务器部署还是Linux好，方便，准确
5. 之前为了找杀指定端口的进程找到了Windows下的一个命令 wmic挺方便的，可以很精确的查找命令行参数的进程
```
wmic process where (caption="java.exe" and commandline like "%jar%") get processid,commandline /value
```
另外Windows下根据端口杀死进程的cmd代码： 8080是目标端口
```
@echo off
setlocal enabledelayedexpansion
for /f "tokens=1-5" %%a in ('netstat -ano ^| findstr "0.0.0.0:8080"') do (
 taskkill /f /pid %%e
)
```

上面的意思是查找进程的进程号，启动是的命令行参数，这个命令行参数里面有jar的参数，然后进程的名字叫java.exe，
这个命令行的查找很奇怪，有点像SQL表达式，还有like正则，我估计后台应该也是一个类似于数据库的项目。

等下次有空了，再帮前端同事用npm打包。
