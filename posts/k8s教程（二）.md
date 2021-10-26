# k8s教程（二）

## 这个教程里面，会以实例的形式展示kubectl常用的资源(Resource): Pod,ReplicaSet,,Deployments,Service等。
kubectl中的资源(Resource)概念和Kubernetes中的资源概念不同，我理解的是针对kubernetes来说的资源是指的是cpu,内存，硬盘等物理硬件资源，kubernetes把所有的node上硬件资源整合在一起形成一个整体对外提供服务。而针对kubectl来说的资源是在kubernetes中用户创建好的用于管理的命名，服务等。
kubectl所有的资源参考[https://kubernetes.io/docs/reference/kubectl/overview/#resource-types](https://kubernetes.io/docs/reference/kubectl/overview/#resource-types)。这里我们主要会用到Pod,ReplicaSet,Deployments,Service几种资源做一个简单的使用。
上面的链接中有说明，nodes的简写是no,pods的简写是po,service的简写是svc,ReplicaSet的简写是rs,Deployment的简写是deploy,后面我们会用简写代替全称，因为全称打起来太多了。特别是Service，ReplicaSet和Deployment这几个。

## 说说我对Pod，ReplicaSet，Deployment，Service的理解：
Pod是k8s里面最小的容器运行实例，一个node里面可以有很多个Pod。如果用微服务的来做比方的话，Pod可以简单的看做是一个微服务实例。如果用SpringCloud的来做比方的话，一个Pod就相当于SpringCloud体系里面的一个SpringBoot实例。当然Pod不只是微服务，可以是其他支持，比如MySQL，Redis，MQ和日志服务等等，都可以创建一个Pod。

ReplicaSet是用来批量生产Pod的，前面说了Pod可以看做是SpringBood微服务，MySQL，Redis等的一个实例，如果我们要部署很多个实例（SpringBoot多个相同的微服务，MySQL集群等）,就需要用ReplicaSet，名字说的很明白了，重复的实例。

Deployment也是用来管理Pod的，和ReplicaSet不同的是，Deployment生成Pod后还可以对Pod进行管理，包括升级，扩展，回滚等操作，更像是在实际运用中的部署场景（版本升级，突发情况扩展应用，严重bug回滚服务），可以用来批量管理一个微服务中的服务（即相当于管理一个SpringCloud中的application-name，一个MySQL集群）。k8s官方推荐的是用Deployment代替ReplicaSet的使用。

Service，看名字就知道是服务了。Service可以通过已经部署好的Pod，然后可以跨Node管理这些Pod统一对外形成一个服务，就是微服务中的一个服务实例。在SpringCloud中，生产者消费者把自己注册到注册中心后，调用的时候是通过服务名来调用并且实现负载均衡的，Service也是同样的道理，Service映射好了Pod之后，会对外暴露出ip和port，然后可以通过Service的ip和port调用到里面Pod的服务，实现负载均衡。我的感觉是Service对SpringCloud项目的微服务没有什么作用，因为在SpringCloud的体系中，微服务是通过服务名来调用的，然后负载均衡的处理是在消费者的FeignClient处理的，而用Go或其他做的微服务，通过k8s的Service调用，负载均衡这些的处理都是k8s的Service处理的。

然后kubectl这个单词也太长了，kubectl看名称应该是KuberneteController的简写，所以我们再次简写一下。用kc表示KuberneteController，后面就用kc来表示kubectl了
```shell
alias kc='kubectl'
```

### 一、创建一个简单的微服务实例
#### 我们先用Golang创建一个简单的HTTP的Server来表示一个微服务。这里用Go的原因

##### 一是我会Java和Go，用Java和Go写一个简单的HTTP服务返回Json，Go几行代码就完成了没有其他依赖，Java还要依赖很多东西。
 
##### 二是要用Docker部署上去，基础镜像的大小，Go如果用alpine的镜像的话，大小只有5m，不过编译代码的时候有限制，而Java的基础镜像，至少要有jre，查了一下，最小的镜像也差不多100m。所以Go用来做微服务，这方面有很大的优势。
 

1. 安装Go并且配置好Go环境后，输入 go 之后看到有提示就说明Go已经可以用了。然后创建目录，并且设置GOPATH环境变量
```shell
cd /home/usr/ && mkdir server && cd server && mkdir go && cd go && mkdir src && cd src # 创建目录
export GOPATH=/home/usr/server/go # 设置GOPATH环境变量
echo $GOPATH # 可以看到输出的是我们设置的路径
pwd # 看一下当前目录
```

2. 然后创建我们后面会用到的文件
```shell
touch main.go Dockerfile k8s.sh # 创建Go的服务代码，Dockerfile和k8s打包上传的文件。
```

3. 然后vim main.go 开始编辑main.go
```Golang
package main
import(
        "encoding/json"
        "net"
        "net/http"
)
func main(){
        http.HandleFunc("/version",msgHandler)
        http.ListenAndServe(":30001",nil)
}
func msgHandler(w http.ResponseWriter, r *http.Request){
        var ip string
        addrs, _ := net.InterfaceAddrs()
        for _, a := range addrs {
                if ipnet, ok := a.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
                        if ipnet.IP.To4() != nil {
                                ip += ipnet.IP.String()+" "
                        }
                }
        }
        m := make(map[string]string)
        m["version"]="1"
        m["ip"]=ip
        b,_ := json.Marshal(&m)
        w.Write(b)
}
```
大概的意思是启动一个http服务，端口是30001，api是/version，访问之后会返回一个版本号和当前服务器的ip。{"version":"1","ip":"192.168.10.154"}
然后在当前目录下 CGO_ENABLED=0 go build，没有报错之后，然后查看路径下有没有apiserver这个二进制文件就可以了。

4. 然后试一下我们的服务有没有问题.
```shell
nohup ./apiserver &
```

5. 启动之后，记住返回的pid。然后调一下我们的接口。
```shell
curl 'http://localhost:30001/version' # 应该可以看到返回的json信息。
```

6. 然后杀掉进程
```shell
kill $pid
```

7. 没问题的话，再vim Dockerfile编辑
```shell
FROM alpine
COPY ./apiserver /vat/tmp/apiserver
EXPOSE 30001
CMD /var/tmp/apiserver
```
大概意思是用alpine镜像，然后拷贝我们的二进制文件过去，暴露30001端口，最后在容器启动的时候运行apiserver。

8. 然后试一下Dockerfile有没有问题
```shell
docker build -t testserver:v0 .
```

9. 构建成功之后，我们看一下这个镜像有没有问题。
```shell
docker run -d --name v0 -d testserver:v0
docker ps
curl 'http://localhost:30001/vesion'
```

10. 如果每步都没问题，说明我们的镜像打包好了。现在删除掉。
```shell
docker stop v0
docker image rm testserver:v0
```

11. 接下来我们编辑上传编译，构建镜像，上传镜像的脚本。vim k8s.sh
```shell
CGO_ENABLED=0 go build
docker build -t 192.168.10.139:5000/apiserver:v1 .
docker push 192.168.10.139:5000/apiserver:v1
```
这里我们编译的时候要加入参数，是因为apline镜像的问题，如果不加参数的话，apline镜像没有cgo的依赖，运行会报错的。
这时候已经把我们的镜像打包好了上传到私有仓库了。


### 二、 Pod的使用
1. 创建一个目录放我们的yaml文件，然后创建yaml文件
```shell
mkdir /home/usr/k8s_demo/ && cd /home/usr && vim pod_demo.yaml
```
```yaml
apiVersion: v1
kind: Pod
metadata:
        name: apiserver
        labels:
                version: v1
spec:
        containers:
        - name: apiserver
          image: 192.168.10.139:5000/apiserver:v1
          imagePullPolicy: IfNotPresent
```

2. 生成Pod
```shell
kc create -f pod_demo.yaml
```

3. 看一下我们的Pod有没有创建好
```
kc get pods -o wide
```
如果我们的Pod是在Running状态，表示已经在使用了，如果失败了，可以通过kc describe pod apiserver-XXXX来看pod启动日志。
具体排错过程参看官方[https://kubernetes.io/zh/docs/tasks/debug-application-cluster/debug-application/](https://kubernetes.io/zh/docs/tasks/debug-application-cluster/debug-application/)

4. 如果没问题的话，可以通过调用接口看有没有问题。
```
curl 'http://上面POD的id:30001/version'
```
5. 到这里，一个微服务Pod创建好了。现在停止掉实例
``` shell
kc delete pod apiserver-XXX
```

### 三、使用ReplicSet批量创建Pod
1. vim rs_demo.yaml 创建yaml文档
```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
        name: apiserver
        labels:
                app: apiserver
                tier: apiserver
spec:
        replicas: 3
        selector:
                matchLabels:
                        tier: apiserver
        template:
                metadata:
                        labels:
                                tier: apiserver
                spec:
                        containers:
                        - name: apiserverv1
                          image: 192.168.10.139:5000/apiserver:v1
                        imagePullSecrets:
                        - name: regcred
```
大概是创建一个三个实例。

2. 然后创建ReplcaSet
```shell
kc create -f rs_demo.yaml
```

3. 查看我们的ReplicaSet和Pod好没有
```shell
kc get rs
kc ge pods -o wide
```

4. 然后用上面的ip每个curl命令一下，看是不是返回对应的ip和版本。
5. ReplicaSet官方不建议用，我们就使用到这里。最后杀掉容器。
```shell
kc delete rs apiserver
```

### 四、使用Deployment创建实例
1. vim deploy_demo_1.yaml 创建配置文件
```yaml
apiVersion: apps/v1beta1
kind: Deployment
metadata:
        name: apiserver-deploy
spec:
        replicas: 3
        template:
                metadata:
                        labels:
                                app: apiserver
                spec:
                        containers:
                                - name: apiserver
                                  image: 192.168.10.139:5000/apiserver:v1
    					imagePullSecrets:
                        		- name: regcred
```

2. 然后创建
``` shell
kc create -f deploy_demo_1.yaml
```

3. 同上所述查看Pod状态然后curl命令查看版本是不是统一的。最后不要杀掉，后面会用到这些东西更新实例。

### 五、 使用Deployment更新服务。
1. 首先我们要去我们的微服务下更新服务
```shell
cd /home/usr/server/go/src && cp main.go main.go.v1 && rm main.go
```
2. 将Go代码的版本返回为2表示已经更新了。
```golang
package main
import(
        "encoding/json"
        "net"
        "net/http"
)
func main(){
        http.HandleFunc("/version",msgHandler)
        http.ListenAndServe(":30001",nil)
}
func msgHandler(w http.ResponseWriter, r *http.Request){
        var ip string
        addrs, _ := net.InterfaceAddrs()
        for _, a := range addrs {
                if ipnet, ok := a.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
                        if ipnet.IP.To4() != nil {
                                ip += ipnet.IP.String()+" "
                        }
                }
        }
        m := make(map[string]string)
        m["version"]="2"
        m["ip"]=ip
        b,_ := json.Marshal(&m)
        w.Write(b)
}
```

3. vim k8s.sh 修改我们的镜像版本
```shell
CGO_ENABLED=0 go build
docker build -t 192.168.10.139:5000/apiserver:v2 .
docker push 192.168.10.139:5000/apiserver:v2
```
4. 最后 sh k8s.sh 运行，成功之后我们的仓库里面就有v2版本的镜像了。

5. cd /home/usr/k8s 然后 vim deploy_demo_2.yaml 创建配置文件
```yaml
apiVersion: apps/v1beta1
kind: Deployment
metadata:
        name: apiserver-deploy
spec:
        replicas: 3
        template:
                metadata:
                        labels:
                                app: apiserver
                spec:
                        containers:
                                - name: apiserver
                                  image: 192.168.10.139:5000/apiserver:v2
                        imagePullSecrets:
                                - name: regcred
```

6. 更新服务
```shell
kc apply -f deploy_demo_2.yaml
```

7. 更新完了之后，赶紧查看kc get pods查看状态，时间快的话可以看到新的Pod创建好了之后，久的Pod在逐渐停止和销毁。Pod的状态是Terminating。
8. 然后查看Pod的状态并且curl命令调一下接口，可以看到返回的Json中version变成2了。
9. 不要销毁我们的Deployment，我们接下来会用Service。

### 六、Service的使用
如果是习惯用SpringCloud的我感觉可以不用Service这个组件，因为前面也说到了，SpringCloud会将分散的服务整理好，然后通过服务名来调用。

1. vim svc_demo.yaml 创建服务的配置文件
```yaml
kind: Service
apiVersion: v1
metadata:
        name: apiserver-service
spec:
        selector:
                app: apiserver
        ports:
                - protocol: TCP
                  port: 80
                  targetPort: 30001
```
意思是创建一个服务，将Pod的label中app为apiserver的Pod整合起来，对外暴露TCP 80端口，映射到里面的Pod的30001端口。

2. 然后创建服务
```shell
kc create -f svc_demo.yaml
```

3. 查看服务详情
```shell
kc get svc -o wide
kc describe svc
```

4. 可以看到我们的服务已经创建好了并且有个ip，详情里面可以看到我们的服务映射了很多Pod，可以通过服务ip来调用。
```shell
curl 'http://SERVICE_IP:80/version'
```
注意，这里我们用的是Service的Ip和80端口，简单理解成nginx就可以了。
多调用几次，可以看到返回的ip都不相同。
现在我们的服务创建好了。
之后，可以通过Deployment回滚， 横向扩展等改变容器，只要容器的selector不变化，我们的Service都可以一直有效。

### TODO:
后面会补上
1. 用Deployment完成横向扩展，升级异常后回滚；
2. k8s集群的内部dns调用，不用ip+port调用服务
3. ca证书完成私有镜像的安全扩展
4. master节点高可用集群化
5. k8s dashboard可视化界面

### Easter Egg:
关于Docker Registry的账号密码为什么要用apdo: Apdo是我最喜欢的一个英雄联盟（LOL)玩家以前的ID。而他现在的ID就是Dopa，哈哈哈，我觉得Dopa这个人超级有天赋又超级努力而且很谦虚好学，虽然不是职业玩家只是一个游戏主播，但是真的把打LOL当做是职业来看，好多从事其他职业的人的职业态度都比不上他。举个简单的例子，Dopa上分的时候一直用卡牌，每局都用卡牌，看都看得要吐了，更别说玩了，对于其他职业中的人来说，就相当于长期重复干一件事，很多人都会受不了吧。比如我，开始一直用Kubectl，感觉都输入恶心了。Dopa的职业态度我觉得我要认真学习。Orz~