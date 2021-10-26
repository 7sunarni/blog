# k8s教程

## 教程是为了记录自己本人学习k8s的安装搭建，以及k8s基础概念的理解，k8s各组件Pod，RepliacationSet，Deployment以及Service的使用，最后会模拟一个真实的生产环境，部署，升级以及回滚的实例。教程需要有Docker的使用经验，能掌握基础的Docker命令，有一定的Linux命令行使用经验。教程所需的依赖以及环境全部可以通过官方网站或国内镜像获得，无需翻墙。

项目假设k8s集群有三个节点，一个master节点，两个node节点。先创建一个master节点的虚拟机，配置好环境之后可以复制过去。

### 一、首先利用VirtualBox创建虚拟机
* 选择Ubuntu/Debian，CentOS都可以。我自己用的Debian10，如果用Ubuntu/Debian的话，可以完全其中的命令来运行。
* 创建虚拟机的时候，主机名最好简单易区分，这里我创建主节点，主机名（hostname）就命名为master。
* 创建虚拟机的时候，版本尽量选择高一点的，也不要太高，免得出现不支持的情况。orz~
* 虚拟机最好选用server版，不用选择desktop版本。
* 创建的虚拟机cpu分配至少要2个，内存最好2GB以上，硬盘大一点。如果cpu数量少于两个，运行k8s的时候会提示报错；内存太小的话，初始化k8s集群的时候可能会出现卡顿，或者初始化完成Docker镜像跑起来之后确提示失败的现象；硬盘太小的话，下载镜像可能出现磁盘空间不足。（有一次我是因为分区给根目录/太小了，导致下载不下来镜像）
* 创建好第一个虚拟机之后不要着急复制，配置好一些环境之后再复制节点。

### 一、 配置虚拟机环境
* 虚拟机安装好后，确定虚拟机里面有vim编辑器，后面会用vim来编辑各种文件；然后看一下hostname是不是自己设定的，如果不是就要改一下；最后看一下虚拟机能不能和宿主机和外网连上，通过ping宿主机ip和ping www.baidu.com来看，如果不能同时访问宿主机和外网，就要去virtualbox配置一下。具体配置网络的可以看其他教程。

### 二、 安装Docker以及配置镜像
* 首先安装Docker，可以参考Docker官网安装，整个安装过程没有被墙，可以简单。
[https://docs.docker.com/install](https://docs.docker.com/install/) 根据自己的系统一步一步复制命令就可以安装刚好了。
如果是用的Debian/Ubuntu，可以用下面我复制过来的命令。
```shell
sudo apt-get update
sudo apt-get install \
	apt-transport-https \
	ca-certificates \
curl \
	gnupg2 \
	software-properties-common
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
	"deb [arch=amd64] https://download.docker.com/linux/debian \
	$(lsb_release -cs) \
	stable"
apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

一路安装下来应该不会出现什么问题。
最后在命令行输入docker version，如果出现docker的详情信息，说明docker已经安装好了。
2. 安装好Docker之后，因为DockerHub在国外，国内很慢，需要切换到国内的镜像（网易的镜像，感谢网易的开源镜像）。
在命令行输入 vim /etc/docker/daemon.json
``` json
{
"registry-mirrors":["http://hub-mirror.c.163.com"]
}
```

编辑完后，保存退出。然后重启docker。
Debian/Ubuntu用 systemctl restart docker，CentOS用service restart docker重启Docker。
这个daemon.json文件后面还会用到，我们会在k8s集群里面建一个私有的Docker镜像仓库，用于存放自己创建的镜像，不用上传到DockerHub上。

### 三、 安装K8s安装包
k8s所必需的依赖包被墙了，需要从阿里的镜像下载[https://opsx.alibaba.com/mirror?lang=zh-CN](https://opsx.alibaba.com/mirror?lang=zh-CN),到网站后找到kubernetes的镜像，然后点击帮助，有Ubuntu/Debian,CentOS的安装命令，下面是Ubuntu/Debian的安装命令。
```shell
apt-get update && apt-get install -y apt-transport-https
curl https://mirrors.aliyun.com/kubernetes/apt/doc/apt-key.gpg | apt-key add - 
cat <<EOF >/etc/apt/sources.list.d/kubernetes.list
deb https://mirrors.aliyun.com/kubernetes/apt/ kubernetes-xenial main
EOF  
apt-get update
apt-get install -y kubelet kubeadm kubectl
```
2. 安装好之后，启动kubelet。Ubuntu/Debian 用systemctl start kubelet，CentOS用service start kubelet。

全部安装好之后，可以复制虚拟机了，复制出来两个，一个叫node1,一个叫node2,然后把node1，node2的主机名改了就可以了。
复制完成后，一共有三台虚拟机
* master： 我用的主机名是master, ip是192.168.10.139
* node1: 我用的主机名是node1，ip是192.168.10.143
* node2: 我用的主机名是node2, ip是192.166.10.145
保证master,node1,node2两两之间能ping通ip。
这里要记住master的ip，后面会有用的。

### 四、 搭建Docker Registry私有镜像仓库
我们连上master，这里我们要用master来做私有仓库，供内部上传下载镜像。
完整的Docker镜像私有仓库安装参考 [https://docs.docker.com/registry/](https://docs.docker.com/registry/) ，里面会有配置Docker私有仓库，安装ca证书，配置登录密码的完整教程。这里我们内网使用不必要ca证书，就免去了。因为没有ca证书，所以其他地方登录的时候会提示https错误，所以后面还要再次配置之前提到的daemon.json文件。
配置只需要密码账号的Docker Registry参考StackOverFlow的回答[https://stackoverflow.com/questions/38247362/how-i-can-use-docker-registry-with-login-password](https://stackoverflow.com/questions/38247362/how-i-can-use-docker-registry-with-login-password)
首先Docker拉去仓库镜像文件
``` shell
docker search registry
docker pull registry:2
```

然后创建存放账号密码的文件夹和存放密码账号的文件，账号为apdo 密码为apdo
``` shell
mkdir registry && cd registry && mkdir auth
docker run --entrypoint htpasswd registry:2 -Bbn apdo apdo > auth/htpasswd
```

这个时候看我们配置的密码生成好没有
``` shell
cat auth/htpasswd # 会有这样的输出表示已经创建好了 apdo: $xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

然后启动Docker镜像仓库
``` shell
docker run -d -p 5000:5000 --restart=always --name registry_private  -v `pwd`/auth:/auth  -e "REGISTRY_AUTH=htpasswd"  -e "REGISTRY_AUTH_HTPASSWD_REALM=Registry Realm"  -e "REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd"  registry:2
```

这个时候我哦们的Docker仓库已经创建好了，登录一下
``` shell
docker login localhost:5000
```

用localhost登录的时候，输入账号密码就可以看到success的提示
但是换成master的ip就不行

``` shell
docker login 192.168.10.139:5000
```

这个时候会提示https错误，这时候因为我们没有用ca证书，解决办法是编辑/etc/docker/daemon.json，增加不安全的仓库地址
``` json
{
"registry-mirrors":["http://hub-mirror.c.163.com"],
"insecure-registries":["192.168.10.139:5000"]
}
```

insecure-registries的ip port是镜像仓库的ip和仓库的port。
然后 systemctl restart docker 或者 service restart docker重启Docker。
配置好了之后，用 docker login 192.168.10.139:5000 登录就可以成功了
然后这个时候第一次登录上node1,node2,按照上面的操作更下/etc/docker/daemon.json然后重启Docker，如果按上面的登录可以成功，表示我们的仓库已经建好了。

### 五、启动k8s集群
1. 在master节点，直接运行
``` shell
kubeadm init --pod-network-cidr=10.244.0.0/16 --kubernetes-version=v1.15.1 --apiserver-advertise-address=192.168.10.139 --image-repository registry.aliyuncs.com/google_containers 
``` 
--apiserver-advertise-address是master的ip,--image-repository一定需要，因为k8s的docker镜像是Google的，被墙了，只能用镜像，如果不用--image-repository就要自己手动去拉去镜像然后导入到私有仓库，再配置k8s的拉去的网址，反正麻烦的很，光是搞那个都可以写一篇文章了。

init中可能会出现很多问题，找到日志中提示的error信息然后先看一下有没有什么是常识性的错误。如果有看不懂的问题直接 kubeadm reset 梭哈，重启kubeadm然后再试，还是有问题的话，就去 [https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/troubleshooting-kubeadm/](https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/troubleshooting-kubeadm/) 看解决，直接输入关键字搜，然后看解决就可以了，虽然是英文，但是描述的比较清楚简单，如果还是不行，就Google或者StackOveflow上找一下，应该很快解决好的。

2. 如果创建成功之后，出现一下提示表示集系统已经创建成功了。
```shell
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
 xxxxxxxxxxxxxxxxxx
 xxxxxxxxxxxxxxxxxx
kubeadm join 192.168.10.139:6443 --token uxtwvc.u5hgo6ssjbnd1yld \
--discovery-token-ca-cert-hash sha256:9d68ce045ff06dbe7d782663f844397fc673394510568ec7e321147b7974a696
```
这里我们要复制下kubeadm join这句话，这个是后面节点加入主节点的token凭证，如果过期了需要重新生成token。

3. 然后按提示初始化一下配置文件
``` shell
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

4. 安装k8s集群里面的网络，这里我用的是flannel网络
``` shell
kubectl apply -f  https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
```

这个时候主节点已经配置好了，可以通过kubectl get nodes看到，已经有个主节点在里面了。

5. 这个时候登陆node1,node2,分别在两个节点执行刚才复制的加入的命令就OK了
``` 
kubeadm join 192.168.10.139:6443 --token uxtwvc.u5hgo6ssjbnd1yld \
--discovery-token-ca-cert-hash sha256:9d68ce045ff06dbe7d782663f844397fc673394510568ec7e321147b7974a696
```

等提示加入成功后，切换到master节点，执行kubectl get nodes ，这个时候已经可以看到三个节点都已经加入了，这个时候我们的k8s集群已经创建好了。