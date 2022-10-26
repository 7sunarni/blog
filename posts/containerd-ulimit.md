有一个 k8s 环境使用 contaierd 作为运行时。
containerd 运行 MySQL 容器失败，提示在 /var/run/mysqld 下创建 unix socket 文件锁失败。
对比 docker 和 containerd 运行相同的镜像，发现目录里面不一样。
[root@docker /]# docker run -it --rm mysql:5.6 test-mysql ls /var/run/
lock  mysqld  utmp
[root@containerd /]# ctr run -t --rm mysql:5.6 test-mysql ls /var/run/
[root@containerd /]#
然后手动在 MySQL 启动脚本 /usr/local/bin/docker-entrypoint.sh 创建目录并修改权限。
#!/bin/bash
set -eo pipefail
shopt -s nullglob

+ mkdir -p /var/run/mysqld
+ chown mysql:mysql /var/run/mysqld

# logging functions
mysql_log() {
....
打包新的镜像后运行，容器 OOM 退出了。
从 docker MySQL issue 中找到了是因为设置 ulimit 的原因，然后 containerd 和 K8S 都不能设置 ulimit，于是继续修改 MySQL 启动脚本 /usr/local/bin/docker-entrypoint.sh
#!/bin/bash
set -eo pipefail
shopt -s nullglob

+ mkdir -p /var/run/mysqld
+ chown mysql:mysql /var/run/mysqld
+ ulimit -S -n 20000
+ ulimit -H -n 40000
# logging functions
mysql_log() {
....

重新打包运行成功。

新的 Dockerfile

FROM mysql:5.6
RUN sed -i '6 i mkdir -p /var/run/mysqld' /usr/local/bin/docker-entrypoint.sh &&\
    sed -i '7 i chown mysql:mysql /var/run/mysqld' /usr/local/bin/docker-entrypoint.sh &&\
    sed -i '8 i ulimit -S -n 20000' /usr/local/bin/docker-entrypoint.sh &&\
    sed -i '9 i ulimit -H -n 40000' /usr/local/bin/docker-entrypoint.sh


containerd 的代码中将 nofile 的限制写死了 soft 和 hard 都为 1024。[https://github.com/containerd/containerd/blob/main/oci/spec.go#L157](https://github.com/containerd/containerd/blob/main/oci/spec.go#L157)


ulimit example

``` golang

package main

import (
    "fmt"
    "log"
    "net/http"
    "os"
    "time"
)

func main() {
    log.Printf("current pid %d", os.Getgid())
    http.HandleFunc("/", func(rw http.ResponseWriter, r *http.Request) {
        for i := 0; i < 100; i++ {
            _, err := os.Create(fmt.Sprintf("./temp/%d", i))
            if err != nil {
                rw.Write([]byte(err.Error()))
                return
            }

            time.Sleep(time.Second)
        }
    })
    http.ListenAndServe(":8080", nil)
}

```

```shell

go build -o ul .

chmod +x ./ul 

./ul # 记录下 pid

```

```shell 

打开另一个窗口

prlimit --nofile=20:20 --pid ${PID} # 使用 20 是因为启动程序会打开 socket 文件等。

curl localhost:8080 # ul 执行会卡住 
```
