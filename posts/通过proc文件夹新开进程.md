# 通过/proc/self/exe文件可以新开一个进程

/proc文件夹是一个特殊的文件系统，进程会在/proc文件夹下有个进程号的文件夹，通过运行/proc/self/exe可以重新运行这个进程

新建一个Go的文件main.go

```go
package main

import(
	"fmt"
	"os"
	"time"
)

func main(){
	fmt.Println("===start===")

	// 打印当前的pid，后面我们根据这个pid去新开一个进程
	fmt.Printf("current pid is %#v \n",os.Getpid())

	// 休眠一分钟，使得我们有时间新开一个进程
	time.Sleep(time.Minute)

	fmt.Println("===stop===")

}
```

然后运行这个代码 go run main.go
控制台打印

```shell
===start===
current pid is 2748
```

这个时候我们新开一个终端，在里面执行
```shell
/proc/2748/exe
```

这个时候会看到控制台打印
```shell
===start===
current pid is 2780
```

所以我们通过/proc文件夹下的功能实现了新开一个进程的功能。
在runC里面，就是通过Go在代码里面调用Cmd实现容器的父进程和字进程的创建。创建容器的时候，首先初始化一个父进程，然后通过父进程调用自己打开容器真正的进程。（factory_linux.go第140行）
容器进程和父进程之间是通过SocketPair方式实现通信的；容器进程和父进程运行的是同一套代码，是通过设置环境变量然后读取环境变量来实现逻辑区分的（环境变量使用的时候就相当于一个跨进程的全局变量）。
