# Golang SocketPair使用demo

SocketPair可以跨进程全双工通信，runC使用SocketPari这种方式实现父进程和子进程的通信。以下是demo。

需要在Linux环境下运行，没有用到其他什么技术，应该Linux环境都可以运行。需要依赖的包有goalng.org/x/sys/unix。

新建文件main.go
```golang
package main

import (
        "flag"
        "fmt"
        "os"
        "time"
        "golang.org/x/sys/unix"
        "net"
        "os/exec"
        "io/ioutil"
)

const socketPairDir = "socketPairTestFile"

func main(){
        fmt.Println("===start===")
        fmt.Println(os.Args)

        fmt.Println(os.Getpid())

        for i:=0; i<len(os.Args); i++{
                if os.Args[i] == "child"{
                        childSend()
                        return
                }
        }

        tempDir, err := ioutil.TempDir("","TestPassFD")
        if err != nil{
                fmt.Printf("parent TempDir error %#v \n", err)
                return
        }
        fmt.Printf("tempDir %#v \n",tempDir)

        fds, err := unix.Socketpair(unix.AF_LOCAL, unix.SOCK_STREAM, 0)
        if err != nil{
                fmt.Printf("socketpair error %#v \n", err)
                return
        }
        defer unix.Close(fds[0])
        defer unix.Close(fds[1])

        writeFile := os.NewFile(uintptr(fds[0]), "child-wirte")
        readFile := os.NewFile(uintptr(fds[1]), "parent-reads")
        defer writeFile.Close()
        defer readFile.Close()

        fmt.Println("new process run")
        cmd := exec.Command("/proc/self/exe", "child", tempDir)
        cmd.Stdin = os.Stdin
        cmd.Stdout = os.Stdout
        cmd.Stderr = os.Stderr
        if err:=cmd.Start(); err != nil {
                fmt.Printf("cmd start error %#v \n",err)
                return
        }

        c, err := net.FileConn(readFile)
        if err != nil{
                fmt.Printf("fileConn error %#v \n",err)
                return
        }
        uc, ok := c.(*net.UnixConn)
        if !ok {
                fmt.Println("UnixConn transfer failed")
        }

        buf := make([]byte,32)
        oob := make([]byte,32)

        closeUnix := time.AfterFunc(5*time.Second, func(){
                fmt.Println("timeout reading from unix socket")
                uc.Close()
        })

        _, oobn, _, _, err := uc.ReadMsgUnix(buf,oob)
        closeUnix.Stop()

        scms, err := unix.ParseSocketControlMessage(oob[:oobn])
        if err != nil{
                fmt.Printf("parseSocketControlMsg %#v \n", err)
                return
        }

        if len(scms) != 1{
                fmt.Println("scm length error")
                return
        }

        scm := scms[0]
        gotFds, err := unix.ParseUnixRights(&scm)
        if err != nil{
                fmt.Printf("parseUnixRights err %#v \n",err)
                return
        }

        f := os.NewFile(uintptr(gotFds[0]),"fd-from-child")

        defer f.Close()
        got, err := ioutil.ReadAll(f)
        fmt.Println("=====parent get msg:",string(got))

}

func childSend(){
        var uc *net.UnixConn
        for fd:= uintptr(3); fd <= 10; fd++{
                f := os.NewFile(fd,"unix-conn")
                var ok bool
                netc,_ := net.FileConn(f)
                uc, ok = netc.(*net.UnixConn)
                if ok{
                        break
                }
        }
        if uc == nil{
                fmt.Println("failed to find unix fd")
                return
        }
        flag.Parse()
        tempDir := flag.Arg(1)
        fmt.Printf("tempDir %#v \n", tempDir)

        f, err := ioutil.TempFile(tempDir, "")
        if err != nil{
                fmt.Printf("TempFile err: %v \n",err)
                return
        }
        f.Write([]byte("hello from child \n"))
        f.Seek(0,0)

        rights := unix.UnixRights(int(f.Fd()))
        dummyByte := []byte("x")
        n, oobn, err := uc.WriteMsgUnix(dummyByte, rights, nil)
        if err != nil{
                fmt.Printf("WriteMsgUnix err %v \n", err)
        }
        time.Sleep(10*time.Second)
        fmt.Println(n, oobn)
}

```

然后直接 go run main.go
可以看到在main里面启动了一个子进程，子进程给父进程发了一条消息。
可以通过top -aef | grep main.go查看，应该有两个进程
代码参考：
[https://github.com/gopcp/example.v2/blob/master/src/gopcp.v2/vendor/golang.org/x/sys/unix/syscall_unix_test.go](https://github.com/gopcp/example.v2/blob/master/src/gopcp.v2/vendor/golang.org/x/sys/unix/syscall_unix_test.go)