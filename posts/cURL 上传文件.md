# cURL上传文件

现在前后端对接口的时候都是用postman或者其他第三方封装的工具，很方便。
但是有时自己通过cURL来调试接口，会更加熟悉http请求中的各个参数，以及这些参数的用途。

一般的接口都比较简单，最近写了一个上传文件的接口，想用cURL来调试却发现不知道用哪个关键字。
然后用 
```shell
curl --help | grep file
```

来找，发现 --append可以用来在uploading 的时候追加一个文件，试了一下，在代码里面却没有找到上传的文件，试了一下其他的几个关键字也不行。
然后Google了一下，找到了一个方法。 通过-F或者--form来上传
```shell
curl -F "file=@/home/user/myfile.txt" -X POST 'http://localhost:port/path'
```

如果需要上传多个文件，则
```shell
curl -F "files[]=@/home/user/myfile1.txt" -F "file[]=@/home/user/myfile2.txt" -X POST 'http://localhost:port/path'
```

但是我为什么通过 grep file正则没找到这个关键字呢，于是我又返回去通过 
```shell
curl --help | grep form
```

发现这个命令是用来上传MIME类型的表单数据的
```
-F, --form <name=content> Specify multipart MIME data
    --form-string <name=string> Specify multipart MIME data
```

我就说怎么通过file关键字没找到。

参考链接:
1. [https://stackoverflow.com/questions/12667797/using-curl-to-upload-post-data-with-files](https://stackoverflow.com/questions/12667797/using-curl-to-upload-post-data-with-files)
2. [https://medium.com/@petehouston/upload-files-with-curl-93064dcccc76](https://medium.com/@petehouston/upload-files-with-curl-93064dcccc76)

另外关于MIME DATA
[https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Basics_of_HTTP/MIME_types](https://developer.mozilla.org/zh-CN/docs/Web/HTTP/Basics_of_HTTP/MIME_types)
