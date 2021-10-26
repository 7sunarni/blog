## 使用Sublime Text编辑器配置关联Mysql
在安装了Mysql和Sublime Text之后，打开Sublime Text，选择Tools=>Build System=>New Build System...
然后在新的配置文件中输入以下json：
```
{
	"cmd": ["D:\\Software\\Software\\mysql-5.7.17-winx64\\bin\\mysql.exe","-u","root","-p123456","-e","source $file"],
	
	"selector": "source.sql"
}
```
然后保存为Mysql.sublime-build，放在Sublime Text安装包下的Data/Package/User文件夹下

然后新建一个语句保存后缀名为.sql，在Sublime Text里面用Ctrl+B就可以直接看到输出了。

sublime快捷键设置
打开Preferences => Key Bindings => 在自定义快捷键中设置
设置删除整行的快捷键

```
{ "keys": ["ctrl+d"], "command": "run_macro_file", "args": {"file": "res://Packages/Default/Delete Line.sublime-macro"} },

```