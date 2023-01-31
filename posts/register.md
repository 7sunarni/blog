# assembly

data section: 初始化变量
text section: 代码
bss section: 变量

a*/b* 输入输出相关
c* 变量
d* 长度

```assembly
SYS_EXIT  equ 1
SYS_WRITE equ 4
STDIN     equ 0
STDOUT    equ 1

mov eax, SYS_WRITE         
mov ebx, STDOUT
```


## macro 与 procedure 区别
1. 指令集数量，macro 更少的指令集，10个
2. 内存，macro 更多的内存
3. CALL/RET macro 没有返回
4. 机器码，macro 每次调用都会产生机器码，procedure 只会产生一次机器码
5. 传参，macro 参数作为 macro 的一部分，procedure 的参数通过寄存器和栈内存传参数
6. Overhead time macro 的时间更少
7. 执行速度，macro 的执行速度更快
总结


## [Control Register](https://en.wikipedia.org/wiki/Control_register)
CR0 i386 上 32 bit，x64 上 64 bit
CR0 有不同的控制标记位用于执行处理器的各种基本修改。 
Bit
0: PE
Value 1: protected mode; Value 0: real mode
protected mode: protected virtual address mode 虚拟地址保护？
real mode: real address
1: MP
Monitor co-processor
2: EM
Emulation: x87 FPU
3: TS
task switched: 允许使用 x87
4: ET
extended type 允许使用额外的数学协处理器
5: NE
numberic error
16: WP
Write Protected When set, the CPU can't write to read-only pages when privilege level is 0
18: Alignment mask
31: PG Paging 启用 Paging 并且使用 CR3 寄存器

CR1 保留寄存器，抛出异常
CR2
Contains a value called Page Fault Linear Address (PFLA). When a page fault occurs, the address the program attempted to access is stored in the CR2 register.

CR3
CR3 enables the processor to translate linear addresses into physical addresses by locating the page directory and page tables for the current task. Typically, the upper 20 bits of CR3 become the page directory base register (PDBR), which stores the physical address of the first page directory. If the PCIDE bit in CR4 is set, the lowest 12 bits are used for the process-context identifier (PCID).[1]

CR4
0: VME virtual mode extension
1: PVI protected-mode virtual interrupts
2: TSD time stamp disable
3: DE Debug extension

CR5-CR7 保留寄存器，同 CR1
