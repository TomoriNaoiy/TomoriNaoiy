python初级
=====
### 非常基础 初中做的 偶尔补充部分知识盲区 现在放到git上面来
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
input=输入
print=打印
问答格式:
```
a=input（".....")
print(a)
```
第一行输入一个数n 在后面3行输入n个数用空格隔开
```
n=int(input())
num=list(map(int,input().strip().split()))
```
strip() 移除字符串开头和结尾的指定字符

如果不传递任何参数，strip() 会移除字符串开头和结尾的空白字符（包括空格、制表符 \t、换行符 \n 等）。

- end=末尾 默认为换行符 可以改成，
- sep=间隔符 默认空格 可以改成，
__________________________________________________________________________________________
- a为变量（可以为任何单词，除特殊单词，开头可以时下划线 不能是数字）such as：color food
不能是关键字 如if
for example：
```
a123=input("请输入一种名字")
_66b=input（“你喜欢吃的东西”）
print（a123+“喜欢吃”+_66b+“....”）
```
___________________________________________________________________________________________
### 小朋友玩的海龟 小时候可喜欢这个了 现在想来真唐
- forward=前进   go to 到达    dot 点

- turtle=海龟
- 使用；先导入turtle模块（import turtle）
- 前进turtle.forward
- 右转+90°turtle.right(90)
- 画点(大小100)；turtle.dot(100)
- 抬笔；turtle.penup
- 落笔；turtle.pendown
- 笔大小；turtle.pensize
- 笔颜色；turtle.pencolor
- 当要使用随机数模块时，需要设置模式
- turtle.colormode(255)
- tuetle.pencolor（random.randint(0,255),random.randint(0,255),random.randint(0,255)）

-------------------------------------------------------------------------------------------
- 随机数：第一种；从1到200中随机选一个整数。random.dandint(1,200)
- 第二种：从1到200中随机选一个数，可小数可分数。random。uniform（1，200）
- 将5个数放入一个碗中，从中随机选一个数。num（200，111，500，233，333，111）random.choice(num)
- 找到最大值：max(num)
- 找到最小值：min（num）
- 移除最大值num.remove（max（num））
----------------------------------------------------------------------------------------------
字符串类型string（简称；str）
a='255'  a=input()
**注意**字符串是不能改变的 也就是以及创建好的字符串 你不能通过a[0]=xxx来强行改变

那么如何修改呢？
使用字符串方法：

字符串提供了许多方法（如replace()、upper()、lower()等），这些方法会返回一个新的字符串，而不是修改原字符串。

```
s = "hello"
s = s.replace("e", "a")  # 将'e'替换为'a'
print(s)  # 输出 "hallo"
```
也可以先转化成列表 修改完后再转回来
```
s = "hello"
s_list = list(s)
s_list[1] = "a"  # 修改第二个字符
s = "".join(s_list)  # 转换回字符串
print(s)  # 输出 "halo"
```

- 整数类型integer（简称；int）
- a=255
- （备注：整数和字符串无法进行运算(加减)）
- 浮点数类型；a=15.0
float
- 整数+浮点数=浮点数
- 整数/整数=浮点数
- 浮点数不能加文字，需要改为字符串

- 将字符串类型改为整数类型
- int（a）
- 将整数类型改为字符串类型
- str（a）
- 将整数或字符串改为浮点数
 - float（a）

每一个字符都有相应的下标
such as：a=hello
print（a[0]）
结果 h
hello
↓↓↓↓↓
0 1 2 3 4
编程中：×=*
÷=/
//是取整 区别于c语言的/
%取余数
int（）向下取整
round（）四舍五入


**for循环 最唐的还是用海龟举的例子 虽说有些黑历史 但还是蛮直观的 毕竟想来我现在也没比初中厉害多少**
```
格式；for i in range（222）：

import turtle,random
r=1
b=1
g=1
turtle.colormode(255)
turtle.speed(100)

for i in range(254):
    turtle.pencolor(r,b,g)
    turtle.forward(60)
    turtle.right(90)
    turtle.forward(60)
    turtle.right(90)
    turtle.forward(60)
    turtle.right(90)
    turtle.forward(60)
    turtle.right(90)
    turtle.right(6)
    b=b+1
    g=g+1
    r=r+1
turtle.forward(200)
for i in range(20):
    turtle.pencolor(r,b,g)
    turtle.forward(60)
    turtle.right(90)
    turtle.forward(60)
    turtle.right(90)
    turtle.forward(60)
    turtle.right(90)
    turtle.forward(60)
    turtle.right(90)
    turtle.right(40)
    b=b-4
    g=g-3
    r=r-9

```
需要使数字每次增加
可以使用for循环
在for循环中
r=r+1
g=g+1
b=b-1

- 获得字符串length的方式
- len(a)
- 获得字符串的ascii值（键值）
- ord（a)
- 将ascii转化成字母 chr（）
```

a=（“输入你想要的内容”）
for i in range（len（a））
	print（ord（a【i】）
#if条件语句
a=100
if a==100：
	print（1）
```
布尔数据（Ture）（False）
逻辑语句and

### 遍历
a=[1,2,3,4,5]
```python
for i in a:
	print(i)
#这个简单 主要讲一个重要用法
for i,j in enumerate(a):
	print(i,j)
#这个方法可以同时遍历元素和索引 有时候很好用
```

### 这个....
按下按键为5
-event.type==KEYDOWN：event.key== K_5
-if elif:
-如果第一个是成功，则不看第二个
-if else
-如果第一个不成功，则第二个一定成功
# 列表
列表推导式 [x for x in range(10) if x%2 == 0]

第一个是填充的部分 后面是循环和判断

list（列表）（可以放多个东西的变量）z
-such as a=['1','2','3'](中间用逗号隔开)
-a.append（1）为这个列表放入1
-print（a[1]）打印出列表中的东西，从0开始，如for循环的i和下标类似
-x%y x除以y取余数
-切片[0:5]
-倒叙[-1：-5]第三位是布长
-不知道具体长度时 可以直接写成[-1::-1]
-a.index("hhhh",n)n是开始的下标.
-.count("h",开始下标，结束下标)
```
a=[]
b=[]
c=a+b
```
会直接把两个列表的元素合并
```
a.insert(下标，str)在某个下标前插入str
pop(下标)
remove(元素)
reverse()倒置列表
.sort()不写 默认从小到大 若sort(reverse=True)则从大到小
sorted()函数 可以对任何形式排序

实用方法

sort()

对列表排序 list.sort()

key=lambda x:x=...根据什么数值排序
```

### 好了 又是以前最喜欢的动画模块 现在用的倒是不多 以后可能会用到？（
```
import pygame
背景设置
from pygame.locals import *
pygame.init() 
screen = pygame.display.set_mode((800, 600))
horse=pygame.image.load(“图片”)
for i in range(1000)1000帧数
导入图片horse=pygame.image.load("图片")
导入多个图片
a=[]
for i in range(8):
	a1=pygame.image,load(str(1+i)+“，pnc”)
	a.append(a1)
把图片放在某一处
screen.blit(图片，(坐标))
把背景图填满整个背景
screen.blit(图片,(0,0))
设置图片大小：
aa=pygame.transform.smoothscale(aa,(50,75))
让图片循环在一个地方出现
screen.blit(a[a%8],(400,300))
clock = pygame.time.Clock() 设置帧数
clock.tick(15)帧数速度15



import pygame
from pygame.locals import *
pygame.init() 
screen = pygame.display.set_mode((800, 600))
house=pygame.image.load("cxkbj.png")
x=800

a=[]
for i in range(51):
    a1=pygame.image.load(str(i+1)"cxk.png")
    a.append(a1)






for i in range(1000):
    screen.blit(house,(0,0))
    screen.blit(a[i%51],(x,300))
    x-=1
  
    
    
    
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
        if event.type == KEYDOWN:
            pass
    pygame.display.update()
```
```
a="........"
设置字体
zhiti=pygame.fong.Font(None,40)
把弹幕变成图片
tanmu=zhiti.render(a,True,(200,200,200))
```
# break continue 还有pass...
跳出循环 break
在循环中使用break，可直接跳出循环
```
such as:
for i in range(3):
	break
```
进入下一个循环
continue
在循环中使用continue，可进入下一次循环
such as:
for i in range(3):
	continue

 ### 和列表切片同理
变量切片
·a=[1,2,3,4,5,6,7,8,9,0,]
·print(a[0:5])【内为下标】
·则打印1，2，3，4，5
·前一个数开始，后一个数的前一个数结束（顾头不顾尾）
·字符串同理
·a=（1，2，3，）
·print（a[0:]）不写表示全部

步伐
- a=[1,2,3,4,5,6,]
- 需要取1，3，5
- print（a[0：6：2]）
```
字典：a={"西瓜"：11，"
苹果"：20}
获得西瓜数量
print（a[西瓜]）
获得所有的键
print（a.keys()）
获得所有的值
print（a.values()）
删除
del a[”西瓜“]
修改
a["西瓜"]=10
获得所有的键和值
print（a.items()）
```
```
嵌套字典
a={aa：{“hh”：1}，
   bb：{zz：2}，
cc：{kk：4
}，}
获得键值方式同上
```
-----------------------------------------------------------------------------------------------------------------------------------
# format 还是蛮重要的

- format
- ：引导
- <左对齐 >右^局中
- ：前面是第几个参数 可以不写
- {0：*（填充的东西）<对齐 20(宽度)}
- {0：.2f}保留俩位小数
- {0：.5}最大显示宽度为5
- {0：b}二进制
- o八进制
- x十六 
- d十进制
- input
- num-list=list(map(int,input().split()))
- abs()绝对值
- print ("{:.}".format(a))**注意 写再print里面**
```
name = "Alice"
age = 25
print("My name is {} and I am {} years old.".format(name, age))
print("My name is {name} and I am {age} years old.".format(name="Alice", age=25))
print("My name is {1} and I am {0} years old.".format(age, name))
```
**这样可以写多个**
----------------------------
既然提到format 那么更好用的f方法当然不能漏掉
```
name = "Alice"
age = 25
print(f"My name is {name} and I am {age} years old.")
```



----------------------------------------------------------------------------------------------------------------------------------------
辗转相除法
```
while n!=0:
m,n=n,m%n
return m
最大公约数
最小公倍数 m*n//m
```
- bin（）hex（）二进制 16进制


二分查找
```
while low<=high:(先排序)
middle=(high+low)//2
if list[middle]>item
high=middle-1
elif list[middle]<item:
	low=middle+1
else:return middle
return None
```
素数判断只需要到int(n**0.5)+1即可

转进制a=input()
- a=int(a,2(后面是标注几进制 转化为10进制))
- strip（）去掉换行符

========================================我是分界线=====================================================
底下就是大一后的笔记了 深度明显增加 有部分重复 可取舍
该笔记为对list lanmda re等语句的认识 谨代表本人的一些简陋认识 附带一些举例语句
============


### List，Dict

list为列表 list=[]

- 可以通过append（） pop（）等方法进行增加和减少
- insert（）进行插入 
- index（）直接获得元素
- （pop（index（a））） remove（）删除指定值

* Dict为字典 dict={}（键值对 a：b，c“d） keys（）
- 以通过输入键从而获得值
- keys（）获得所有键 
- items获得所有键值对 
- values（）获得所有值

# lambda
 将函数使用一行代码表示 带有输入值 输出值（表达式）
```python
f=lambda x: x*x
print(f(5))
#进阶用法
dict=[[lambda x,y:x+1,y],
[lambda x,y:x,y+1]，
[lambda x,y:x-1,y]，
[lambda x,y:x,y-1]】

```
* :.4保留4位小数 *
# Decorator（装饰器）
eg:
```python
def pd(func):
	def pn():#需要调用的函数
	    t1=time.time()
	    num=func()
	    t2=time.time()
	    return num
    return pn
```




调用时
```python
@pd
def p():
	for i in range():
		if a:
			print(a)
		a+=1
	return a
```
运行p（）时 实际运行@pd

# class 魔法方法
calss  定义类
```python
eg:
calss APG：
	def _init_(self,name):
		self.name=name
	def o(self):
		()=()
A=APG（）
#调用时
A.o();
```
魔法方法
-init-(self)
`self.a=a`
编译器会自动运行上方法 在该类中的所有函数都可以调用这个值(类似于c++语言里面的类的public)
还有-str-等魔法方法
(*arg可以用在不知到需要用到多少个属性的时候)

# re正则表达式 
~~根本不好用 仅仅在部分情况 数据及其对称的很有优势 之后学习beautifulsoup和xpath之后就会发现这玩意...~~
* 多用于爬虫
```python
import re  
a=re.compile(r"")
```
此时a是列表
a.findall(变量)
.表示要匹配除了换行符之外的任何单个字符
eg：
 1是黑色
 2是白色
 3是粉色
 4是绿色
 - .色 获得的就是黑色/绿色....
 - * 表示出现任意次数包括0次
eg
1，是黑色
2，是白色
3，是粉色
4，是绿色
要获得，后面的文本
写,.*

- +同上 但不包括0次

- ？ 1次或0次

- {}指定次数 如{3，4}3~4次

- \d  表数字 
- \D 表示一个不是0-9的数字字符
- \s空白字符 空格 tab 换行符等 
- \S非空白字符 
- \w文字字符 
- \W 任意非文字字符
* 贪婪模式
在使用*+？时 他们会尽可能匹配多个
因此需要将其分离时 使用.*?

需要获得的字符串中含有.时 用到转义字符\
eg:
苹果.绿色
西瓜.黑色
菠萝.白色
.*\.
可获得苹果....

获得这一项可能出现几个有效字符
eg:
13500344
1b213121
15231312
10111111
则1[3-7]\d{9}(注：.在中括号里面单纯就是.)
使用^则是非的意思

（）  （里面是需要的）（）外面的是匹配的

# 列表推导式
正常写法
```python
a=[]
for x in range(0,10):
	a.append(x)
```
推导式
`a=[x for x in  range(0,10)]`
最开始的x相当于变量 可以进行符号运算range后面可以加判断
eg` a=[x for x in range(0,10) if x%10==0]`
# generator生成器
eg
```python
nums=[1,2,3,4,5]
def a(nums):
for i in nums:
	yield (i*i)
(不需要return)
b=a(nums)
for c in b:
print(c)
```
# oop面对对象编程
能够减少属性 使编程逻辑更清晰 （其实就是类和对象）
# type hint类型注释
减少由于类型而引起的错误
def f(a:int,b:int) -> int
可以使用mypy进行检测
















