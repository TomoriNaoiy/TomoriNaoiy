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
n=int(input())
num=list(map(int,input().strip().split()))
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



- 整数类型integer（简称；int）
- a=255
- （备注：整数和字符串无法进行运算）
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


**for循环 最唐的还是用海龟举的例子 虽说有些黑历史 但还是蛮直观的 比较我也没比初中厉害多少**
```
格式；for i in range（222）：
：非常重要，不能忘记，还有缩进
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

获得字符串的方式
-len(a)
获得字符串的ascii值（键值）
-ord（a)
获得建值机器
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
### 这个....
按下按键为5
-event.type==KEYDOWN：event.key== K_5
-if elif:
-如果第一个是成功，则不看第二个
-if else
-如果第一个不成功，则第二个一定成功
# 列表
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
----------------------------
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










