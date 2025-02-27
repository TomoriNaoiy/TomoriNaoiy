numpy的使用
================
Numpy是Python中用于科学计算的核心库。它提供了高性能的多维数组对象，以及相关工具。
---------------------
由于numpy做为python中非常关键的一个库 且使用有一定难度并且需要记忆的部分比较多 故这里写一个教程 记录部分细节和操作awa

### 学习numpy之前 尤其是机器学习 先学习关于线性代数的知识
**什么是维度？**
在线性代数中，维度（dimension）通常指的是向量空间的基的个数，也就是向量空间中线性无关向量的最大数目。在数据科学和机器学习中，维度的概念与线性代数中的维度概念有一定的联系，但更具体地指数据的结构和形状。


**这是ai的解释**
维度和矩阵的区别？

维度是一维 二维 三维 而矩阵的阶数仅仅是几行几列？（

例如数组中 （3，1）（3，）前者表示的是2维数组 后者表示的是一维数组
如果在矩阵中 或许会想 3，1和3，有啥区别 不都是三行一列吗？

但这里是数组 也就是说 （3，1）是二维数组 三行一列 而（3，）是一维数组 举个例子就是一行（没有列）

也就是说 二维数组 便是矩阵

例如 （100，10）便是100行10列 数据科学中 便是100个样本数量 10个特征

而第一维度一般指第一个元素 也就是行数（样本数量） 第二维度便是特征数量

而 (100, 28, 28, 1)，表示有100个样本，每个样本是一个28x28的单通道图像。我们可以这样解释：(括号里面的每一个元素都是一个维度)
- 第一维度（样本数量）：100，表示有100个样本。
- 第二维度（图像高度）：28，表示每个图像的高度为28像素。
- 第三维度（图像宽度）：28，表示每个图像的宽度为28像素。
- 第四维度（通道数）：1，表示每个图像是单通道的（灰度图）。
# 数组Arrays 
基本的数组创建 跟列表相似
```python
  import numpy as np
  
  a = np.array([1, 2, 3])  # Create a rank 1 array
  print type(a)            # Prints "<type 'numpy.ndarray'>"
  print a.shape            # Prints "(3,)"
  print a[0], a[1], a[2]   # Prints "1 2 3"
  a[0] = 5                 # Change an element of the array
  print a                  # Prints "[5, 2, 3]"
  
  b = np.array([[1,2,3],[4,5,6]])   # Create a rank 2 array
  print b                           # 显示一下矩阵b
  print b.shape                     # Prints "(2, 3)"
  print b[0, 0], b[0, 1], b[1, 0]   # Prints "1 2 4"
```
将列表转化为数组
```python
np.array(list)
```

还有部分特殊用法
```python
import numpy as np

a = np.zeros((2,2))  # Create an array of all zeros
print a              # Prints "[[ 0.  0.]
                     #          [ 0.  0.]]"

b = np.ones((1,2))   # Create an array of all ones
print b              # Prints "[[ 1.  1.]]"

c = np.full((2,2), 7) # Create a constant array
print c               # Prints "[[ 7.  7.]
                      #          [ 7.  7.]]"

d = np.eye(2)        # Create a 2x2 identity matrix
print d              # Prints "[[ 1.  0.]
                     #          [ 0.  1.]]"

e = np.random.random((2,2)) # Create an array filled with random values
print e                     # Might print "[[ 0.91940167  0.08143941]
                            #               [ 0.68744134  0.87236687]]"
```
**记得多元数组里面 是俩个中括号**
# 重要部分 访问数组
1. 切片
```python
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print([:2,:2])#[[1,2],[5,6]]

```
2. 直接访问
```python
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a[0][1],a[0,1]) #都可以
```
1. **注意！！a[0][1]等价于a[0,1]也就是说 切片中a[0][:2]等同于a[0,:2] 要看得懂**
2. **注意：如果b=a[:] 对b进行操作 a中的数值一样会改变**
3. **np里面是可以通过a[0]直接获得一维数组的**
# 特殊选择
```python
b = np.array([0, 2, 0, 1])

# 会对第一到第4行中 第b[i]列进行访问
print a[np.arange(4), b]  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print a  # prints "array([[11,  2,  3],
         #                [ 4,  5, 16],
         #                [17,  8,  9],
         #                [10, 21, 12]])
```
**值的舍去与保留**
```python
np.max()#获得数组中的最大值 可以通过axis=1来使其以行列区分
np.maxinum()#逐元素对比 返回一个数组
np.argmax()#获得最大值的下标 一样大以前一个为主
np.argsort()#获得排序后的数组索引
#**重要方法**#
np.unique()#返回每个元素的唯一值 舍去重复值

```
这一个方法是真的有用
```python
import numpy as np

# 创建一个包含重复元素的数组
array = np.array([1, 2, 2, 3, 4, 4, 4, 5, 6])

# 找出唯一元素
unique_elements = np.unique(array)
print("唯一元素:", unique_elements)

# 找出唯一元素及其在原数组中第一次出现的索引
unique_elements, indices = np.unique(array, return_index=True)
print("唯一元素:", unique_elements)
print("索引:", indices)

# 找出唯一元素及其在原数组中的索引
unique_elements, inverse_indices = np.unique(array, return_inverse=True)
print("唯一元素:", unique_elements)
print("原数组中每个元素在唯一元素数组中的索引:", inverse_indices)

# 找出唯一元素及其在原数组中出现的次数
unique_elements, counts = np.unique(array, return_counts=True)
print("唯一元素:", unique_elements)
print("每个唯一元素出现的次数:", counts)
```
```
**高级索引**
1.
```python
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])


print a[[0, 1, 2], [0, 1, 0]]  # Prints "[1 4 5]"#找到[0,0],[1,1],[2,0]
```
2.列表索引
```python
import numpy as np
a=list(range(3))
aa=np.array([1,3,5,7,9])
aa=aa[a]
print(aa)
#output [1 3 5]
```
np中的数组允许通过以列表为索引获得另一个列表 而在初级list中是不支持的

### 数组的加减
```python
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
#+
print x - y
print np.subtract(x, y)
#-
print x - y
print np.subtract(x, y)
#* 并非矩阵乘法 只是简单的对数相乘
print x * y
print np.multiply(x, y)
#/
print x / y
print np.divide(x, y)
#开根
np.sqrt(x)
#矩阵乘法

dot()
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print v.dot(w)
print np.dot(v, w)
#相当于内积
```
# 数组的处理
数据元素和
```python
import numpy as np

x = np.array([[1,2],[3,4]])

print np.sum(x)  # Compute sum of all elements; prints "10"
print np.sum(x, axis=0)  # Compute sum of each column; prints "[4 6]"
print np.sum(x, axis=1)  # Compute sum of each row; prints "[3 7]"
```
数据的转置
```python
import numpy as np

x = np.array([[1,2], [3,4]])
print x    # Prints "[[1 2]
           #          [3 4]]"
print x.T  # Prints "[[1 3]
           #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print v    # Prints "[1 2 3]"
print v.T  # Prints "[1 2 3]"
```
### 重塑
np.reshape()方法 一般用作改变（重塑）数组的维度
```python
numpy.reshape(a, newshape, order='C')
```
- a：要重塑的数组。
- newshape：新的形状，可以是一个整数或整数元组。如果指定为 -1，NumPy 会自动计算该维度的大小。
- order：可选参数，用于决定元素在内存中的读取顺序。默认值为 'C'，表示按 C 语言风格（行优先）读取；'F' 表示按 Fortran 语言风格（列优先）读取；'A' 表示按数组内存布局读取。

用法
```python
import numpy as np

# 创建一个一维数组
a = np.array([1, 2, 3, 4, 5, 6])

# 重塑为 2 行 3 列的二维数组
b = np.reshape(a, (2, 3))
print(b)
# 输出:
# [[1 2 3]
#  [4 5 6]]
```
**特别** 如果使用-1 会直接计算这个维度大小 并维持
这里以cs231n里面的例子举例
```python
import numpy as np

# 假设 X_train 的原始形状是 (100, 28, 28, 1)
X_train = np.random.rand(100, 28, 28, 1)

# 重塑 X_train
X_train = np.reshape(X_train, (X_train.shape[0], -1))

# 打印重塑后的形状
print(X_train.shape)  # 输出 (100, 784),这里将后面几个维度的数据压缩了
```
对的 你也发现了 我们可以通过print(qwq.shape)来获得一个数组的维度
## 数组处理
**计算非0数**
```python
 row_num=np.count_nonzero(margin > 0, axis=1)#以行为标准 计算非0的个数
np.random.randn(dim, num_classes)#创造一个以高斯分布的数组第一个(dim,num_classes)的数组
np.concatenate((a,b))#合并俩个数组
```

## 广播机制
# 非常重要（好用）
有些复杂 我直接把cs21n里面的说法搬下来
对两个数组使用广播机制要遵守下列规则：

如果数组的秩不同，使用1来将秩较小的数组进行扩展，直到两个数组的尺寸的长度都一样。
如果两个数组在某个维度上的长度是一样的，或者其中一个数组在该维度上长度为1，那么我们就说这两个数组在该维度上是相容的。
如果两个数组在所有维度上都是相容的，他们就能使用广播。
如果两个输入数组的尺寸不同，那么注意其中较大的那个尺寸。因为广播之后，两个数组的尺寸将和那个较大的尺寸一样。
在任何一个维度上，如果一个数组的长度为1，另一个数组长度大于1，那么在该维度上，就好像是对第一个数组进行了复制。
```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print y  # Prints "[[ 2  2  4]
         #          [ 5  5  7]
         #          [ 8  8 10]
         #          [11 11 13]]"
```
以上种种 用人话来说 就是会把一个小的矩阵往另一个矩阵靠 使得俩个矩阵大小相等（一般采用复制的形式）

最后再通过正常运算加减乘除

但是最后获得的数组是变换大小之后的
**下面这个我目前看不太懂 能看懂的看看**
```python
import numpy as np

# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print np.reshape(v, (3, 1)) * w

# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print x + v

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print (x.T + w).T

# Another solution is to reshape w to be a row vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print x + np.reshape(w, (2, 1))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print x * 2
```
**反正我看不懂 我举个比较好懂的例子吧**
```python
a=[[0]
[1]
[2]]
b=[[0,1,2]]
print(a+b)
'''
[[0,0,0]      [[0,1,2]      [[0,1,2]
[1,1,1]     +  [0,1,2]  =    [1,2,3]
[2,2,2]]       [0,1,2]]     [2,3,4]]
```
这下能懂了吧？？

 豪了 numpy目前就到这里了 但numpy是一个强大的库 这些内容远远不止 ~~但我懒了~~
后面有遇到坑或者新的点再做补充吧
顺带写一下scipy和Matplotlib库吧
# Scipy
Numpy提供了高性能的多维数组，以及计算和操作数组的基本工具。SciPy基于Numpy，提供了大量的计算和操作数组的函数，这些函数对于不同类型的科学和工程计算非常有用。

熟悉SciPy的最好方法就是阅读文档。我们会强调对于本课程有用的部分。
### 图像操作
SciPy提供了一些操作图像的基本函数。比如，它提供了将图像从硬盘读入到数组的函数，也提供了将数组中数据写入的硬盘成为图像的函数。下面是一个简单的例子：
```python
from scipy.misc import imread, imsave, imresize

# Read an JPEG image into a numpy array
img = imread('assets/cat.jpg')#图片的相对路径
print img.dtype, img.shape  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.

img_tinted = img * [1, 0.95, 0.9]#对图片颜色数据的改动

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))#img_tinted是对象素进行改动

# Write the tinted image back to disk
imsave('assets/cat_tinted.jpg', img_tinted)#展示图片
```
报错就pip install Pillow
# 计算数组点之间距离
```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print x

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
d = squareform(pdist(x, 'euclidean'))
print d
```
先到这里 之后补充
# Matplotlib
Matplotlib是一个作图库。这里简要介绍matplotlib.pyplot模块，功能和MATLAB的作图功能类似。
对于这个库 之前写数据处理的task时学的不少 这里加一些我自己的之间积累的部分方法吧
```python
import matplotlib.pyplot as plt
plt.plot([1,2,3],[4,5,6])
plt.show()#展示画出的函数
#意味着 以（1，4），（2，5），（3，6）三个点练成线
plt.plot(x,y,laber="Line")
plt.xlable("X")
plt.ylable('y')#给x y轴命名
plt.title()
plt.legend()#添加图例 把上面的xy命名放进去
plt.bar(x, y)  # 绘制柱状图
plt.scatter(x, y)  # 绘制散点图
plt.hist(data)  # 绘制直方图
plt.figure(figsize=(8, 6))设置大小
plt.grid(True)启动网格线
x=np.arrange(0,10,2)一个0，10的数组 布长为2
y=np.linspace(0,1,5)创建0到1直接的5个等间距的数字
#还有一些基本函数
np.sin(x)
from scipy.special import expit 
e^x函数
expit(x)

```
# 多个图像
subplot()
可以在同一个图像里面画多个函数 便于对比~~阅读~~
```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()
```

# 在函数中显示图片
好奇怪的用法qwq
```python
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()
```
这样会标注出长和宽....
**差不多了 后面再补充 俺累了**













