numpy的使用
================
Numpy是Python中用于科学计算的核心库。它提供了高性能的多维数组对象，以及相关工具。
---------------------
由于numpy做为python中非常关键的一个库 且使用有一定难度并且需要记忆的部分比较多 故这里写一个教程 记录部分细节和操作awa
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
**高级索引**
```python
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])


print a[[0, 1, 2], [0, 1, 0]]  # Prints "[1 4 5]"#找到[0,0],[1,1],[2,0]
```
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
## 广播机制
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
**下面这个我目前看不太懂 后面练习中慢慢理解吧**
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













