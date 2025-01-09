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

# 重要部分 访问数组
也就是切片
```python

```



















