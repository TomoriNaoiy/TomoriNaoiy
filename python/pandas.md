pandas
====
前段时间刚刚学了文件操作 想了想还是放到这里来把（
## 文件
先说说不使用pandas的时候怎么写 
```python
f=open('.txt','w',encoding='utf-8')
f=f.read()
#获得全部
f=f.readline()
f=f.readlines()
把一行行变成列表 但是此时一行中的数据大概率会以‘，’连接 也就是一维数据 而且后面还会有‘\n’
这个时候 要干的就是数据处理 首先使用
```python
for i in f:
f.strip()#去掉换行
f.split("，")#逗号也行 变成列表
append（)....
```
这样就完成了简单的数据处理

接下来是数据写入
```python
ls = ['中国','美国','日本']
f = open(fname,'w')
f.write('$'.join(ls))
f.close()
```
其实也很直观 只是f.read变成write 

简单的读写完成了 接下来是一些数据处理的进阶操作
```python
#排序
a=[[1,2,3,4],[2,3,4,5]]
a.sort(key=lambda x:(x[-1],x[0]))
#通过输入的x（就是a）进行以x最后一位进行排序 如果一样就通过x【0】来判断 可以加一个reversed=True 进行倒序排序
a.replace("\n","")#一个小方法 能把里面的比如\n去掉 省的切片 或者对元素进行strip（）处理

join 一个常用的列表操作（不止

a=['wo','shi','da','mo','wang']
print(" ".join(a))通过“” 把可迭代对象a里面的元素通过“ ”连接
```
下面则是一些进阶·
```python
with open(文件名，读写模式。编码格式，行行之间是否有空行)as 文件变量:
#省的使用f.close() 但要记得缩进
csv.write(csvfile) 创建写入器（文件变量）
writerow（）一行行写入数据
读取数据pd.read_csv()
pd.read_csv("文件名.csv",encoding="编码格式",header=None(这一句没有就会把第一行当作表头))
pandas中添加列名:dataframe.column=["需要的内容",'','','']
把dataframe数据写入csv文件
pets.to_csv(“名字.csv”,encoding="gbk",header=True,index=None)
pd.to_numeric()吧数据装换成数字类型，不能字母型字符串
df.apply
```


## 创建
data-{'A':[1,2,3],'B':[4,5,6]}

df=pd.DataFrame(data)

读取df['A']标题为A的一列

df["C"]=df["A"]+df["B"]#新加入一列

df.drop("C",axis=1)删除（axis=0是行 =1是列）

## 分组聚合
df.groupby('A').sum()
A分组 求和
```python
eg：
df = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Value1': [10, 20, 30, 40, 50, 60],
    'Value2': [5, 15, 25, 35, 45, 55]
})
```
# 按 Category 列分组并计算 Value1 和 Value2 的和
```python
grouped = df.groupby('Category').sum()
print(grouped)
```
### 排序
grouped_sorted=df.groupby(Category).apply(lambda x: x.sort_values('Value', ascending=False))

size()返回每个组的大小 即每个组里面有几行
```python
df = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'A', 'B'],
    'Value1': [10, 20, 30, None, 50, 60]
})
```
# 使用 size 统计每个组的大小
```python
size = df.groupby('Category').size()
print(size)
```
output：

- A 3
- B 3


### count（）计算非缺失值的数量
A 3

B 2

### 计算
```python
holder_num=r_f['Holder'].value_counts(normalize=True)*100
value_count()计算每个唯一holder出现的频次 normalize=True说明转化为相对频率
```
### 分组计算
  df.groupby("Category)[values1].sum()
  
根据分好的category计算每个values

  r_f['Date']=pd.to_datetime(r_f['Date'])
  
将这一行转化为日期格式 方便后面获得数据

  r_f["Date"]=r_f["Date"].dt.date#只获得日期


