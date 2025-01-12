pandas
====
## 创建
data-{'A':[1,2,3],'B':[4,5,6]}

df=pd.DataFrame(data)

读取df['A']标题为A的一列

df["C"]=df["A"]+df["B"]#新加入一列

df.drop("C",axis=1)删除（axis=0是行 =1是列）

## 分组聚合
df.groupby('A').sum()
A分组 求和
```
eg：
df = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Value1': [10, 20, 30, 40, 50, 60],
    'Value2': [5, 15, 25, 35, 45, 55]
})
```
# 按 Category 列分组并计算 Value1 和 Value2 的和
```
grouped = df.groupby('Category').sum()
print(grouped)
```
### 排序
grouped_sorted=df.groupby(Category).apply(lambda x: x.sort_values('Value', ascending=False))

size()返回每个组的大小 即每个组里面有几行
```
df = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'A', 'B'],
    'Value1': [10, 20, 30, None, 50, 60]
})
```
# 使用 size 统计每个组的大小
```
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
```
holder_num=r_f['Holder'].value_counts(normalize=True)*100
value_count()计算每个唯一holder出现的频次 normalize=True说明转化为相对频率
```
### 分组计算
  df.groupby("Category)[values1].sum()
  
根据分好的category计算每个values

  r_f['Date']=pd.to_datetime(r_f['Date'])
  
将这一行转化为日期格式 方便后面获得数据

  r_f["Date"]=r_f["Date"].dt.date#只获得日期


