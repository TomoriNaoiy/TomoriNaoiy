# pytorch
## Dataset 
其实就是 数据集 有好几种 分为图片和标签相连或是一个文件夹放图片 一个文件夹放label 

在pytorch中的导入就是 
```py
from torchvision import Dataset
```
![51da9ba3f93e70831671fb3164360f4b](https://github.com/user-attachments/assets/63c3366b-4a37-4b4c-b1bb-4057ddc90461)

这就是self dataset类的组成 (自己的训练数据) 可以形成自己的数据集以及获得图片

如果直接使用pytorch自带的数据集呢？

```py
train_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
```

### transform
一个很重要的数据处理方法集 里面有一些很重要的方法
- **ToTensor**最重要的一个 将PIL(Image方法获得的图片)或者数组（numpy获得的图片）转化为tensor张量 （一个和数组很像的数据类型 但是他支持GPU加速 更适配深度学习架构）
```py
from torchvision import transforms
train_data = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
```
- 图片裁剪 选择 等等
- 组合操作（compose）
```py
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

## DataLoader
数据处理工具 主要是数据集分组的作用
```py
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
```
batch_size是批次大小 相当于将64份图片分成一个批次 

shuffle参数用于判断每个epoch是否打乱顺序

另外 train_loader是一个迭代器 可以通过遍历获得其中的元素
## 模型构建的开始
通过Sequential打包网络模型 （不用重复写整个网络组成）
```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10)
        )

    def forward(self,x):
        return self.net(x)
```
**super().__init__()** 调用父类的构造函数

**为什么要调用呢？**

module是一个工具箱 必须先调用他 才能知道其中的网络结构

怎么得知的大小为28*28

因为这里是mnist数据集 大小就是28*28 可以通过print（x.shape)来看一看 至于为什么是128 隐藏层你自己设计 64 128 256都可以

**不加flatten呢？**

由于loader进去的是[batch_size,1,28,28] 会报错 因此要展品
## 模型训练
```python
model = Model()
model_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epcho in range(1, 6):
    for data in train_loader:
        x, y = data# x是图像的数据 [64,1,28,28] y是标签 这里是tensor([3,0,4,1,9...])（随机取的）
        output = model(x)#这个呢 很明显 就是最终得分 是一个（64,10）  里面包含了各个的得分 因此我们可以直接argmax获得结果 例如output.argmax(dim=1)
        loss = model_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch{epcho},Loss:{loss.item():.4f}")
```
### 损失函数获得函数  nn.CrossEntropyLoss() 里面包含了sofrmax+log+NLLLoss

### 优化器（当然是梯度下降）
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
使用adam方法 

**parameters()是什么？**

一个生成器 返回model里面所有需要训练的参数 （所有权重和偏置）

lr是学习率 这个不必多说了 cs231n见的不少了 小一点为好 免得跳过了最优点

### 模型评估
```python
correct = 0
total = 0
with torch.no_grad():#关闭梯度计算
    for x, y in test_loader:
        output = model(x)
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
print(f"Test Accuracy: {100 * correct / total:.2f}%")
```
看代码应该挺好理解

### 保存和加载
```python
torch.save(model.state_dict(),'mnist_model.pth')
model2 = Model()
model2.load_state_dict(torch.load("mnist_model.pth"))
```
state_dict可以返回所有参数 

save保存到磁盘

load_state_dict(把参数加载进模型)

**模型会保存到当前脚本运行的路径下**
### 单张图片预测
```python
image, label = test_data[0]
plt.imshow(image.squeeze(), cmap='gray')#squeeze() 去掉多余的维度[1,28,28]->[28,28]
plt.title(label)
```



# kaggle——1 数字识别
```python
import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
class Mydata_1(Dataset):
    def __init__(self,csv_path):
        df=pd.read_csv(csv_path)
        self.x=df.drop('label',axis=1).values.reshape(-1,1,28,28).astype('float32')/255
        self.y=df['label'].values.astype('int64')
    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        return torch.tensor(self.x[idx]),torch.tensor(self.y[idx])

class Mydata_2(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
      
        self.x = df.values.reshape(-1, 1, 28, 28).astype('float32') / 255
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx])

train_data=Mydata_1('train.csv')
test_data=Mydata_2('test.csv')

train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,10)
        

    )
    def forward(self,x):
        return self.net(x)

model=Model()
model_loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)
for epcho in range(1,20):
    for data in train_loader:
        x,y=data
        output=model(x)
        loss=model_loss(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch{epcho},Loss:{loss.item():.4f}")

predictions = []
with torch.no_grad():
    for x in test_loader:
        output = model(x)
        pred = output.argmax(dim=1)
        predictions.extend(pred.tolist())
#print(predictions)


torch.save(model.state_dict(),'mnist_model.pth')
model2=Model()
model2.load_state_dict(torch.load("mnist_model.pth"))

submission=pd.DataFrame({
    "ImageId":np.arange(1,len(predictions)+1),
    "Label":predictions
})
submission.to_csv("submission.csv",index=False)
# image,label=test_data[0]
# plt.imshow(image.squeeze(),cmap='gray')
# plt.title(label)
# model.eval()
# with torch.no_grad():
#     pred=model(image.unsqueeze(0))
#     print(pred.argmax().item())

```
神经网络沿用了先前的线性网络 在训练上取得了95%准确率的成绩

**数据集**

由于这里使用的是kaggle下载的数据集 我们需要根据实际进行数据处理 写出Mydata

在这里 train具有一个label标签 test没用 我们写两个Mydata 第一个在去掉label标签后把剩下的部分重塑成28*28的大小

test部分的mydata直接重塑即可

**test部分的loader不要shuffle** test顺序是固定的 shuffle就打乱了

这里上传的部分  根据下载的csv格式 使用pandas写出csv

