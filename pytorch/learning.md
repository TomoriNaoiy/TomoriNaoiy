# pytorch的步骤
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


## 模型构建的开始



