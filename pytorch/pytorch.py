import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
train_data=datasets.MNIST(root='data',train=True,transform=transforms.ToTensor(),download=True)
test_data=datasets.MNIST(root='data',train=False,transform=transforms.ToTensor(),download=False)
train_loader=DataLoader(train_data,shuffle=True,batch_size=64)
test_loader = DataLoader(test_data, batch_size=1000)
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
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
for epcho in range(1,6):
    for data in train_loader:
        x,y=data
        output=model(x)
        loss=model_loss(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch{epcho},Loss:{loss.item():.4f}")
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        output = model(x)
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(),'mnist_model.pth')
model2=Model()
model2.load_state_dict(torch.load("mnist_model.pth"))

image,label=test_data[0]
plt.imshow(image.squeeze(),cmap='gray')
plt.title(label)
model.eval()
with torch.no_grad():
    pred=model(image.unsqueeze(0))
    print(pred.argmax().item())

