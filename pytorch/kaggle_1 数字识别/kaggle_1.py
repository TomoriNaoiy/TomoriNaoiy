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

