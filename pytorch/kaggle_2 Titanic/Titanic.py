import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# =====================
# 1. 数据加载
# =====================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# 保留 PassengerId 以便最后提交
test_pids = test_df["PassengerId"].values

# 特征选择
num_features = ["Age", "Fare"]
cat_features = ["Pclass", "Sex", "Embarked"]

# 缺失值填充
for col in num_features:
    train_df[col] = train_df[col].fillna(train_df[col].median())
    test_df[col] = test_df[col].fillna(train_df[col].median())

for col in cat_features:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col] = test_df[col].fillna(train_df[col].mode()[0])

# 类别编码
from sklearn.preprocessing import LabelEncoder
encoders = {}
for col in cat_features:
    enc = LabelEncoder()
    train_df[col] = enc.fit_transform(train_df[col])
    test_df[col] = enc.transform(test_df[col])
    encoders[col] = enc

# =====================
# 2. Dataset 类
# =====================
class TitanicDataset(Dataset):
    def __init__(self, df, labels=None):
        self.num_feats = df[num_features].values.astype(np.float32)
        self.cat_feats = df[cat_features].values.astype(np.int64)
        self.labels = labels.values.astype(np.float32) if labels is not None else None

    def __len__(self):
        return len(self.num_feats)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.num_feats[idx], self.cat_feats[idx], self.labels[idx]
        else:
            return self.num_feats[idx], self.cat_feats[idx]

# train/valid 划分
train_data, valid_data = train_test_split(train_df, test_size=0.2, random_state=42)
train_dataset = TitanicDataset(train_data, train_data["Survived"])
valid_dataset = TitanicDataset(valid_data, valid_data["Survived"])
test_dataset = TitanicDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# =====================
# 3. 模型定义
# =====================
class TitanicModel(nn.Module):
    def __init__(self, num_dim, cat_dims, emb_dim=4):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, emb_dim) for cat_dim in cat_dims
        ])
        input_dim = num_dim + len(cat_dims) * emb_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, num_feats, cat_feats):
        embs = [emb(cat_feats[:, i]) for i, emb in enumerate(self.embeddings)]
        embs = torch.cat(embs, dim=1)
        x = torch.cat([num_feats, embs], dim=1)
        return self.fc(x)

cat_dims = [train_df[col].nunique() for col in cat_features]
model = TitanicModel(len(num_features), cat_dims)

# =====================
# 4. 训练
# =====================
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def accuracy(preds, labels):
    return (preds.round() == labels).float().mean().item()

for epoch in range(20):
    # --- train ---
    model.train()
    total_loss, total_acc = 0, 0
    for num_feats, cat_feats, labels in train_loader:
        optimizer.zero_grad()
        preds = model(num_feats, cat_feats).squeeze()
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy(preds, labels)

    train_loss = total_loss / len(train_loader)
    train_acc = total_acc / len(train_loader)

    # --- valid ---
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for num_feats, cat_feats, labels in valid_loader:
            preds = model(num_feats, cat_feats).squeeze()
            loss = criterion(preds, labels)
            val_loss += loss.item()
            val_acc += accuracy(preds, labels)

    val_loss /= len(valid_loader)
    val_acc /= len(valid_loader)

    print(f"Epoch {epoch+1:02d}: "
          f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

# =====================
# 5. 预测 + 提交
# =====================
model.eval()
preds = []
with torch.no_grad():
    for num_feats, cat_feats in test_loader:
        out = model(num_feats, cat_feats).squeeze()
        pred = (out >= 0.5).int().numpy()
        preds.extend(pred)

submission = pd.DataFrame({
    "PassengerId": test_pids,
    "Survived": preds
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv 已生成")
