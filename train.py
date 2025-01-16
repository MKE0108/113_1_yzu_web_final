import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 自定义 Dataset
class PostureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 训练函数
def train_model(normal_data,abnormal_data,epochs=40):

    normal_data = np.array(normal_data)
    abnormal_data = np.array(abnormal_data)
    # 读取数据
    # normal_data = []
    # abnormal_data = []  
    # for file in os.listdir('data/normal'):
    #     normal_data.append(np.load(os.path.join('data/normal', file)))
    # for file in os.listdir('data/abnormal'):
    #     abnormal_data.append(np.load(os.path.join('data/abnormal', file)))

    # 创建标签
    normal_labels = np.ones(len(normal_data))
    abnormal_labels = np.zeros(len(abnormal_data))

    # 合并数据和标签
    features = np.concatenate([normal_data, abnormal_data], axis=0)
    labels = np.concatenate([normal_labels, abnormal_labels], axis=0)

    # 转换为 PyTorch 张量并移动到设备
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    # 创建 Dataset 和 DataLoader
    dataset = PostureDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 定义模型
    class PostureClassifier(nn.Module):
        def __init__(self, input_dim):
            super(PostureClassifier, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # 输出为 0-1 的概率
            )

        def forward(self, x):
            return self.fc(x)

    # 初始化模型并移动到设备
    input_dim = features.shape[1]  # 特征数量
    model = PostureClassifier(input_dim)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 开始训练
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        for batch_features, batch_labels in dataloader:
            # 将批数据移动到设备
            batch_features = batch_features
            batch_labels = batch_labels

            # 前向传播、计算损失和反向传播
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            batch_labels = batch_labels.float().squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # 计算训练集上的准确率
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        outputs = model(features).squeeze()
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == labels).sum().item() / len(labels)
        print(f"Accuracy: {accuracy:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "posture_classifier.pth")
    return accuracy, model