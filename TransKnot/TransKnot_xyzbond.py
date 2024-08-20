from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import random
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split
from torch.nn.init import kaiming_uniform_
# 假设你已经有了一个完整的数据集实例
LABEL = ["un","31","41","51","52","61","62","63"]
class MyDatasetOptimized(Dataset):
    def __init__(self, data_dir, num_tail=0, transform=None,number_of_files=0,max_length=1100, dropout =0.5):
        #self.data_files = [file for file in os.listdir(data_dir) if file.startswith('traj') and file.endswith('.npy')]
        # filename rule: traj_knot{knottype}_L{length}_close.npy
        # only load the data with length <= max_length
        self.data_files = [file for file in os.listdir(data_dir) if file.startswith('traj') and file.endswith('.npy') and int(file.split('_')[2][1:])==max_length] # <=max_length
        self.num_files = len(self.data_files)
        self.num_labels = len(LABEL)

        self.data = []
        self.transform = transform
        self.num_tail = num_tail

        for count, data_file in enumerate(self.data_files):
            print("文件序号对应的数值",count, data_file)
            if(number_of_files!=0 and count>=number_of_files):
                break
            data_path = os.path.join(data_dir, data_file)
            data = np.load(data_path, allow_pickle=True)
            # drop part
            if dropout > 0:
                data = data[:int(len(data)*dropout)]
            print("first point",data[0,0,:])
            # label 应该是文件名中的knottype对应在self.labels中的index
            label = data_file.split('_')[1][4:]
            label_int = LABEL.index(label)
            print("label",label_int)
            for item in data:
                item = self.preprocess_item(item)
                self.data.append((label_int, item))


    def preprocess_item(self, item):
        # recenter
        item = item - item.mean(axis=0)
        # stack bond vectors
        bond = np.concatenate((item[1:] - item[:-1], item[0:1] - item[-1:]), axis=0)
        item = np.concatenate((item, bond), axis=1)

        return item
    
    def preprocess_item_simple(self, item):
        # recenter 
        item = item - item.mean(axis=0)
        return item

    def __getitem__(self, index):
        label, item = self.data[index]
        if self.transform:
            item = self.transform(item)
        return torch.tensor(item, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)

    def __len__(self):
        return len(self.data)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        return x
    
# 输入 batchsize*N*3的tensor，输出batchsize*1的矩阵
# 先用全链接层向量化输入，共享权重。然后加上位置编码，通过几个transformer编码器层，最后使用序列的聚合表示进行分类。

class TransformerSequenceClassifier(nn.Module):
    def __init__(self, input_dim=3, d_model=512, nhead=8, num_encoder_layers=3, dim_feedforward=1024, num_classes=10, max_seq_len=500, dropout=0.2):
        super(TransformerSequenceClassifier, self).__init__()

        # vectorize input features, share weights
        self.feature_embedding = nn.Linear(input_dim, d_model)
        # add positional encoding
        self.position_encoder = PositionalEncoding(d_model, max_seq_len)

        self.layer_norm = nn.LayerNorm(d_model)
        # 添加CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # classifier
        self.classifier = nn.Linear(d_model, num_classes)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        # add src mask
        src_mask = (src!=0).all(dim=2)

        # Embed input features to d_model dimensions, 共享权重 
        src = self.feature_embedding(src)
        # Add positional encoding
        src = self.position_encoder(src)

        src = self.layer_norm(src)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(src.size(0), -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)

        # expand mask
        src_mask = torch.cat([torch.ones(src.size(0), 1).bool().to(src.device), src_mask], dim=1)

        # Pass through the transformer encoder
        src = self.transformer_encoder(src, src_key_padding_mask=~src_mask)

        src = self.dropout(src)

        # 取出CLS token的输出
        output = src[:, 0]

        # Classify
        output = self.classifier(output)
        return output

dataset = MyDatasetOptimized(data_dir='/home/zzhang/Flexible_knot_id/L300',max_length=300,dropout=0)

# 定义训练集和测试集的比例
train_ratio = 0.90
train_size = int(len(dataset) * train_ratio)
test_size = len(dataset) - train_size

# 使用 random_split 来随机分割数据集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建 DataLoader
def collate_batch(batch):
    src_batch, label_batch = [], []
    for (src_item, label) in batch:
        src_batch.append(src_item.clone().detach()) # 假设src_item已经是数字化的数据
        label_batch.append(label)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0.0)  # 填充操作
    label_batch = torch.tensor(label_batch, dtype=torch.int64)
    return src_batch, label_batch
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=4, collate_fn=collate_batch)

# 输出shape
print("input shape:", train_dataset[0][0].shape)
print("input sample:", train_dataset[0][0][:5])

# 计算训练集上的label分布
label_count = [0 for _ in range(dataset.num_labels)]
for _, label in train_dataset:
    label_count[label] += 1
print(label_count)

# 定义正确率计算函数,输入数据集的data_loader和模型，返回正确率
def accuracy(data_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def save_checkpt(model,loss, optimizer, epoch, acc,scheduler):
    torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss}, '/home/zzhang/Flexible_knot_id/L300/checkpt_no_sta/Transformer_model_{}_epoch_{}_acc_{:.4f}.pt'.format(time.strftime('%Y%m%d_%H%M%S'), int(epoch), acc)) 

def train_model(model, train_loader, epochs=20, lr=0.001, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.5)

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the correct device

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            # print('EPOCH {}: Loss = {:.4f} at train step = {}'.format(epoch,loss.item()))

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.abs().mean())

        # Compute the training accuracy
        train_acc = accuracy(train_loader, model, device)
        print(f'Training Accuracy: {train_acc:.4f}')
        # Compute the validation accuracy
        test_acc = accuracy(test_loader, model, device)
        print(f'Test Accuracy: {test_acc:.4f}')

        if train_acc > 0.9:
            save_checkpt(model,loss, optimizer, epoch, train_acc, scheduler)
        scheduler.step()

# Note: When calling train_model, ensure you have a model, a DataLoader (train_dl),
# and that your device is correctly configured (e.g., 'cuda' for GPU or 'cpu' for CPU).
    
# clear model
# del model

model = TransformerSequenceClassifier(input_dim=6,d_model=512,nhead=8, num_classes=8,dim_feedforward=128,num_encoder_layers=5,dropout=0.1,max_seq_len=300)
train_model(model, train_loader, epochs=20, lr=0.0001, device='cuda')