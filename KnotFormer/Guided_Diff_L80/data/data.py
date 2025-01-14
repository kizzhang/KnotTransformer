from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
import os
import torch
import numpy as np

LABEL = ["UN","31","41","51","52","61","62","63"]

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
            print("Openning file ",count, data_file)
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
        # item = np.concatenate((item, bond), axis=1)
        item = bond
        return item
    

    def __getitem__(self, index):
        label, item = self.data[index]
        if self.transform:
            item = self.transform(item)
        return torch.tensor(item, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)

    def __len__(self):
        return len(self.data)


def collate_batch(batch):
    src_batch, label_batch = [], []
    for (src_item, label) in batch:
        src_batch.append(src_item.clone().detach()) # 假设src_item已经是数字化的数据
        label_batch.append(label)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0.0)  # 填充操作
    label_batch = torch.tensor(label_batch, dtype=torch.int64)
    return src_batch, label_batch
