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
from typing import Tuple, Optional, List
import torch.utils.data



class MLP(nn.Module):
    """Multi-layer perceptron.
    """
    def __init__(self, dims: List[int], act=None) -> None:
        """
        Args:
            dims (list of int): Input, hidden, and output dimensions.
            act (activation function, or None): Activation function that
                applies to all but the output layer. For example, 'nn.ReLU()'.
                If None, no activation function is applied.
        """
        super().__init__()
        self.dims = dims
        self.act = act
        
        num_layers = len(dims)

        layers = []
        for i in range(num_layers-1):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if (act is not None) and (i < num_layers-2):
                layers += [act]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    
class GaussianRandomFourierFeatures(nn.Module):
    """Gaussian random Fourier features.

    Reference: https://arxiv.org/abs/2006.10739
    """
    def __init__(self, embed_dim: int, seq_len: int, sigma: float = 1.0) -> None:
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.register_buffer('B', torch.randn(embed_dim//2) * sigma)
        self.seq_len = seq_len

    @torch.no_grad()
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v_proj =  2 * torch.pi * v.type(torch.float)[:,None] @ self.B[None,:]
        v_proj = v_proj.unsqueeze(1).expand(-1, self.seq_len, -1)
        return torch.cat([torch.cos(v_proj), torch.sin(v_proj)], dim=-1)
    
def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)

# class Topo_condtional(nn.Module):
#     def __init__(self, dim_topo:int, d_model:int) -> None:
#         super().__init__()
        
#         self.topo_embedding = nn.Sequential(
#             MLP([dim_topo, d_model, d_model, d_model], act=nn.SiLU()),
#             nn.LayerNorm(d_model),
#         )
#     def forward(self, topo: torch.Tensor) -> torch.Tensor:

#         return self.topo_embedding(topo)
    
class DenoiseDiffusion:
    
    @torch.no_grad()
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        """
        * `eps_model` is $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        * `n_steps` is $t$
        * `device` is the device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model

        # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)

        # $\alpha_t = 1 - \beta_t$
        self.alpha = 1. - self.beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # $T$
        self.n_steps = n_steps
        # $\sigma^2 = \beta$
        self.sigma2 = self.beta

    @torch.no_grad()
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        #### Get $q(x_t|x_0)$ distribution

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t)
        #
        return mean, var

    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        #### Sample from $q(x_t|x_0)$

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)
        # Sample from $q(x_t|x_0)$
        return mean + (var ** 0.5) * eps
    
    @torch.no_grad()
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, topo:torch.Tensor):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        \begin{align}
        \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
        \textcolor{lightgreen}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """

        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        eps_theta = self.eps_model(xt, t,topo)
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** .5) * eps
    
    def loss(self, x0: torch.Tensor, topo: torch.Tensor, noise: Optional[torch.Tensor] = None):
        
        with torch.no_grad():
            # Get batch size
            batch_size = x0.shape[0]
            # Get random $t$ for each sample in the batch
            t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

            # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
            if noise is None:
                noise = torch.randn_like(x0)

            # Sample $x_t$ for $q(x_t|x_0)$
            xt = self.q_sample(x0, t, eps=noise)
            
            # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.eps_model(xt, t, topo)

        # MSE loss
        return F.mse_loss(noise, eps_theta)
    

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

class TransDiffuKnotGenerator(nn.Module):
    def __init__(self, input_dim=3, topo_dim = 3, target_dim = 3, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024,max_seq_len=500, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        # feature embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.target_embedding = nn.Linear(target_dim, d_model)
        self.topo_embedding = nn.Linear(topo_dim, d_model)

        # add positional encoding
        self.position_encoder = PositionalEncoding(d_model, max_seq_len)
        
        self.embed_time = nn.Sequential(
            GaussianRandomFourierFeatures(embed_dim = d_model,seq_len= max_seq_len),
            MLP([d_model, d_model, d_model], act=nn.SiLU()), nn.LayerNorm(d_model),
        )

        # Topological encoder and decoder
        topo_en_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,batch_first=True)
        self.topo_encoder = nn.TransformerEncoder(topo_en_layer, num_layers=num_encoder_layers)

        topo_de_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,batch_first=True) 
        self.topo_decoder = nn.TransformerDecoder(topo_de_layer, num_layers=num_decoder_layers)

        # Atom encoder and decoder
        src_en_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,batch_first=True)
        self.src_encoder =  nn.TransformerEncoder(src_en_layer, num_layers=num_encoder_layers)
        
        src_de_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,batch_first=True)
        self.src_decoder =  nn.TransformerDecoder(src_de_layer, num_layers=num_decoder_layers)


        self.layer_norm = nn.LayerNorm(d_model)
        # 添加CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # transformer encoder layers
        # classifier
        self.dimdown =  nn.Sequential(MLP([d_model, d_model, input_dim], act=nn.SiLU()))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src,t,topo):
        # Embed topological features to be passed through transformer
        topo_emb = self.topo_embedding(topo)
        topo_emb = self.layer_norm(topo_emb)

        # Embed xyz to be passed through transformer
        src_emb = self.input_embedding(src)
        src_emb = self.position_encoder(src_emb)

        time_embedding = self.embed_time(t)

        src_emb = src_emb + time_embedding
        src_emb = self.layer_norm(src_emb)
        
        
        src_attn = self.src_encoder(src_emb)

        # causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(topo.size(1))
        conditioned = self.topo_decoder(tgt = topo_emb, memory = src_attn)#, tgt_mask =causal_mask, tgt_is_causal =True)
        conditioned = self.src_decoder(tgt =  src_attn, memory = src_attn+conditioned)#, tgt_mask =causal_mask, tgt_is_causal =True)
        output = self.dimdown(conditioned)

        return output


class MyDatasetOptimized(Dataset):
    def __init__(self, data_dir, num_tail=0, transform=None,number_of_files=0,test_length=1100, file_label = '31', dropout =0.5):
        #self.data_files = [file for file in os.listdir(data_dir) if file.startswith('traj') and file.endswith('.npy')]
        # filename rule: traj_knot{knottype}_L{length}_close.npy
        # only load the data with length <= max_length
        self.data_files = [file for file in os.listdir(data_dir) if file.startswith('traj') and file.endswith('.npy') and file.split('_')[1][4:] ==file_label and
                            int(file.split('_')[2][1:])==test_length]
        self.num_files = len(self.data_files)
        self.labels = [file.split('_')[1][4:] for file in self.data_files]
        self.num_labels = len(set(self.labels))

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
            #print(label, data_file)
            label_int = Knot_list.index(label)
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


# 创建 DataLoader
def collate_batch(batch):
    src_batch, label_batch = [], []
    for (src_item, label) in batch:
        src_batch.append(src_item.clone().detach())  # 假设src_item已经是数字化的数据
        label_batch.append(label)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0.0)  # 填充操作
    label_batch = torch.tensor(label_batch, dtype=torch.int64)
    return src_batch, label_batch

def Generate_knot(diffusion, topo_condition:torch.Tensor, n_samples: int, seq_len: int, dim: int, n_step: int, device = 'cuda:0'):

    with torch.no_grad():
        # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
        x = torch.randn([n_samples, seq_len, dim],device=device)
        
        # Remove noise for $T$ steps
        for t_ in range(n_step):
            # $t$
            t = n_step - t_ - 1
            # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
            x = diffusion.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long), topo_condition)
                
        return x



Knot_list = ['un', '31', '41', '51','52','61','62',  '63']
datasets = []
for knottype in Knot_list:
    datasets.append(MyDatasetOptimized(data_dir='/home/zzhang/Lp_knot_id/L300',test_length=300 ,dropout=0.2, file_label=knottype))

TIME_STEP = 1000
model = TransDiffuKnotGenerator(input_dim=3,topo_dim = 3,d_model=512,nhead=8,dim_feedforward=128,num_encoder_layers=5,num_decoder_layers=5,dropout=0.1,max_seq_len=300)
model.load_state_dict(torch.load('/home/zzhang/KnotFormer/Diffusion/Guidance_free/Bond/check_pt/Bond_all_GF_model_20240723_214230_epoch_19.pt')['model_state_dict'])
model.eval()

diffuser = DenoiseDiffusion(model.to('cuda:0'), n_steps=TIME_STEP, device= 'cuda:0')


for knottype, dataset in zip(Knot_list,datasets):
    knot_arr=[]
    batch_size = 16
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, collate_fn=collate_batch)
    
    counter=0
    for data in loader:
        batch, _ = data
        counter+=1
        topodata  = batch[:,:,3:].to('cuda:0')
        print("start generating knot type: {}".format(knottype))
        knot_arr.append(Generate_knot(diffuser,topodata.to('cuda:0'), batch_size, 300, dim = 3, n_step= TIME_STEP))
        print("generated {} configurations...".format(counter*batch_size))
        if counter == 90:
            print('done generating knot type: {}'.format(knottype))
            break

    num = 300
    f = open("rest_generated_from_{}.txt".format(knottype), "w")
    for knot in knot_arr:
        for i in range(knot.shape[0]):
            x,y,z = knot[i,:,0].cpu().detach().numpy(),knot[i,:,1].cpu().detach().numpy(),knot[i,:,2].cpu().detach().numpy()
            f.write(str(num)+"\n"+str("{:.3e}".format(int(133)))+"\n")
            for j in range(num):
                f.write(str(j)+"\t"+ str("{:.5f}".format(x[j]))+"\t"+str("{:.5f}".format(y[j]))+"\t"+str("{:.5f}".format(z[j]))+"\n")
                
    f.close()
    print('done printing knot type: {}'.format(knottype))
    