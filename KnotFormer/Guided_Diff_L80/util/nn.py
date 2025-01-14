import math
import torch
import torch.nn as nn
from typing import Tuple, Optional, List    
from einops import rearrange

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
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

class CrossAttention(nn.Module):
    def __init__(self, query_dim, cond_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads * 2

        if cond_dim is None:
            cond_dim = query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, cond=None):
        h = self.heads

        q = self.to_q(x)

        if cond is None:
            cond = x

        k = self.to_k(cond)
        v = self.to_v(cond)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)    
    
class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        timestep = self.silu(timestep)
        emb = self.linear(timestep)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x
    
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) 

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., cond_dim=None): # dim is d_model = 512
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout)
        self.attn2 = CrossAttention(query_dim=dim, cond_dim=cond_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = AdaLayerNorm(dim)
        self.norm2 = AdaLayerNorm(dim)
        self.norm3 = AdaLayerNorm(dim)

    def forward(self, x, t, cond=None):
        x = self.attn1(self.norm1(x, t)) + x
        x = self.attn2(self.norm2(x, t), cond=cond)+ x
        x = self.ff(self.norm3(x, t)) + x
        return x

class PointEmbed_Add(nn.Module):
    def __init__(self, input_dim=3, d_model=512,  max_len=5000):
        super().__init__()
        self.input_embedding =  nn.Linear(input_dim, d_model)
        # self.input_embedding = nn.Sequential(MLP([input_dim, d_model, d_model],act=nn.SiLU()),
        #                                         nn.LayerNorm(d_model))  # B x N x d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        

    def forward(self, x):
        x = self.input_embedding(x)
        x = x + self.pe[:,:x.size(1)]
        return x
    
class PointEmbed_RBF(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed   
    

class FinalLayer(nn.Module):
    """
    The final layer of Transformer Diffusion Model
    adapted from DiT model
    """
    def __init__(self, hidden_size, input_dim):
        super().__init__()
        self.cross_attn = CrossAttention(query_dim=hidden_size, cond_dim=hidden_size,
                                    heads=8, dim_head=64, dropout=0.1)
        # self.mid =nn.Sequential(MLP([hidden_size, hidden_size, 2 * input_dim], act=nn.SiLU()),
        #                         nn.LayerNorm(hidden_size)) 
        self.linear =MLP([hidden_size, hidden_size, 2 * hidden_size], act=nn.SiLU()) #nn.Linear(hidden_size, 2 * input_dim, bias=True)


    def forward(self, x, c):
        x = self.cross_attn(x, c) + x
        # x= self.mid(x)
        x = self.linear(x)
        return x