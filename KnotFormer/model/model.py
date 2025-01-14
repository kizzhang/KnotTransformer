import torch
import torch.nn as nn
from util.nn import PositionalEncoding,GaussianRandomFourierFeatures,BasicTransformerBlock,MLP, FinalLayer

class TransDiffuKnotGenerator(nn.Module):
    def __init__(self, input_dim=3, num_classes = 8,  d_model=512, nhead=8, depth=3, max_seq_len=500, dropout=0.2):
        super().__init__()
        self.channel = max_seq_len
        self.d_model = d_model
        self.input_dim = input_dim
        # feature embedding
        self.input_embedding = nn.Linear(input_dim, d_model)

        # add label embedding
        if num_classes is not None:
            self.label_embedding = nn.Embedding(num_classes, d_model)

            # label transformer block
            # lab_en_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=dropout, batch_first=True)
            # self.lab_encoder = nn.TransformerEncoder(lab_en_layer, num_layers=depth)

        # add positional encoding
        self.position_encoder = PositionalEncoding(d_model, max_seq_len)
        
        self.embed_time = nn.Sequential(
            GaussianRandomFourierFeatures(embed_dim = d_model,seq_len= max_seq_len),
            MLP([d_model, d_model, d_model], act=nn.SiLU()), nn.LayerNorm(d_model),
        )
        
        # main blocks
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(d_model, nhead, d_model // nhead, dropout=dropout, cond_dim=d_model)
                for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(d_model)

        self.out = MLP([d_model, d_model, input_dim], act=nn.SiLU())
        self.var_out = MLP([d_model, d_model, d_model, input_dim], act=nn.SiLU())

    def forward(self, src, t, label):
        
        # Embed xyz and atom position
        src_emb = self.input_embedding(src)
        src_emb = self.position_encoder(src_emb)
        
        # Embed time 
        t_emb = self.embed_time(t)
        
        if label is not None:  
            label = label.unsqueeze(-1)
            label_emb = self.label_embedding(label)
            # label_emb = self.lab_encoder(label_emb)

        else:
            label_emb = None

        # Transformer encoder to encode molecules and cross-attention with label embedding
        for block in self.transformer_blocks:
            src_emb = block(src_emb, t_emb, cond=label_emb)
        src_emb = self.norm(src_emb)  
        var_emb = src_emb.detach()  
        
        out = self.out(src_emb)
        var =  self.var_out(var_emb)
        
        return torch.cat([out,var],dim=1)
    
# class TransDiffuKnotGenerator(nn.Module):
#     def __init__(self, input_dim=3, num_classes = 8,  d_model=512, nhead=8, depth=3, max_seq_len=500, dropout=0.2):
#         super().__init__()
#         self.channel = max_seq_len
#         self.d_model = d_model
#         self.input_dim = input_dim
#         # feature embedding
#         self.input_embedding = nn.Linear(input_dim, d_model)

#         # add label embedding
#         if num_classes is not None:
#             self.label_embedding = nn.Embedding(num_classes, d_model)

#             # label transformer block
#             # lab_en_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=dropout, batch_first=True)
#             # self.lab_encoder = nn.TransformerEncoder(lab_en_layer, num_layers=depth)

#         # add positional encoding
#         self.position_encoder = PositionalEncoding(d_model, max_seq_len)
        
#         self.embed_time = nn.Sequential(
#             GaussianRandomFourierFeatures(embed_dim = d_model,seq_len= max_seq_len),
#             MLP([d_model, d_model, d_model], act=nn.SiLU()), nn.LayerNorm(d_model),
#         )
        
#         # main blocks
#         self.transformer_blocks = nn.ModuleList(
#             [BasicTransformerBlock(d_model, nhead, d_model // nhead, dropout=dropout, cond_dim=d_model)
#                 for _ in range(depth)]
#         )

#         self.norm = nn.LayerNorm(d_model)

#         self.split = FinalLayer(d_model, input_dim)
#         self.noise_out = MLP([d_model, d_model, input_dim], act=nn.SiLU())
#         self.var_out = MLP([d_model, d_model, input_dim], act=nn.SiLU())
#     def forward(self, src, t, label):
        
#         # Embed xyz and atom position
#         src_emb = self.input_embedding(src)
#         src_emb = self.position_encoder(src_emb)
        
#         # Embed time 
#         t_emb = self.embed_time(t)
        
#         if label is not None:  
#             label = label.unsqueeze(-1)
#             label_emb = self.label_embedding(label)
#             # label_emb = self.lab_encoder(label_emb)

#         else:
#             label_emb = None

#         # Transformer encoder to encode molecules and cross-attention with label embedding
#         for block in self.transformer_blocks:
#             src_emb = block(src_emb, t_emb, cond=label_emb)
#         src_emb = self.norm(src_emb)    

#         src_emb = self.split(src_emb, label_emb)
#         noise= self.noise_out(src_emb[:,:,:self.d_model])
#         var =  self.var_out(src_emb[:,:,self.d_model:])
        
#         return torch.cat([noise,var],dim=1)

        
class TransformerSequenceClassifier(nn.Module):
    def __init__(self, input_dim=3, d_model=512, nhead=8, depth=3, num_classes=10, max_seq_len=500, dropout=0.2):
        super(TransformerSequenceClassifier, self).__init__()

        # vectorize input features, share weights
        self.feature_embedding = nn.Linear(input_dim*2, d_model)

        # add positional encoding
        self.position_encoder = PositionalEncoding(d_model, max_seq_len)

        # add time encoding
        self.embed_time = nn.Sequential(
            GaussianRandomFourierFeatures(embed_dim = d_model,seq_len= max_seq_len+1),
            MLP([d_model, d_model, d_model], act=nn.SiLU()), nn.LayerNorm(d_model),
        )
        
        # layer norm
        self.layer_norm = nn.LayerNorm(d_model)

        # add CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(d_model, nhead, d_model // nhead, dropout=dropout, cond_dim=d_model)
                for _ in range(depth)]
        )


        # classifier
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, src, t):
        # Embed input features to d_model dimensions, 共享权重 
        bond = src.detach().clone()
        xyz = bond.cumsum(dim=1)
        src = torch.cat([xyz, src], dim=-1)
        src_emb = self.feature_embedding(src)
        
        # Add positional encoding
        src_emb = self.position_encoder(src_emb)

        src_emb = self.layer_norm(src_emb)
        
        # Embed time 
        t_emb = self.embed_time(t)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(src_emb.size(0), -1, -1)
        src_emb = torch.cat((cls_tokens, src_emb), dim=1)

        # Pass through the transformer encoder
        for block in self.transformer_blocks:
            src_emb = block(src_emb, t_emb, cond=None)

        output = src_emb[:, 0]
        output = self.layer_norm(output)
        # Classify
        output = self.classifier(output)
        
        return output