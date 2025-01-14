from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from data.data import MyDatasetOptimized, collate_batch
from model.model import TransDiffuKnotGenerator
from config.config import Config as cfg
from gaussian.gaussian import GuidedDiffusionProcess 
import numpy as np
import gc


def train_diffuser(cfg):
    
    # Dataset and Dataloader
    knot_ds = MyDatasetOptimized(data_dir=cfg.data_dir,max_length=cfg.max_length, dropout=cfg.data_drop)

    knot_dl = DataLoader(knot_ds, cfg.batch_size, shuffle=True, num_workers=4, collate_fn=collate_batch)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')
    
    # Initiate Model
    model = TransDiffuKnotGenerator(input_dim = cfg.input_dim,
                                    num_classes=cfg.num_classes, 
                                    d_model=cfg.dmodel, 
                                    nhead=cfg.nheads, 
                                    depth=cfg.depth, 
                                    max_seq_len=cfg.max_length, 
                                    dropout = cfg.trans_drop).to(device)
    if cfg.train_dif_from_ckpt:
        try: 
            model = torch.load(cfg.dif_cktp_path)
        except FileNotFoundError:
            print("No checkpoint found for Diffusion model")
                  
    # Initialize Optimizer and Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

    # Initialize Diffusion Process 
    diffusion_process = GuidedDiffusionProcess(
        num_timesteps=cfg.num_diffusion_timesteps, 
        num_sampling_timesteps=cfg.num_diffusion_timesteps
    )
    
    # Best Loss
    best_eval_loss = float('inf')
    
    print('\n--------------------------------------------')
    print(f'\033[93mEpoch  MSE-Loss   VLB-Loss   Total-Loss\033[0m')
    print('--------------------------------------------')
    
    # Train
    for epoch in range(cfg.num_epochs):
        
        # For Loss Tracking
        losses = []
        losses_mse = []
        losses_vlb = []
        
        # Set model to train mode
        model.train()
        
        # Loop over dataloader
        for (knots, labels) in tqdm(knot_dl):
            
            knots = knots.to(device)
            labels = labels.to(device)
            
            # Generate noise and timestamps
            noise = torch.randn_like(knots).to(device)
            t = torch.randint(0, cfg.num_diffusion_timesteps, (knots.shape[0],)).to(device)
            
            # Avoid Gradient Accumulation
            optimizer.zero_grad()
            
            # Calculate training loss
            loss_dict = diffusion_process.training_losses(model, knots, t, noise, labels)
            loss_mse = loss_dict["mse_loss"].mean()
            loss_vlb = loss_dict["vlb_loss"].mean()
            loss = loss_mse + loss_vlb
            
            losses.append(loss.item())
            losses_mse.append(loss_mse.item())
            losses_vlb.append(loss_vlb.item())
            
            # Backprop + Update model params
            loss.backward()
            optimizer.step()
        
        # Mean Losses
        mean_mse_loss = np.mean(losses_mse)
        mean_vlb_loss = np.mean(losses_vlb)
        mean_total_loss = np.mean(losses)
        
        # Display
        print(f'{epoch+1:<6} {mean_mse_loss:<10.4f}  {mean_vlb_loss:<10.4f} {mean_total_loss:<8.4f}', end = "   ")
        
        # Save based on train-loss
        if mean_total_loss < best_eval_loss:
            best_eval_loss = mean_total_loss
            torch.save(model, cfg.dif_cktp_path)
        scheduler.step()   

    print('--------------------------------------------')
    
    # Memory Management
    del model, knots, labels, diffusion_process
    gc.collect()
    torch.cuda.empty_cache()

train_diffuser(cfg.CONFIG)