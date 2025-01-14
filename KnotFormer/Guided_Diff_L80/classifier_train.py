import config.config as cfg 

from tqdm import tqdm
import gc
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


from model.model import TransformerSequenceClassifier
from data.data import MyDatasetOptimized, collate_batch
from gaussian.gaussian import GuidedDiffusionProcess

def train_classifier(cfg):
    
    # Dataset and Dataloader
    knot_ds = MyDatasetOptimized(data_dir=cfg.data_dir,max_length=cfg.max_length, dropout=cfg.data_drop)
    
    # Split Dataset
    train_ratio = cfg.train_ratio
    train_size = int(len(knot_ds) * train_ratio)
    test_size = len(knot_ds) - train_size
    train_dataset, test_dataset = random_split(knot_ds, [train_size, test_size]) ## split into test and train datasets
    print(f'Train Size: {train_size}, Test Size: {test_size}')
    
    # Create Dataloader (add test loader)
    knot_train_dl = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=4, collate_fn=collate_batch)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')
    
    # Initiate Model
    model = TransformerSequenceClassifier(input_dim=cfg.input_dim, 
                                          d_model=cfg.dmodel, 
                                          nhead=cfg.nheads, 
                                          depth=cfg.depth, 
                                          num_classes=cfg.num_classes, 
                                          max_seq_len=cfg.max_length, 
                                          dropout=cfg.trans_drop).to(device)
    if cfg.train_cls_from_ckpt:
        try:
            model = torch.load(cfg.class_ckpt_path)
            print("Loaded model from checkpoint")
        except FileNotFoundError:
            print("No checkpoint found for Classifier model")
    # Initialize Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
    
    # Initialize Diffusion Process 
    diffusion_process = GuidedDiffusionProcess(
        num_timesteps=cfg.num_diffusion_timesteps, 
        num_sampling_timesteps=cfg.num_diffusion_timesteps
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Best Loss
    best_eval_loss = float('inf')
    
    print('\n----------------------------------')
    print(f'\033[93mEpoch  Train-Loss   Accuracy\033[0m')
    print('----------------------------------')
    
    # Train
    for epoch in range(cfg.num_epochs//2):
        
        # For Loss Tracking
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Set model to train mode
        model.train()
        
        # Loop over dataloader
        for (knots, labels) in (pbar:=tqdm(knot_train_dl)):
            
            knots = knots.to(device)
            labels = labels.to(device)
            
            # Add noise
            noise = torch.randn_like(knots).to(device)
            t = torch.randint(0, cfg.num_diffusion_timesteps, (knots.shape[0],)).to(device)
            noisy_knots = diffusion_process.add_noise(knots, noise, t)          
            
            # Avoid Gradient Accumulation
            optimizer.zero_grad()
            
            # Prediction
            outputs = model(noisy_knots, t)
            loss = criterion(outputs, labels)
            
            # Backprop + Update model params
            loss.backward()
            optimizer.step()
            
            # Loss Tracking
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_description(f'Loss: {loss.item():.4f}, Acc: {correct/total:.4f}')
        # Mean Losses and Acc
        train_loss = running_loss / len(knot_train_dl)
        accuracy = correct / total
        
        # Display
        print(f'{epoch+1:<6} {train_loss:<10.4f}  {accuracy:<10.4f}', end = "   \n")
        
        # Save based on train-loss
        if train_loss < best_eval_loss:
            best_eval_loss = train_loss
            torch.save(model, cfg.class_ckpt_path)
        scheduler.step()    
    print('----------------------------------')
    
    # Memory Management
    del model, knots, noisy_knots, labels, diffusion_process
    gc.collect()
    torch.cuda.empty_cache()


# TRAIN
train_classifier(cfg.CONFIG)