import gc
import torch
from gaussian.gaussian import GuidedDiffusionProcess
from tqdm import tqdm
import numpy as np

def generate(cfg, y):
    """
    Given Pretrained Transformer model and labels, Generate corresponding
    knots conditioned on label y from noise by going backward step by step. i.e.,
    Mapping of Random Noise to real knots.
    """
    
    # Device
    device = torch.device(f'cuda:{cfg.device}')
    #print(f'Device: {device}\n')
    
    # Initialize Diffusion Reverse Process
    diffusion_process = GuidedDiffusionProcess(
        num_timesteps=cfg.num_diffusion_timesteps, 
        num_sampling_timesteps=cfg.num_sampling_timesteps,
        classifier_guidance = cfg.classifier_guidance,
        classifier_scale = cfg.classifier_scale
    )
    
    # Classifier Guidance
    classifier = None
    if cfg.classifier_guidance:
        # Load Classifier Model
        classifier = torch.load(cfg.class_ckpt_path, map_location="cpu").to(device)
        classifier.eval()
    
    # Set model to eval mode
    model = torch.load(cfg.dif_cktp_path, map_location="cpu").to(device)
    model.eval()
    
    # Generate Noise sample from N(0, 1)
    xt = torch.randn(1, cfg.atom_num, 3).to(device)
    
    # Denoise step by step by going backward.
    num_sampling_timesteps = len(diffusion_process.use_timesteps)
    with torch.no_grad():
        for t in tqdm(reversed(range(num_sampling_timesteps))):    
            xt = diffusion_process.p_sample(model, 
                                            xt, 
                                            torch.as_tensor(t).unsqueeze(0).to(device), 
                                            torch.as_tensor(y).unsqueeze(0).to(device),
                                            classifier
                                           )['sample']

    xt = xt.detach().cpu()
    xt = np.cumsum(xt.numpy(), axis=1)
    
    # Memory Management
    del model, diffusion_process
    gc.collect()
    torch.cuda.empty_cache()
    
    return xt

from data.data import LABEL
import os 
opj=os.path.join    

def generate_process(cfg, y, path,time_step):
    """
    Given Pretrained Transformer model and labels, Generate corresponding
    knots conditioned on label y from noise by going backward step by step. i.e.,
    Mapping of Random Noise to real knots.
    """
    
    # Device
    device = torch.device(f'cuda:{cfg.device}')
    #print(f'Device: {device}\n')
    
    # Initialize Diffusion Reverse Process
    diffusion_process = GuidedDiffusionProcess(
        num_timesteps=cfg.num_diffusion_timesteps, 
        num_sampling_timesteps=cfg.num_sampling_timesteps,
        classifier_guidance = cfg.classifier_guidance,
        classifier_scale = cfg.classifier_scale
    )
    
    # Classifier Guidance
    classifier = None
    if cfg.classifier_guidance:
        # Load Classifier Model
        classifier = torch.load(cfg.class_ckpt_path, map_location="cpu").to(device)
        classifier.eval()
    
    # Set model to eval mode
    model = torch.load(cfg.dif_cktp_path, map_location="cpu").to(device)
    model.eval()
    
    # Generate Noise sample from N(0, 1)
    xt = torch.randn(1, cfg.atom_num, 3).to(device)
    
    # Denoise step by step by going backward.
    num_sampling_timesteps = len(diffusion_process.use_timesteps)
    with torch.no_grad():
        f = open(opj(path,f"gen_process_Knot{LABEL[y]}.txt"), "w")
        for t in tqdm(reversed(range(num_sampling_timesteps))):    
            xt = diffusion_process.p_sample(model, 
                                            xt, 
                                            torch.as_tensor(t).unsqueeze(0).to(device), 
                                            torch.as_tensor(y).unsqueeze(0).to(device),
                                            classifier
                                           )['sample']
            if t % time_step == 0:
                save_xt = xt.detach().cpu()
                save_xt = np.cumsum(save_xt.numpy(), axis=1)
                f.write('80\n')
                f.write(f'{LABEL[y]}\n')
                for atom in save_xt[0]:
                    f.write(f'1 {atom[0]} {atom[1]} {atom[2]}\n')
            if t == time_step - 1:
                save_xt = xt.detach().cpu()
                save_xt = np.cumsum(save_xt.numpy(), axis=1)
                f.write('80\n')
                f.write(f'{LABEL[y]}\n')
                for atom in save_xt[0]:
                    f.write(f'1 {atom[0]} {atom[1]} {atom[2]}\n')
        f.close()

    xt = xt.detach().cpu()
    xt = np.cumsum(xt.numpy(), axis=1)
    
    # Memory Management
    del model, diffusion_process
    gc.collect()
    torch.cuda.empty_cache()
    
    return xt
