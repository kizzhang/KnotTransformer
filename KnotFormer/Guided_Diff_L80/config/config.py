
class CONFIG:
    
    # Data path
    data_dir = '/home/zzhang/LP_knot_id/L80'
    
    # Checkpoint saving path
    ckpt_path = '/home/zzhang/KnotFormer/Guided_Diff_L80/checkpt/'

    cls_name = 'cls_ckpt.pth'
    dif_name = 'dif_ckpt.pth'

    class_ckpt_path = ckpt_path+cls_name
    dif_cktp_path = ckpt_path+dif_name
    

    # Training Hyperparams
    device = 0
    num_epochs = 100
    lr = 2e-6
    data_drop = 0.
    trans_drop = 0.2
    num_diffusion_timesteps = 1000
    batch_size = 128
    input_dim = 3
    max_length= 80 
    dmodel = 512
    nheads = 8
    depth = 5
    num_classes = 8
    train_ratio = 0.9
    train_cls_from_ckpt = True
    train_dif_from_ckpt = True

    # Sampling Hyperparams
    atom_num = 80
    num_sampling_timesteps = 1000
    classifier_guidance = True
    classifier_scale = 1.0