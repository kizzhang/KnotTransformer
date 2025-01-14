import math
import numpy as np

def sample_timesteps(num_total_timesteps: int = 1000,
                     num_sample_timesteps: int = 1000
                    ):
    """
    Uniformaly sample num_sample_timesteps timesteps given 
    total number of timesteps.
    Later we will utilize for speed up sampling.
    """
    # Get New sampled timesteps (Uniform stepping)
    step_size = num_total_timesteps//num_sample_timesteps
    use_timesteps = np.arange(0, num_total_timesteps, step_size)
    
    # Make sure to append final timestep if not there already
    if (num_total_timesteps - 1) not in use_timesteps:
        use_timesteps = np.append(use_timesteps, num_total_timesteps - 1)
        
    return use_timesteps

#---------------------------------------------------------------------------


def cosine_noise_schedular(num_original_timesteps: int = 1000, 
                           num_sampling_timesteps: int = 1000, 
                           max_beta: float = 0.999
                          ) -> np.ndarray:
    """
    Calculate noise controlling parameter betas for all time steps 
    according to the cosine noise scheduler.

    Parameters:
    num_original_timesteps (int): Number of original diffusion timesteps (used during training).
    num_sampling_timesteps (int): Number of timesteps used for sampling.
    max_beta (float): Maximum value for beta.

    Returns:
    array: Beta values for each timestep.
    array: timesteps used to calculate betas.
    
    NOTE: Set "num_sampling_timesteps" = "num_original_timesteps" during training.
    """
    # s is set to 0.008 by Authors
    s = 0.008
    
    # Calculate alpha bars
    alpha_bars = [
        math.cos(((t / num_original_timesteps + s) / (1 + s)) * (math.pi / 2)) ** 2 
        for t in range(num_original_timesteps + 1)
    ]
    
    # Calculate betas
    betas = np.array([
        min(1 - a_t / a_tminus1, max_beta) 
        for a_t, a_tminus1 in zip(alpha_bars[1:], alpha_bars[:-1])
    ])
    
    # Modify Betas as described in the "Improved Sampling Speed" section above.
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    
    # Get New sampled timesteps (Uniform stepping)
    use_timesteps = sample_timesteps(num_original_timesteps, 
                                     num_sampling_timesteps
                                    )

    # Calculate new betas according to new timesteps.
    last_alpha_cumprod = 1.0
    new_betas = []
    for i, alpha_cumprod in enumerate(alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
    
    return np.array(new_betas), use_timesteps

