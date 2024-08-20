# KnotTransformer
## A repository for the TransKnot and KnotFormer models

### TransKnot
TransKnot is a Transformer-based model that classify knots of different types. 

This repository contains the training codes: 

- `TransKnot_xyzbond.py` is the training code for $N=300$ flexible polymer knots;

- `large_model_xyzbond.py` is the training code for the large model.

Our model architecture loos like this:
![TransKnot model architecture](https://github.com/kizzhang/KnotTransformer/blob/main/assets/imgs/TransKnot.png))

### KnotFormer
KnotFormer is a Transfomer-based classifier-free diffusion model that generates accurate knots of different types

This repository contains the both the code for training and generating: 

- `Bond_dffusion.py` is the training code for $N=300$ semi-flexible polymer knots;

- `Bond_dffusion_generate.py` is the code for generation of polymer knots.

Our model architecture loos like this:
![TransKnot model architecture](https://github.com/kizzhang/KnotTransformer/blob/main/assets/imgs/KnotFormer.png)
