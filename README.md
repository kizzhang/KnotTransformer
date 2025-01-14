# KnotTransformer
## A repository for our classification and generation models

### Classification
TransKnot is a Transformer-based model that classifies knots of different types. 

This repository contains the training codes: 

- `TransKnot_xyzbond.py` is the training code for $N=300$ flexible polymer knots;

- `large_model_xyzbond.py` is the training code for the large model.

Our model architecture schematics:

![TransKnot model architecture](https://github.com/kizzhang/KnotTransformer/blob/main/assets/imgs/TransKnot.png))

### KnotFormer
KnotFormer is a Transformer-based diffusion model that generates accurate knots of different types

- To train the diffusion model: `python diffusion_train.py`

- To train the classifier: `python classifier_train.py`

- To generate: `python generate.py`

Our model architecture schematics:
![TransKnot model architecture](https://github.com/kizzhang/KnotTransformer/blob/main/assets/imgs/KnotFormer.png)
