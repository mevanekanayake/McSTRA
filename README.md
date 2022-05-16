# Multi-head Cascaded Swin Transformers (McSTRA)
This is the official Pytorch implementation of the Multi-head Cascaded Swin Transformers (McSTRA)

![](img/model.png?raw=true)

## Environment
Prepare a python=3.8 environment and install the following python libraries.
1. xxx
2. xxx
3. xxx
4. x
5. x
6. x
7. x
8. x
9. x
10. x


Alternatively, you can run the command "pip install -r requirements.txt" to install all the required libraries directly.

## Data Preparation
- For experiments we used the fastMRI knee and brain datasets which are publicly available at: https://fastmri.med.nyu.edu/


## File structure

```
 data
  ├── nuclei
  |   ├── train
  │   │   ├── image
  │   │   │   └── 00ae65...
  │   │   └── mask
  │   │       └── 00ae65...       
  ├── spleen
  ├── heart
  │   
  |
 Duo-SegNet
  ├──train.py
...
```

## Train McSTRA



## Test McSTRA


## Some Results


## Acknowledgements
- This repository makes use of the fastMRI codebase for training, evaluation, and data preparation: https://github.com/facebookresearch/fastMRI
- For the implementation of Swin-Unet and its constituents, the official implementation of Swin-Unet is utilized: https://github.com/HuCaoFighting/Swin-Unet


## Citing Our Work
Please cite: