# Multi-head Cascaded Swin Transformers (McSTRA)
```
This is the official Pytorch implementation of the Multi-head Cascaded Swin Transformers (McSTRA)
```
![](img/model.png?raw=true)

## Environment
```
Prepare a python=3.8 environment and install the following python libraries.
1. pytorch==1.10.0
2. torchvision==0.11.1
3. numpy
4. h5py
5. pandas
6. scikit-image
7. tqdm
8. einops
9. timm

Alternatively, you can run the command "pip install -r requirements.txt" to install all the required libraries directly.
```

## Data Preparation
```
- For experiments we used the fastMRI knee and brain datasets (https://fastmri.med.nyu.edu/).

- The intensity values of the data were scaled in range [0,1] as preprocessing.
```


## File structure

```
 McSTRA
  ├── train_mcstra.py
  ├── models
  │     └── mcstra.py
  ├── utils
  │     ├── data.py
  │     ├── evaluate.py
  │     ├── fourier.py
  │     ├── manager.py
  │     ├── mask.py
  │     ├── math.py
  │     ├── mulitcoil.py
  │     └── transforms.py
  ├── paths.json
  ├── requirements.txt
  ├── data
  │     ├── knee
  │     └── brain
  ├── experiments
  │     ├── Experiment#1
  │     ├── Experiment#2
  │     :
  :     
    
```


## Train McSTRA

```
python train_mcstra.py --batch_size=8 --num_epochs=50 --tvsr=1. --vvsr=1. --num_hts=2 --embed_dims=48,96,48
```

## Pretrained Models


## Some Results


## Acknowledgements
- This repository makes use of the fastMRI codebase for training, evaluation, and data preparation: https://github.com/facebookresearch/fastMRI
- For the implementation of Swin-Unet and its constituents, the official implementation of Swin-Unet is utilized: https://github.com/HuCaoFighting/Swin-Unet


## Citing Our Work
Please cite: