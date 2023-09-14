# Multi-modal normative modelling using mVAE

Code for the paper "Multi-modal normative modelling for abnormality detection across multiple imaging modalities".

## Implementation details

All models were implemented using the ``multi-view-AE`` package (https://multi-view-ae.readthedocs.io/en/latest/).

All models were trained on a machine with Intel i7-10750H CPU and 1 Intel UHD Graphics GPU.

## Installation and running scripts

### Installation
Clone repository and move to folder:
```bash
git clone https://github.com/alawryaguila/multi-view-AE
cd multi-view-AE
```

Create the customised python environment:
```bash
conda create --name MICCAI23 python=3.9
```

Activate python environment:
```bash
conda activate MICCAI23
```

Install the package:
```bash
pip install -e ./
```

Example of how to run scripts:
```bash
python UKBB_train_models/train_multimodal_unimodal_models.py 
```

### Training script
The ```UKBB_train_models/train_multimodal_unimodal_models.py```  script within the ```UKBB_train_models``` folder is used to train the normVAE models on the UK Biobank data. 

### UK Biobank test scripts
The scripts to create the results for the UK Biobank dataset are stored in the  ```UKBB_results``` folder.

### ADNI test scripts
The scripts to create the results for the ADNI dataset are stored in the  ```ADNI_models``` folder. This includes the ```transfer_learning.py``` script for fine-tuning of the UK Biobank trained models on the ADNI dataset.

## Trained models
Trained models (including training parameters) are stored in the ```trained_models``` folder.

