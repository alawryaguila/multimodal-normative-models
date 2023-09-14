#fine-tine models trained on ukbb on adni data

import pandas as pd 
from os.path import join 
import numpy as np
import torch
import os
import numpy as np

DATA_PATH = '../../ADNI_data/splines_regression'

mri_data = pd.read_csv(join(DATA_PATH, 't1/230224/t1_data_adjusted.csv'), header=0)

dti_data = pd.read_csv(join(DATA_PATH, 'dti/230224/dti_data_adjusted.csv'), header=0)

non_data_cols = ['FID', 'IID', 'RID', 'iAGE', 'ICV', 'DX_bl', 'PTGENDER'] 
covariates = mri_data[non_data_cols]
mri_data = mri_data[mri_data.columns.drop(non_data_cols)]
dti_data = dti_data[dti_data.columns.drop(non_data_cols)]

to_drop_dti = ['FA.FX', 'MD.FX']
dti_data = dti_data[dti_data.columns.drop(to_drop_dti)]

mri_transfer = mri_data[covariates.DX_bl == 'CN']
dti_transfer = dti_data[covariates.DX_bl == 'CN']
mri_test = mri_data[~(covariates.DX_bl == 'CN')]
dti_test = dti_data[~(covariates.DX_bl == 'CN')]
transfer_covariates = covariates[covariates.DX_bl == 'CN']
test_covariates = covariates[~(covariates.DX_bl == 'CN')]

#scale and centre data ADNI data
mean_controls = mri_transfer.to_numpy().mean(axis=0)
sd_controls = mri_transfer.to_numpy().std(axis=0)

mri_transfer = (mri_transfer - mean_controls)/sd_controls
mri_test = (mri_test - mean_controls)/sd_controls

mean_controls = np.mean(dti_transfer, axis=0)
sd_controls = np.std(dti_transfer, axis=0)

dti_transfer = (dti_transfer - mean_controls)/sd_controls
dti_test = (dti_test - mean_controls)/sd_controls

dti_test = dti_test.to_numpy()
mri_test = mri_test.to_numpy()
mri_transfer = mri_transfer.to_numpy()
dti_transfer = dti_transfer.to_numpy()

transfer = np.concatenate((mri_transfer, dti_transfer), axis=1)
test = np.concatenate((mri_test, dti_test), axis=1)

#load models and predict latents for each cohort 
date = 'date/of/model/training'
model_dict = {
'weighted_mVAE': './results/weighted_mVAE/{0}'.format(date),
'mVAE': './results/mVAE/{0}'.format(date),
'mmVAE': './results/mmVAE/{0}'.format(date),
'mVAE_concat': './results/mVAE_concat/{0}'.format(date),
'mVAE_t1': './results/mVAE_t1/{0}'.format(date),
'mVAE_dti': './results/mVAE_dti/{0}'.format(date),
}

#model parameters
max_epochs = 100
batch_size = 20

for key, val in model_dict.items():
    model = torch.load(join(val, "model.pkl"))
    NEW_PATH = join(val, "ADNI_finetuned")
    model.cfg.out_dir = NEW_PATH
    if key  == 'mVAE_concat':
        model.fit(transfer, max_epochs=max_epochs, batch_size=batch_size)
    elif key == 'mVAE_t1':
        model.fit(mri_transfer, max_epochs=max_epochs, batch_size=batch_size)
    elif key == 'mVAE_dti':
        model.fit(dti_transfer, max_epochs=max_epochs, batch_size=batch_size)
    else:
        model.fit(mri_transfer, dti_transfer, max_epochs=max_epochs, batch_size=batch_size)
