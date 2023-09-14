#look at deviations across different ADNI cohorts for different models

import pandas as pd 
from os.path import join 
import numpy as np
import random
import torch
from scipy.stats import chi2, linregress
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import numpy as np
import seaborn as sns
from statannotations.Annotator import Annotator
from itertools import combinations

def latent_deviations_mahalanobis_across(cohort, train):
    dists = calc_mahalanobis_distance(cohort[0], train[0])
    return dists

def calc_mahalanobis_distance(values, train_values):
    covariance  = np.cov(train_values, rowvar=False)
    covariance_pm1 = np.linalg.matrix_power(covariance, -1)
    centerpoint = np.mean(train_values, axis=0)
    dists = np.zeros((values.shape[0]))
    for i in range(0, values.shape[0]):
        p0 = values[i,:]
        dist = (p0-centerpoint).T.dot(covariance_pm1).dot(p0-centerpoint)
        dists[i] = dist
    return dists

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
'mVAE_t1': './results/mVAE_t1/{0}'.format(date),
'mVAE': './results/mVAE/{0}'.format(date),
'weighted_mVAE': './results/weighted_mVAE/{0}'.format(date),
}

OUT_PATH = './results/ADNI_finetuned'
os.makedirs(OUT_PATH, exist_ok=True)

colour_dict = {'mVAE': 'b',
'weighted_mVAE': 'c',
'mVAE_t1': 'm',}

row_order = ['SMC', 'EMCI', 'LMCI', 'AD']
ind = np.arange(len(row_order)) 
bar_width = 0.1
bps = []
bp_labels = []
i =0
fig = plt.figure(figsize=(18,4))  
for key, val in model_dict.items():
    NEW_PATH = join(val, "ADNI_finetuned")
    model = torch.load(join(NEW_PATH, "model.pkl"))

    if key == 'mVAE_concat':
        train_latents = model.predict_latents(transfer)
        test_latents = model.predict_latents(test)
        dists_test = latent_deviations_mahalanobis_across(test_latents, train_latents)
    elif key == 'mVAE_t1':
        train_latents = model.predict_latents(mri_transfer)
        test_latents = model.predict_latents(mri_test)
        dists_test = latent_deviations_mahalanobis_across(test_latents, train_latents)
    else:
        train_latents = model.predict_latents(mri_transfer, dti_transfer)
        test_latents = model.predict_latents(mri_test, dti_test)
        dists_test = latent_deviations_mahalanobis_across(test_latents, train_latents)
    
    ax = fig.add_subplot(1,3,i+1)
    df = test_covariates.copy()
    df['dev'] = dists_test
    ax = sns.boxplot(data=df, x='DX_bl', y='dev', order=row_order)
    annotator = Annotator(ax, list(combinations(row_order, 2)), data=df, x='DX_bl', y='dev', order=row_order)
    annotator.configure(test='t-test_welch', text_format='star', loc='inside')
    annotator.apply_and_annotate()
    df['DX_bl'].replace(['SMC', 'EMCI', 'LMCI', 'AD'],
                        [0, 1, 2, 3], inplace=True)
    slope, intercept, r, p, sterr = linregress(x=df['DX_bl'],
                                                        y=df['dev'])

    ax.set_xticklabels(row_order)
    if key == 'mVAE':
        ax.set_ylabel(r'$D_{ml}$')
        ax.set_title('PoE-normVAE (baseline): slope = {0}'.format(round(slope,3)))
    elif key == 'mVAE_concat':
        ax.set_ylabel(r'$D_{ml}$')
        ax.set_title('concatenated normVAE (baseline): slope = {0}'.format(round(slope, 3)))
    elif key == 'mVAE_t1':
        ax.set_ylabel(r'$D_{ml}$')
        ax.set_title('T1 normVAE (baseline): slope = {0}'.format(round(slope, 3)))
    else:
        ax.set_ylabel(r'$D_{ml}$')
        ax.set_title('gPoE-normVAE (ours): slope = {0}'.format(round(slope, 3)))
    ax.set_xlabel('Disease label')
    ax.set_ylim(0, 250)
    i+=1
plt.tight_layout()
plt.savefig(join(OUT_PATH, 'Baseline_vs_proposed_deviations.png'.format(key)))
plt.close()
