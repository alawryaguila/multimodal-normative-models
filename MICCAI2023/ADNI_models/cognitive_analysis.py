#look at correlations between deviation measures and cognitive scores

import pandas as pd 
from os.path import join 
import numpy as np
import random
import torch
from sklearn.covariance import MinCovDet
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from decimal import Decimal

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

def calc_robust_mahalanobis_distance(values, train_values):
    # fit a MCD robust estimator to data
    robust_cov = MinCovDet(random_state=42).fit(train_values)
    mahal_robust_cov = robust_cov.mahalanobis(values)
    return mahal_robust_cov

def data_deviations_zscores(cohort_recon, train_recon):
    cohort_recon = np.mean(cohort_recon, axis=1)
    train_recon = np.mean(train_recon, axis=1)
    mean_train = np.mean(train_recon, axis=0)
    sd_train = np.std(train_recon, axis=0)
    z_scores = (cohort_recon - mean_train)/sd_train
    return z_scores

def deviation(orig, recon):
    return np.sqrt((orig - recon)**2)

DATA_PATH = '../../ADNI_data/splines_regression'

mri_data = pd.read_csv(join(DATA_PATH, 't1/230224/t1_data_adjusted.csv'), header=0)
dti_data = pd.read_csv(join(DATA_PATH, 'dti/230224/dti_data_adjusted.csv'), header=0)

non_data_cols = ['FID', 'IID', 'RID', 'iAGE', 'ICV', 'DX_bl', 'PTGENDER'] 
covariates = mri_data[non_data_cols]
mri_data = mri_data[mri_data.columns.drop(non_data_cols)]
dti_data = dti_data[dti_data.columns.drop(non_data_cols)]

#process dti data
to_drop_dti = ['FA.FX', 'MD.FX']
dti_data = dti_data[dti_data.columns.drop(to_drop_dti)]

mri_transfer = mri_data[covariates.DX_bl == 'CN']
dti_transfer = dti_data[covariates.DX_bl == 'CN']
mri_test = mri_data[~(covariates.DX_bl == 'CN')]
dti_test = dti_data[~(covariates.DX_bl == 'CN')]

test_covariates = covariates[~(covariates.DX_bl == 'CN')]
test_covariates.reset_index(drop=True, inplace=True)

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

#load cognitive data
COG_PATH = '../../ADNI_data'

cog_data = pd.read_csv(join(COG_PATH, 'UWNPSYCHSUM_01_23_23.csv'), header=0)
cog_data = cog_data[cog_data['VISCODE2'] == 'bl']

cog_data = pd.merge(cog_data, covariates, on='RID', how='inner')

#regress out effect of age from cognitive scores
x = cog_data['iAGE'].to_numpy().reshape(-1,1)
for col in ['ADNI_MEM', 'ADNI_EF']:
    lin_reg = LinearRegression()
    y = cog_data[col].to_numpy().reshape(-1,1)
    lin_reg.fit(x,y)
    y_pred = lin_reg.predict(x)
    resid = y - y_pred
    cog_data['{0}_reg'.format(col)] = np.squeeze(resid)


cog_data_test = cog_data[cog_data['RID'].isin(test_covariates.RID)]

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
        dists_test = latent_deviations_mahalanobis_across(test_latents, train_latents, key)
    elif key == 'mVAE_t1':
        train_latents = model.predict_latents(mri_transfer)
        test_latents = model.predict_latents(mri_test)
        dists_test = latent_deviations_mahalanobis_across(test_latents, train_latents)
    else:
        train_latents = model.predict_latents(mri_transfer, dti_transfer)
        test_latents = model.predict_latents(mri_test, dti_test)
        dists_test = latent_deviations_mahalanobis_across(test_latents, train_latents)
    
    ax = fig.add_subplot(1,3,i+1)
    df = cog_data_test.copy()
    df['dev'] = dists_test

    #col = 'ADNI_MEM_reg'
    col = 'ADNI_EF_reg'
    df = df[df[col].notna()]
    df[col] = df[col].astype(float)
    ax = sns.scatterplot(x=df[col], y=df['dev'])
    ax = sns.regplot(x=df[col], y=df['dev'])
    slope, intercept, r, p, sterr = linregress(x=df[col],
                                                        y=df['dev'])
    print(key, slope, r, p)
    if key == 'mVAE':
        ax.set_ylabel(r'$D_{ml}$')
        ax.set_title('PoE-normVAE (baseline): r = {0}, p = {1}'.format(round(r,3), '{:.2E}'.format(Decimal(p))))
    elif key == 'weighted_mVAE':
        ax.set_ylabel(r'$D_{ml}$')
        ax.set_title('gPoE-normVAE (ours): r = {0}, p = {1}'.format(round(r,3), '{:.2E}'.format(Decimal(p))))
    elif key == 'mVAE_t1':
        ax.set_ylabel(r'$D_{ml}$')
        ax.set_title('t1 normVAE (baseline): r = {0}, p = {1}'.format(round(r,3), '{:.2E}'.format(Decimal(p))))
    #ax.set_xlabel('Memory score')
    ax.set_xlabel('Executive function score')

    i+=1
plt.tight_layout()
plt.savefig(join(OUT_PATH, 'Baseline_vs_proposed_EF_correlations_adjustedforage.png'.format(key)))
#plt.savefig(join(OUT_PATH, 'Baseline_vs_proposed_MEM_correlations_adjustedforage.png'.format(key)))
plt.close()
