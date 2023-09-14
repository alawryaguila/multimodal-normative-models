#create reconstruction plots for ADNI cohorts using gPoE-normVAE model

import pandas as pd 
from os.path import join 
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import numpy as np
import seaborn as sns
from statannotations.Annotator import Annotator
from itertools import combinations
from enigmatoolbox.plotting import plot_subcortical
from enigmatoolbox.utils.parcellation import parcel_to_surface
from enigmatoolbox.plotting import plot_cortical
from scipy import stats

def data_deviations_zscores(cohort_recon, train_recon):
    mean_train = np.mean(train_recon, axis=0)
    sd_train = np.std(train_recon, axis=0)
    z_scores = (cohort_recon - mean_train)/sd_train
    return z_scores

def deviation(orig, recon):
    return np.sqrt((orig - recon)**2)

def plotCort(data, cort_file, lim=2.2, color_map=None):
    if color_map is None:
        color_map = 'bwr'  
    MRI_dict = pd.read_csv(cort_file, header=0)
    enigma_list = list(MRI_dict['Enigma'])
    data.columns = data.columns.str.replace(' ', '_')
    data.columns = data.columns.str.replace('ctx.rh.', 'R_')
    data.columns = data.columns.str.replace('ctx.lh.', 'L_')
    data = data.add_suffix('_thickavg')
    if isinstance(lim, list):
        extra_cols = pd.DataFrame(np.zeros((np.shape(data)[0], 2)), columns = ['L_temporalpole_thickavg','R_temporalpole_thickavg'])
        data = pd.concat([data,extra_cols], axis=1)
        data = data[enigma_list]
        data = data.to_numpy()

        data = parcel_to_surface(data, 'aparc_fsa5')
        fig = plot_cortical(array_name=data, surface_name="fsa5", size=(1000, 400),
            cmap=color_map, color_bar=True, color_range=(lim[0], lim[1]), return_plotter=True)
    else:
        extra_cols = pd.DataFrame(np.zeros((np.shape(data)[0], 2)), columns = ['L_temporalpole_thickavg','R_temporalpole_thickavg'])
        data = pd.concat([data,extra_cols], axis=1)
        data = data[enigma_list]
        data = data.to_numpy()

        data = parcel_to_surface(data, 'aparc_fsa5')
        fig = plot_cortical(array_name=data, surface_name="fsa5", size=(1000, 400),
                    cmap=color_map, color_bar=True, color_range=(-lim, lim), return_plotter=True)
    return fig

def plotSubcort(data, subcort_file, lim=2.2, color_map=None):  
    subcort_file = subcort_file        
    if color_map is None:
        color_map = 'bwr'
    MRI_dict = pd.read_csv(subcort_file, header=0)
    enigma_list = list(MRI_dict['Enigma'])
    UKBB_list = list(MRI_dict['UKBio'])
    for col in data.columns:
        if col in UKBB_list:
            data = data.rename(columns={col:enigma_list[UKBB_list.index(col)]})
    data.columns = data.columns.str.replace(' ', '_')

    data = data[enigma_list]
    data = data.to_numpy()[0] 
    if isinstance(lim, list):
        fig = plot_subcortical(array_name=data, size=(1000, 400),
                        cmap=color_map, color_bar=True, color_range=(lim[0],lim[1]), return_plotter=True)
    else:
        fig = plot_subcortical(array_name=data, size=(1000, 400),
                        cmap=color_map, color_bar=True, color_range=(-lim,lim), return_plotter=True)
    return fig

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
transfer_covariates = covariates[covariates.DX_bl == 'CN']
test_covariates = covariates[~(covariates.DX_bl == 'CN')]
transfer_covariates.reset_index(drop=True, inplace=True)
test_covariates.reset_index(drop=True, inplace=True)

mri_cols = mri_data.columns
dti_cols = dti_data.columns
data_cols = mri_cols.append(dti_cols)

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
}


colour_dict = {'mVAE': 'b',
'weighted_mVAE': 'c',
'mmVAE': 'g',
'VAE_barlow2': 'r',
'mVAE_concat': 'y',
'mVAE_t1': 'm',
'mVAE_dti': 'k'}

OUT_PATH = './results/ADNI_finetuned/recon_results'
os.makedirs(OUT_PATH, exist_ok=True)

row_order = ['SMC', 'EMCI', 'LMCI', 'AD'] 

for key, val in model_dict.items():
    NEW_PATH = join(val, "ADNI_finetuned")
    model = torch.load(join(NEW_PATH, "model.pkl"))

    if key == 'mVAE_concat':
        train_recon = model.predict_reconstruction(transfer)
        test_recon = model.predict_reconstruction(test)
        train_dev = deviation(transfer, train_recon[0][0])
        test_dev = deviation(test, test_recon[0][0])
        dists_test = data_deviations_zscores(test_dev, train_dev)

    elif key == 'mVAE_t1':
        train_recon = model.predict_reconstruction(mri_transfer)
        test_recon = model.predict_reconstruction(mri_test)
        train_dev = deviation(mri_transfer, train_recon[0][0])
        test_dev = deviation(mri_test, test_recon[0][0])
        dists_test = data_deviations_zscores(test_dev, train_dev)
    else:
        train_recon = model.predict_reconstruction(mri_transfer, dti_transfer)
        train_recon = np.concatenate((train_recon[0][0], train_recon[0][1]), axis=1)
        train_dev = deviation(transfer, train_recon)
        test_recon = model.predict_reconstruction(mri_test, dti_test)
        test_recon = np.concatenate((test_recon[0][0], test_recon[0][1]), axis=1)
        test_dev = deviation(test, test_recon)
        dists_test = data_deviations_zscores(test_dev, train_dev)

    cort_map = '../../UKBB_data/Enigma_cortical_mapping.csv'
    subcort_map =  '../../UKBB_data/enigma_subcortical_mapping.csv'
    data = pd.DataFrame(dists_test, columns=data_cols)
    data['RID'] = test_covariates.RID
    data = data.merge(covariates, on='RID', how='inner')

    mri_data = data[mri_cols]
    dti_data = data[dti_cols]
    mri_max = 0
    mri_min = 0
    dti_max = 0
    dti_min = 0
    for row in row_order:
        data_ = data[data.DX_bl == row]
        mri_data_ = data_[mri_cols]
        dti_data_ = data_[dti_cols]
        mri_max_ = np.max(np.mean(mri_data_, axis=0))
        mri_min_ = np.min(np.mean(mri_data_, axis=0))
        dti_max_ = np.max(np.mean(dti_data_, axis=0))
        dti_min_ = np.min(np.mean(dti_data_, axis=0))
        if mri_max_ > mri_max:
            mri_max = mri_max_
        if mri_min_ < mri_min:
            mri_min = mri_min_
        if dti_max_ > dti_max:
            dti_max = dti_max_
        if dti_min_ < dti_min:
            dti_min = dti_min_
    mri_max = mri_max + 0.1
    mri_min = mri_min - 0.1
    dti_max = dti_max + 0.1
    dti_min = dti_min - 0.1
    for row in row_order:
        data_ = data[data.DX_bl == row]
        mri_data_ = pd.DataFrame(np.mean(data_[mri_cols],axis=0).values.reshape(1,-1), columns=mri_cols)
        dti_data_ = pd.DataFrame(np.mean(data_[dti_cols],axis=0).values.reshape(1,-1), columns=dti_cols)

        data_max = np.max(np.mean(mri_data, axis=0))
        fig = plotCort(mri_data_, cort_map, lim=[mri_min, mri_max], color_map='Reds')
        fig._to_image(filename=join(OUT_PATH, 'mean_{0}_cortical_{1}.png'.format(row, key)), transparent_bg=True, scale=(1, 1))
        fig = plotSubcort(mri_data_, subcort_map, lim=[mri_min, mri_max], color_map='Reds')
        fig._to_image(filename=join(OUT_PATH, 'mean_{0}_subcortical_{1}.png'.format(row, key)), transparent_bg=True, scale=(1, 1))

        fig = plt.figure(figsize=(18, 5))
        dti_data_fa = dti_data_.filter(regex='FA')
        dti_data_md = dti_data_.filter(regex='MD')

        ax = fig.add_subplot(1,2,1)
        ax.bar(range(0,dti_data_fa.shape[1]), dti_data_fa.to_numpy().tolist()[0], color='r', label='FA')
        ax.set_xticks(range(0,dti_data_fa.shape[1]))
        ax.set_xticklabels(dti_data_fa.columns, rotation=90)
        ax.set_ylim([dti_min, dti_max])
        ax.set_title('DTI FA')

        ax = fig.add_subplot(1,2,2)
        ax.bar(range(0,dti_data_md.shape[1]), dti_data_md.to_numpy().tolist()[0], color='g', label='MD')
        ax.set_xticks(range(0,dti_data_md.shape[1]))
        ax.set_xticklabels(dti_data_md.columns, rotation=90)
        ax.set_ylim([dti_min, dti_max])
        ax.set_title('DTI MD')
        plt.tight_layout()
        plt.savefig(join(OUT_PATH, 'mean_{0}_DTI_{1}.png'.format(row, key)))

