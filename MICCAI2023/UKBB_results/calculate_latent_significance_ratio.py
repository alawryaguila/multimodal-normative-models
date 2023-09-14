# create significance ratio for latent deviations for UKBB disease cohort and holdout cohort

import pandas as pd 
from os.path import join 
import numpy as np
import random
import torch
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
import numpy as np

def latent_deviations_mahalanobis_across_sig(cohort, train):
    latent_dim = cohort[0].shape[1]
    dists = calc_robust_mahalanobis_distance(cohort[0], train[0])
    pvals = 1 - chi2.cdf(dists, latent_dim - 1)
    return pvals

def calc_mahalanobis_distance(values, train_values):
    covariance  = np.cov(train_values, rowvar=False)
    covariance_pm1 = np.linalg.matrix_power(covariance, -1)
    centerpoint = np.mean(train_values, axis=0)
    dists = np.zeros((values.shape[0],1))
    for i in range(0, values.shape[0]):
        p0 = values[i,:]
        dist = (p0-centerpoint).T.dot(covariance_pm1).dot(p0-centerpoint)
        dists[i,:] = dist
    return dists

def calc_robust_mahalanobis_distance(values, train_values):
    # fit a MCD robust estimator to data
    robust_cov = MinCovDet(random_state=42).fit(train_values)
    mahal_robust_cov = robust_cov.mahalanobis(values)
    return mahal_robust_cov

def latent_count_ratio(pvals_cohort, pvals_holdout, model_type, cols):
    thresh = 0.001
    count_cohort = (pvals_cohort <= thresh).sum()
    count_holdout = (pvals_holdout <= thresh).sum()
    ratio_cohort = count_cohort/pvals_cohort.shape[0]
    ratio_holdout = count_holdout/pvals_holdout.shape[0]
    ratio = ratio_cohort/ratio_holdout
    df = pd.DataFrame(np.array([key, count_cohort, ratio_cohort, count_holdout, ratio_holdout, ratio]).reshape(1,-1),
    columns=cols)
    return df

DATA_PATH = '../../UKBB_data/splines_regression'

mri_data = pd.read_csv(join(DATA_PATH, 't1/230215/t1_data_adjusted.csv'), header=0)
dti_data = pd.read_csv(join(DATA_PATH, 'dti/230215/dti_data_adjusted.csv'), header=0)

#process dti data
to_drop_dti = ['FA.FX', 'MD.FX']

dti_data = dti_data[dti_data.columns.drop(to_drop_dti)]


#process t1 data
mri_data = mri_data.groupby('SEX').apply(lambda x: x.sample(n=mri_data[mri_data['SEX'] == 1].shape[0], random_state=42)) #select equal numbers of male and female


#separate out controls and cases data
mri_train = mri_data[mri_data.Dx == 0]
mri_test = mri_data[mri_data.Dx != 0]

dti_train = dti_data[dti_data.eid.isin(mri_train.eid)]
dti_test = dti_data[dti_data.eid.isin(mri_test.eid)]

mri_train = mri_train.set_index('eid')
mri_train = mri_train.reindex(index=dti_train['eid'])
mri_train = mri_train.reset_index()

mri_test = mri_test.set_index('eid')
mri_test = mri_test.reindex(index=dti_test['eid'])
mri_test = mri_test.reset_index()

#remove non imaging features
non_data_cols = ['eid', 'AGE', 'SEX', 'ICV', 'eid.1', 'SDx', 'Dx']

train_covariates = mri_train[non_data_cols]
mri_train = mri_train[mri_train.columns.drop(non_data_cols)]

dti_train = dti_train[dti_train.columns.drop(['eid'])]

mri_test = mri_test[mri_test.columns.drop(non_data_cols)]
dti_test = dti_test[dti_test.columns.drop(['eid'])]

#scale and centre data
mean_controls = np.mean(mri_train, axis=0)
sd_controls = np.std(mri_train, axis=0)

mri_train = (mri_train - mean_controls)/sd_controls
mri_test = (mri_test - mean_controls)/sd_controls

mean_controls = np.mean(dti_train, axis=0)
sd_controls = np.std(dti_train, axis=0)
dti_train = (dti_train - mean_controls)/sd_controls
dti_test = (dti_test - mean_controls)/sd_controls

mri_train = mri_train.to_numpy()
dti_train = dti_train.to_numpy()

mri_test = mri_test.to_numpy()
dti_test = dti_test.to_numpy()

#holdout 20% of controls for testing purposes
random.seed(42)
subj = list(range(0,mri_train.shape[0]))
holdout_subset = random.sample(subj, int(mri_train.shape[0]*0.2))
train_subset = list(set(subj) - set(holdout_subset))


dti_holdout = dti_train[holdout_subset,:]
mri_holdout = mri_train[holdout_subset,:]

mri_train = mri_train[train_subset,:]
dti_train = dti_train[train_subset,:]

train = np.concatenate((mri_train, dti_train), axis=1)
holdout = np.concatenate((mri_holdout, dti_holdout), axis=1)
test = np.concatenate((mri_test, dti_test), axis=1)

#load models and predict latents for each cohort 

date = 'date/of/model/training'
model_dict = {'mVAE': './results/mVAE/{0}'.format(date),
'weighted_mVAE': './results/weighted_mVAE/{0}'.format(date),
'mmVAE': './results/mmVAE/{0}'.format(date),
'mVAE_concat': './results/mVAE_concat/{0}'.format(date),
'mVAE_t1': './results/mVAE_t1/{0}'.format(date),
'mVAE_dti': './results/mVAE_dti/{0}'.format(date),}

cols = ['model', 'count cohort', 'ratio cohort', 'count holdout', 'ratio holdout', 'significance ratio']
results_df = pd.DataFrame(columns=cols)
for key, val in model_dict.items():
    model = torch.load(join(val, 'model.pkl'))
    if key == 'mVAE_concat':
        train_latents = model.predict_latents(train)
        holdout_latents = model.predict_latents(holdout)
        test_latents = model.predict_latents(test)
    elif key == 'mVAE_t1':
        train_latents = model.predict_latents(mri_train)
        holdout_latents = model.predict_latents(mri_holdout)
        test_latents = model.predict_latents(mri_test)

    elif key == 'mVAE_dti':
        train_latents = model.predict_latents(dti_train)
        holdout_latents = model.predict_latents(dti_holdout)
        test_latents = model.predict_latents(dti_test)
    elif key == 'mmVAE':
        train_latents = model.predict_latents(mri_train, dti_train)
        train_latents = [np.mean([train_latents[0], train_latents[1]], axis=0)]
        holdout_latents = model.predict_latents(mri_holdout, dti_holdout)
        holdout_latents = [np.mean([holdout_latents[0], holdout_latents[1]], axis=0)]
        test_latents = model.predict_latents(mri_test, dti_test)
        test_latents = [np.mean([test_latents[0], test_latents[1]], axis=0)]
    else:
        train_latents = model.predict_latents(mri_train, dti_train)
        holdout_latents = model.predict_latents(mri_holdout, dti_holdout)
        test_latents = model.predict_latents(mri_test, dti_test)

    pvals_holdout = latent_deviations_mahalanobis_across_sig(holdout_latents, train_latents)
    pvals_test = latent_deviations_mahalanobis_across_sig(test_latents, train_latents)
    results = latent_count_ratio(pvals_test, pvals_holdout, key, cols)
    results_df = pd.concat([results_df, results], axis=0)
results_df.to_csv('./results/Significance_results_{0}.csv'.format(date), header=True, index=False)
