# create significance ratio for data deviations for UKBB disease cohort and holdout cohort
# both data space metrics

import pandas as pd 
from os.path import join 
import numpy as np
import random
import torch
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.covariance import MinCovDet


def deviation_sig_count(cohort_recon, holdout_recon, train_recon, key, cols, title='', zscore=True):
    if zscore:
        thresh = 0.05
        cohort_recon = np.mean(cohort_recon, axis=1)
        holdout_recon = np.mean(holdout_recon, axis=1)
        mean_holdout = np.mean(holdout_recon, axis=0)
        sd_holdout = np.std(holdout_recon, axis=0)
        z_scores = (cohort_recon - mean_holdout)/sd_holdout
        pvals = norm.sf(z_scores)
        count = (pvals <= thresh).sum()
        ratio = count/cohort_recon.shape[0]
        df = pd.DataFrame(np.array([key+title, count, ratio]).reshape(1,-1),
        columns=cols)
    else:  
        thresh = 0.001
        dists = calc_robust_mahalanobis_distance(cohort_recon, train_recon)
        pvals_cohort = pvals = 1 - chi2.cdf(dists, cohort_recon.shape[1] - 1)
        dists = calc_robust_mahalanobis_distance(holdout_recon, train_recon)
        pvals_holdout = pvals = 1 - chi2.cdf(dists, cohort_recon.shape[1] - 1)

        count_cohort = (pvals_cohort <= thresh).sum()
        count_holdout = (pvals_holdout <= thresh).sum()

        ratio_cohort = count_cohort/pvals_cohort.shape[0]
        ratio_holdout = count_holdout/pvals_holdout.shape[0]
        ratio = ratio_cohort/ratio_holdout
        
        df = pd.DataFrame(np.array([key+title, count_cohort, ratio_cohort, count_holdout, ratio_holdout, ratio]).reshape(1,-1),
        columns=cols)

    return df

def calc_robust_mahalanobis_distance(values, train_values):
    # fit a MCD robust estimator to data
    robust_cov = MinCovDet(random_state=42).fit(train_values)
    mahal_robust_cov = robust_cov.mahalanobis(values)
    return mahal_robust_cov

def deviation(orig, recon, recon_type='abs'):
    return np.sqrt((orig - recon)**2)

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

mri_cols = mri_train.columns
dti_cols = dti_train.columns

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

mri_cols = ['model_type'] + list(mri_cols)
mri_df = pd.DataFrame(columns = mri_cols)
dti_cols = ['model_type'] + list(dti_cols)

dti_df = pd.DataFrame(columns = dti_cols)

zscore = False
if zscore:
    cols = ['model', 'count cohort', 'ratio cohort']
else:
    cols = ['model', 'count cohort', 'ratio cohort', 'count holdout', 'ratio holdout', 'cohort/holdout ratio']
results_df = pd.DataFrame(columns=cols)
for key, val in model_dict.items():
    model = torch.load(join(val, 'model.pkl'))
    if key == 'mVAE_concat':
        train_recon = model.predict_reconstruction(train)
        holdout_recon = model.predict_reconstruction(holdout)
        test_recon = model.predict_reconstruction(test)#
        dev_holdout = deviation(holdout, holdout_recon[0][0])
        dev_test = deviation(test, test_recon[0][0])
        dev_train = deviation(train, train_recon[0][0])
        mri_dev_holdout, dti_dev_holdout = dev_holdout[:, 0:len(mri_cols)-1], dev_holdout[:, len(mri_cols)-1:]
        mri_dev_test, dti_dev_test = dev_test[:, 0:len(mri_cols)-1], dev_test[:, len(mri_cols)-1:]
        mri_dev_train, dti_dev_train = dev_train[:, 0:len(mri_cols)-1], dev_train[:, len(mri_cols)-1:]

        results = deviation_sig_count(mri_dev_test, mri_dev_holdout, mri_dev_train, key, cols, title='_mri',zscore=zscore)
        results_df = pd.concat([results_df, results], axis=0)
        results = deviation_sig_count(dti_dev_test, dti_dev_holdout, dti_dev_train, key, cols, title='_dti',zscore=zscore)
        results_df = pd.concat([results_df, results], axis=0)

    elif key == 'mVAE_t1':
        train_recon = model.predict_reconstruction(mri_train)
        holdout_recon = model.predict_reconstruction(mri_holdout)
        test_recon = model.predict_reconstruction(mri_test)
        mri_dev_holdout = deviation(mri_holdout, holdout_recon[0][0])
        mri_dev_test = deviation(mri_test, test_recon[0][0])
        mri_dev_train = deviation(mri_train, train_recon[0][0])
        results = deviation_sig_count(mri_dev_test, mri_dev_holdout, mri_dev_train, key, cols, title='_mri', zscore=zscore)
        results_df = pd.concat([results_df, results], axis=0)

    elif key == 'mVAE_dti':
        train_recon = model.predict_reconstruction(dti_train)
        holdout_recon = model.predict_reconstruction(dti_holdout)
        test_recon = model.predict_reconstruction(dti_test)
        dti_dev_holdout = deviation(dti_holdout, holdout_recon[0][0])
        dti_dev_test = deviation(dti_test, test_recon[0][0])
        dti_dev_train = deviation(dti_train, train_recon[0][0])

        results = deviation_sig_count(dti_dev_test, dti_dev_holdout, dti_dev_train, key, cols, title='_dti', zscore=zscore)
        results_df = pd.concat([results_df, results], axis=0)

    elif key == 'mmVAE':
        train_recon = model.predict_reconstruction(mri_train, dti_train)
        holdout_recon = model.predict_reconstruction(mri_holdout, dti_holdout)
        test_recon = model.predict_reconstruction(mri_test, dti_test)
        
        mri_dev_holdout = deviation(mri_holdout, holdout_recon[0][0])
        dti_dev_holdout = deviation(dti_holdout, holdout_recon[1][1])
        mri_dev_train = deviation(mri_train, train_recon[0][0])

        mri_dev_test = deviation(mri_test, test_recon[0][0])
        dti_dev_test = deviation(dti_test, test_recon[1][1])
        dti_dev_train = deviation(dti_train, train_recon[1][1])

        results = deviation_sig_count(mri_dev_test, mri_dev_holdout, mri_dev_train, key, cols, title='_mri',zscore=zscore)
        results_df = pd.concat([results_df, results], axis=0)
        results = deviation_sig_count(dti_dev_test, dti_dev_holdout, dti_dev_train, key, cols, title='_dti',zscore=zscore)
        results_df = pd.concat([results_df, results], axis=0)

    else:
        train_recon = model.predict_reconstruction(mri_train, dti_train)
        holdout_recon = model.predict_reconstruction(mri_holdout, dti_holdout)
        test_recon = model.predict_reconstruction(mri_test, dti_test)
        
        mri_dev_holdout = deviation(mri_holdout, holdout_recon[0][0])
        dti_dev_holdout = deviation(dti_holdout, holdout_recon[0][1])
        
        mri_dev_test = deviation(mri_test, test_recon[0][0])
        dti_dev_test = deviation(dti_test, test_recon[0][1])

        mri_dev_train = deviation(mri_train, train_recon[0][0])
        dti_dev_train = deviation(dti_train, train_recon[0][1])

        results = deviation_sig_count(mri_dev_test, mri_dev_holdout, mri_dev_train, key, cols, title='_mri',zscore=zscore)
        results_df = pd.concat([results_df, results], axis=0)
        results = deviation_sig_count(dti_dev_test, dti_dev_holdout, dti_dev_train, key, cols, title='_dti',zscore=zscore)
        results_df = pd.concat([results_df, results], axis=0)

if zscore:
    results_df.to_csv('./results/Significance_results_data_zscore_{0}.csv'.format(date), header=True, index=False)
else:
    results_df.to_csv('./results/Significance_results_data_mahal_{0}.csv'.format(date), header=True, index=False)
