# train all the models being considered using a subset of the ukbb training data
# keep 20% of the controls data for testing purposes 
# remove case individuals as these will be used to assess models ability to detect outliers 

import pandas as pd 
from os.path import join 
from multiviewae import mVAE, weighted_mVAE, mmVAE
import numpy as np
import os
from os.path import join
import random
import datetime

DATA_PATH = '../../UKBB_data/splines_regression'

mri_data = pd.read_csv(join(DATA_PATH, 't1/230215/t1_data_adjusted.csv'), header=0)
dti_data = pd.read_csv(join(DATA_PATH, 'dti/230215/dti_data_adjusted.csv'), header=0)

#process dti data
to_drop_dti = ['FA.FX', 'MD.FX']

dti_data = dti_data[dti_data.columns.drop(to_drop_dti)]
print(dti_data.shape)

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

#scale and centre data
mean_controls = np.mean(mri_train, axis=0)
sd_controls = np.std(mri_train, axis=0)

mri_train = (mri_train - mean_controls)/sd_controls

mean_controls = np.mean(dti_train, axis=0)
sd_controls = np.std(dti_train, axis=0)
dti_train = (dti_train - mean_controls)/sd_controls

mri_train = mri_train.to_numpy()
dti_train = dti_train.to_numpy()

#holdout 20% of controls for testing purposes
random.seed(42)
subj = list(range(0,mri_train.shape[0]))
holdout_subset = random.sample(subj, int(mri_train.shape[0]*0.2))
train_subset = list(set(subj) - set(holdout_subset))

mri_train = mri_train[train_subset,:]
dti_train = dti_train[train_subset,:]

#set model parameters 
input_dims=[mri_train.shape[1],dti_train.shape[1]]
max_epochs = 2000
batch_size = 256

train = np.concatenate((mri_train, dti_train), axis=1)

t = datetime.datetime.now()
tstr = t.strftime('%Y-%m-%d_%H%M')

#train models 
### TRAIN MULTI VIEW MODELS ###
mvae = mmVAE(
        cfg="./multi_view.yaml",
        input_dim=input_dims,
    )
mvae.cfg.out_dir = './results/mmVAE/{0}'.format(tstr)
print('fit mixture multi-view')
mvae.fit(mri_train, dti_train,  max_epochs=max_epochs, batch_size=batch_size)

mvae = weighted_mVAE(
        cfg="./multi_view.yaml",
        input_dim=input_dims,
    )
mvae.cfg.out_dir = './results/weighted_mVAE/{0}'.format(tstr)
print('fit weighted multi-view')
mvae.fit(mri_train, dti_train,  max_epochs=max_epochs, batch_size=batch_size)

mvae = mVAE(
        cfg="./multi_view.yaml",
        input_dim=input_dims,
    )
mvae.cfg.out_dir = './results/mVAE/{0}'.format(tstr)
print('fit mvae multi-view')
mvae.fit(mri_train, dti_train,  max_epochs=max_epochs, batch_size=batch_size)


### TRAIN SINGLE VIEW MODELS ###
mvae = mVAE(
        cfg="./single_view.yaml",
        input_dim=[input_dims[0]+input_dims[1]],
    )

print('fit vae concat' )
mvae.cfg.out_dir = './results/mVAE_concat/{0}'.format(tstr)
mvae.fit(train, max_epochs=max_epochs, batch_size=batch_size)

mvae = mVAE(
        cfg="./single_view.yaml",
        input_dim=[input_dims[0]],
    )

print('fit mvae mri')
mvae.cfg.out_dir = './results/mVAE_t1/{0}'.format(tstr)
mvae.fit(mri_train, max_epochs=max_epochs, batch_size=batch_size)

mvae = mVAE(
        cfg="./single_view.yaml",
        input_dim=[input_dims[1]],
    )

print('fit mvae dti')
mvae.cfg.out_dir = './results/mVAE_dti/{0}'.format(tstr)
mvae.fit(dti_train, max_epochs=max_epochs, batch_size=batch_size)
