'''
Predictions for the whole channel
k-fold CV, with ensemble model and ensemble fit error estimation
BK V1/2 predictor with random forest
ebn jun 2022
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
'''

import numpy  as np
import pandas as pd
import csv
import scipy
import pickle
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing   import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import forestci as fci

# want the three_to_one
import Bio
from Bio.PDB import *
from Bio.PDB.Polypeptide import three_to_one
import random

myparamdict = {
              'n_estimators':          500,  # The number of trees in the forest.
              'min_samples_split':       2,  # The minimum number of samples required to split an internal node
              'max_leaf_nodes':        100,  # grow trees with MLN in best-first fashion (None = unlimited)
              'max_depth':              20,  # max depth of tree
              'max_features':            1,  # The fraction of features to consider when looking for the best split, used 0.01 or just int(1)
              'ccp_alpha':            0.01,  # Complexity parameter used for Minimal Cost-Complexity Pruning.
              'bootstrap':            True,  # Bootstrap AGGregation (BAGGing): aggregate many weak models built on sub-sampling the data
              'oob_score':            True,  # use Out-of-Bag samples to estimate generalization score
              'criterion': 'squared_error',  # rmse
              'max_samples':           1.0,  # fraction of samples to train each base estimator with
              'min_samples_leaf':        1,  # The minimum number of samples required to be at a leaf node. Split if >= MLS training samples in each of the left and right branches.
              'min_weight_fraction_leaf': 0.0,  # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. must be <=0.5
            }

# read in data
X_train = np.load('data/X5_train.npy')
y_train = np.load('data/y5_train.npy').ravel()
X_test  = np.load('data/validation/X_test.npy')
y_test  = np.load('data/validation/y_test.npy').ravel()
X_all   = np.concatenate([X_train,X_test])
y_all   = np.concatenate([y_train,y_test])
print(X_train.shape,X_test.shape,X_all.shape)
print(y_train.shape,y_test.shape,y_all.shape)
print('Fitting grand master model on all data')

model = RandomForestRegressor(**myparamdict)
model.fit(X_all, y_all)

# easier to do this all in a loop by residue, so I have the per-residue predictions as well as the full scan
# then I can come back with another script to map them to PDB structures
predictions = {}
for mt_resname in ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']:
  print('  ',mt_resname)
  # read in all X for the (mt_resname) scan
  X = np.load('data/scans/'+mt_resname+'.npy')
  py_train = model.predict(X)
  error = fci.random_forest_error(model, X_train, X, calibrate=False)
  error = np.sqrt(np.abs(error))
  
  with open('/home/enordquist/work/bk/model/rosetta/data/ros2pdb.pkl', 'rb') as f: ros2pdb = pickle.load(f)
  
  for index in np.arange(856):
    resid = ros2pdb[ index + 1 ] # resid starts at 1, we want the PDB numbering (Rosetta's is sequential: [1-856])

    seq = pd.read_csv('seq')
    wt_resname = seq.iloc[index,:].values[0]
 
    # get the predicted dV shift
    predicted_shift = py_train[ index ]
    predicted_error = error[ index ]
    # name of mutation as in A316S 
    mutation_name = three_to_one(wt_resname)+str(resid)+mt_resname
    
    # construct entry in predictions dict
    predictions[ mutation_name ] = [ predicted_shift, predicted_error, resid, three_to_one(wt_resname), mt_resname ]

preds_df = pd.DataFrame.from_dict(predictions, orient='index')
preds_df.columns = ['dV','error','ResID','WT','MT']
preds_df.index.name = 'Mutation'
preds_df = preds_df.sort_values(by=['ResID','MT'])
print(preds_df)
preds_df.to_csv('preds/predictionsCI.csv', float_format='%.2f')
print('preds/predictionsCI.csv written')
