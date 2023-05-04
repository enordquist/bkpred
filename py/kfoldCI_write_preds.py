'''
Script for writing out predictions for splitN
Train on the 5-fold, then combine all training data to predict to test set
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
import sys
import forestci as fci

def rmse(y_hat, y): return np.sqrt( (y_hat - y)**2 ).mean()

paramdict = {
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
           'min_weight_fraction_leaf': 0.0,  # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
              
            }
folds = [0,1,2,3,4,5]
K = 5
splitN=sys.argv[1]

# loop over the folds, last one is full data
for fold in folds:
  # read in data
  # for the 5th fold (or last/"-1th" fold) this is the full training data
  X_train = np.load('data/'+splitN+'/X'+str(fold)+'_train.npy')
  y_train = np.load('data/'+splitN+'/y'+str(fold)+'_train.npy').ravel()
 
  # train fresh model
  model = RandomForestRegressor(**paramdict)
  model.fit(X_train,y_train)
  py_train = model.predict(X_train)  
  np.save('data/'+splitN+'/py'+str(fold)+'_train.npy', py_train)

  if fold < folds[-1]:
    X_test  = np.load('data/'+splitN+'/X'+str(fold)+'_test.npy')
    y_test  = np.load('data/'+splitN+'/y'+str(fold)+'_test.npy').ravel()
    py_test = model.predict(X_test)  
    np.save('data/'+splitN+'/py'+str(fold)+'_test.npy', py_test) # save prediction
    # ger error estimate
    error = fci.random_forest_error(model, X_train, X_test, calibrate=False)
    np.save('data/'+splitN+'/py'+str(fold)+'_test_error.npy', error) # save prediction error
  
  if fold == folds[-1]:
    X_train = np.load('data/'+splitN+'/X5_train.npy')
    y_train = np.load('data/'+splitN+'/y5_train.npy').ravel()
    
    X_test  = np.load('data/'+splitN+'/validation/X_test.npy')
    y_test  = np.load('data/'+splitN+'/validation/y_test.npy')
    
    # train fresh model
    model = RandomForestRegressor(**paramdict)
    py_train = model.fit(X_train,y_train)
    py_train = model.predict(X_train)
    np.save('data/'+splitN+'/validation/py_train.npy', py_train)
    
    py_test = model.predict(X_test)
    np.save('data/'+splitN+'/validation/py_test.npy', py_test)
    # calculate variance
    error = fci.random_forest_error(model, X_train, X_test, calibrate=False)
    np.save('data/'+splitN+'/validation/py_test_error.npy', error)
