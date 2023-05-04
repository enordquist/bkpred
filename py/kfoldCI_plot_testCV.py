'''
Script for plotting predicitons
plot all 5 folds of complete training data and validation
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
'''

import numpy  as np
import pandas as pd
import csv
import scipy
import pickle
from sklearn.ensemble        import RandomForestRegressor
from sklearn.linear_model import LinearRegression # for fitting weighted linear regression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing   import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import addcopyfighandler
import sys
import forestci as fci

def rmse(y_hat, y): return np.sqrt( (y_hat - y)**2 ).mean()

def plot(ty,py,yerr,title,ax,elw=0,boxloc=(0.63,0.07),color='C0'):
  ax.grid(zorder=-1)
  ax.axline((0,0),slope=1,linewidth=0.5,color='k',zorder=0)
  #ty = ty.reshape(len(ty),)
  #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(ty, py)
  regress = LinearRegression()
  regress.fit(ty.reshape(-1, 1), py.reshape(-1, 1) )
  r_value,p_value=scipy.stats.pearsonr(ty, py)

  ax.plot(np.unique(ty), regress.predict(np.unique(ty).reshape(-1,1)),color=color,alpha=0.7,zorder=0)
  #ax.plot(np.unique(ty), intercept + slope*np.unique(ty),color=color,alpha=0.7,zorder=0)

  ax.errorbar(ty,py,yerr=yerr,color=color,fmt='o',elinewidth=elw,ecolor='k',markersize=3,capsize=2.0*elw,capthick=elw,zorder=10)
  ax.set_xticks(np.arange(-150,151,50))
  ax.set_yticks(np.arange(-150,151,50))
  RMSE = rmse(ty,py)
  text=title+'\nRMSE = '+f'{RMSE:.1f}'+'\nR = '+f'{r_value:.2f}'
  ax.annotate(text, xy=boxloc, xycoords='axes fraction', bbox=dict(boxstyle='round', fc='w'))
  ax.set_xlim([-195,195])
  ax.set_ylim([-195,195])
  print('Pearson,%.2f'%r_value)
  #print('Spearman,%.2f,%.2E'%scipy.stats.spearmanr(py,ty))

fig,axes = plt.subplots(3,2,figsize=(7,8),sharex=True,sharey=True)    # k-fold CV

folds = [0,1,2,3,4,5]
K = 5
splitNs=['split'+str(i) for i in [0,1,2,3,4]]

axes = axes.flatten()
axes[1].axis('off')
axes=[axes[0],axes[2],axes[3],axes[4],axes[5]]
# loop over splits
for splitN,ax in zip(splitNs,axes): 
  X_train = np.load('data/'+splitN+'/X5_train.npy')
  y_train = np.load('data/'+splitN+'/y5_train.npy').ravel()
  py_train = np.load('data/'+splitN+'/py5_train.npy').ravel()
  
  X_test  = np.load('data/'+splitN+'/validation/X_test.npy')
  y_test  = np.load('data/'+splitN+'/validation/y_test.npy')
  
  plot(y_train, py_train, np.zeros(y_train.shape[0]), r'$\bf{Train}$', ax, boxloc=(0.05,0.75))
  
  py_test  = np.load('data/'+splitN+'/validation/py_test.npy')
  # calculate variance
  error = np.load('data/'+splitN+'/validation/py_test_error.npy')
  plot(y_test, py_test, np.sqrt(error), r'$\bf{Test}$', ax, elw=1.0, color='C1')

fig.supylabel(r'Predicted $\bf{\Delta V_{1/2}}$',fontweight='bold')
fig.supxlabel(r'True $\bf{\Delta V_{1/2}}$',fontweight='bold')

fig.tight_layout()
plt.show()
