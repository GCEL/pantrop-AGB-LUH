"""
23/3/17 - JFE
updated to get biomass a function of all
LU classes

22/3/17 - JFE
This script reconstructs ABC as a function of climate
in regions where LUH database indicates a certain level
of primary land in 2001

15/10/2019 - DTM
Major modifications - PCA, permutation importance
"""

import numpy as np
import xarray as xr
import sys
import pylab as pl
from sklearn.externals import joblib
from eli5.sklearn import PermutationImportance

sys.path.append('../')
import useful

# load PCA
pca = joblib.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/pca_pipeline.pkl')

# load predictors and transform
predictors,landmask = useful.get_predictors(y0=2000,y1=2009)
X = pca.transform(predictors)
varnames = np.arange(1,X.shape[0]+1).astype('str')
agb = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Map_0.25d.tif')[0].values[landmask]

# load random forest algorithm
rf = joblib.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rf_mean.pkl')

# Permutation Importance
perm = PermutationImportance(rf).fit(X, agb)
imp = perm.feature_importances_
impstd = perm.feature_importances_std_ #np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

fig = pl.figure('importances',figsize=(4,8));fig.clf()
ax=fig.add_subplot(111)
ax.barh(range(imp.size),imp,color='0.5',xerr=impstd,align='center')
pl.yticks(range(imp.size),varnames)
ax.tick_params('y',left=False,right=False)

ax.set_xlim(0,1)
pl.xlabel('variable importance')
pl.ylabel('principal component')
ax.set_ylim(-.5,imp.size-.5)
#fig.show()
fig.savefig('../figures/manuscript/figS4_importances',bbox_inches='tight')
