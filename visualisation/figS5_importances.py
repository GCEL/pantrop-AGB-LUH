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
import pandas as pd
import sys
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.externals import joblib
from eli5.permutation_importance import get_score_importances
from sklearn.model_selection import train_test_split
from scipy import stats

sys.path.append('../')
import useful

# load PCA
pca = joblib.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/pca_pipeline.pkl')

# load predictors and transform
predictors,landmask = useful.get_predictors(y0=2000,y1=2009)
y = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Map_0.25d.tif')[0].values[landmask]
X = pca.transform(predictors)
continents = useful.get_continents(landmask)
continents = continents[landmask].reshape(landmask.sum(),1)

#transform the data
X = pca.transform(predictors)
X = np.hstack((X,continents))

# load random forest algorithm
rf = joblib.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rfbc_mean.pkl')
rf1=rf['rf1']
rf2=rf['rf2']

# Permutation Importance
# - define the score used to underpin importance values
def r2_score(X,y):
    y_rfbc = useful.rfbc_predict(rf1,rf2,X)
    temp1,temp2,r,temp3,temp4 = stats.linregress(y,y_rfbc)
    return r**2
n_iter=5
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,test_size=0.25,random_state=23)
base_score,score_drops = get_score_importances(r2_score,X_test,y_test,n_iter=n_iter)
labels = []
for ii in range(1,X.shape[1]):
    labels.append('PC%s' % str(ii).zfill(2))
labels.append('Region')

var_labels = labels*n_iter
var_imp = np.zeros(n_iter*len(labels))
for ii,drops_iter in enumerate(score_drops):
    var_imp[ii*len(labels):(ii+1)*len(labels)] = drops_iter
imp_df = pd.DataFrame(data = {'variable': var_labels,
                              'permutation_importance': var_imp})

fig,axis= plt.subplots(nrows=1,ncols=1,figsize=[5,8],sharex=True)
sns.barplot(x='permutation_importance',y='variable',ci='sd',data=imp_df,ax=axis,color='0.5')
axis.set_ylabel('Principal component')
axis.set_xlabel('Permutation importance')
fig.tight_layout()
axis.set_xlim(0,1)
#fig.show()
fig.savefig('../figures/manuscript/figS5_importances',bbox_inches='tight')
