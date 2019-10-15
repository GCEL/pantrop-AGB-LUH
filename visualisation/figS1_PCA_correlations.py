"""
15/10/2019 - DTM
Plotting PCA correlation matrix
"""

from useful import *
from sklearn.externals import joblib
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
pca = joblib.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/pca_pipeline.pkl')
predictors,landmask = get_predictors(y0=2000,y1=2009)
X = pca.transform(predictors)

#calculate a correlation matrix
corrmat = np.zeros([predictors.shape[1],X.shape[1]])
for ii in range(corrmat.shape[0]):
    for jj in range(corrmat.shape[1]):
        corrmat[ii,jj] = pearsonr(predictors[:,ii],X[:,jj])[0]

plt.imshow(corrmat)
