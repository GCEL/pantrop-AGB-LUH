"""
15/10/2019 - DTM
Plotting PCA correlation matrix
"""

from useful import *
from sklearn.externals import joblib
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import glob

# Load PCA pipeline
pca = joblib.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/pca_pipeline.pkl')
# Load and transform predictors
predictors,landmask = get_predictors(y0=2000,y1=2009)
X = pca.transform(predictors)

# get variable names
# LUH
luhvars = ['primf','primn','secdf','secdn','urban','c3ann','c4ann','c3per','c4per','c3nfx','pastr','range','secma']

# - WorldClim2
wc2vars = []
for f in sorted(glob.glob('/disk/scratch/local.2/jexbraya/WorldClim2/0.25deg/*tif')):
    wc2vars.append(f.split('/')[-1][6:16])
# Soilgrids
sgvars=[]
for f in  sorted(glob.glob('/disk/scratch/local.2/jexbraya/soilgrids/0.25deg/*tif')):
    sgvars.append(f.split('/')[-1][:12])

# Calculate a correlation matrix
corrmat = np.zeros([predictors.shape[1],X.shape[1]])
for ii in range(corrmat.shape[0]):
    for jj in range(corrmat.shape[1]):
        corrmat[ii,jj] = pearsonr(predictors[:,ii],X[:,jj])[0]

# Plot correlation matrix
start_luh = 0
start_wc2 = start_luh+len(luhvars)
start_sg  = start_wc2+len(wc2vars)
fig = plt.figure(tight_layout=True,figsize=[8,11])
gs = GridSpec(7,2,figure=fig)
ax1=fig.add_subplot(gs[0:3,0])
ax2=fig.add_subplot(gs[3:6,0])
ax3=fig.add_subplot(gs[:,1])
ax1.imshow(corrmat[:start_wc2],cmap='bwr',vmin=-1,vmax=1)
ax2.imshow(corrmat[start_wc2:start_sg],cmap='bwr',vmin=-1,vmax=1)
im=ax3.imshow(corrmat[start_sg:],cmap='bwr',vmin=-1,vmax=1)

cax = fig.add_axes([0.14,0.1,0.3,0.02])
plt.colorbar(im,orientation='horizontal',label='Pearson correlation',cax=cax)

ax1.set_title('LUH')
ax1.set_yticks(range(len(luhvars)))
ax1.set_yticklabels(luhvars)
ax1.set_xticks(range(X.shape[1]))
ax1.set_xticklabels(np.arange(X.shape[1])+1)
ax1.set_xlabel('Principal components')

ax2.set_title('WorldClim2')
ax2.set_yticks(range(len(wc2vars)))
ax2.set_yticklabels(wc2vars)
ax2.set_xticks(range(X.shape[1]))
ax2.set_xticklabels(np.arange(X.shape[1])+1)
ax2.set_xlabel('Principal components')

ax3.set_title('SoilGrids')
ax3.set_yticks(range(len(sgvars)))
ax3.set_yticklabels(sgvars)
ax3.set_xticks(range(X.shape[1]))
ax3.set_xticklabels(np.arange(X.shape[1])+1)
ax3.set_xlabel('Principal components')

ax1.tick_params(axis='both',labelsize=8)
ax2.tick_params(axis='both',labelsize=8)
ax3.tick_params(axis='both',labelsize=8)
fig.savefig('../figures/manuscript/figS1_correlation_matrices.png')
plt.show()
