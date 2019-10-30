"""
29/10/2019 - DTM
This files plots summary of optimisation search
"""
import sys
sys.path.append('../')
from useful import *
from sklearn.externals import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load the fitted rf_random
rf_random = np.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rfbc_random.npz')['arr_0'][()]
rf_grid = np.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rfbc_grid.npz')['arr_0'][()]

# create a pandas dataframe storing parameters and results of the cv
cv_res = pd.DataFrame(rf_random['params'])
#get the scores
cv_res['mean_train_score'] = rf_random['mean_train_score']
cv_res['mean_test_score'] = rf_random['mean_test_score']
cv_res['grad_test'] = rf_random['gradient_test']
cv_res['grad_train'] = rf_random['gradient_train']
cv_res['ratio_score'] = cv_res['mean_test_score'] / cv_res['mean_train_score']


grid_res = pd.DataFrame(rf_grid['params'])
#get the scores
grid_res['mean_train_score'] = rf_grid['mean_train_score']
grid_res['mean_test_score'] = rf_grid['mean_test_score']
grid_res['grad_test'] = rf_grid['gradient_test']
grid_res['grad_train'] = rf_grid['gradient_train']
grid_res['ratio_score'] = grid_res['mean_test_score'] / grid_res['mean_train_score']

cv_res = cv_res.append(grid_res)

#do some plots
col_vars = [ 'max_depth', 'max_features', 'min_samples_leaf', 'n_estimators', 'mean_test_score']
col_labels = [ 'max depth', 'max features', 'min samples\nleaf', 'N estimators', 'validation\nRMSE\nMg ha$^{-1}$']
row_vars = ['mean_test_score','mean_train_score','ratio_score','grad_test','grad_train']
row_labels = ['validation\nRMSE\nMg ha$^{-1}$', 'calibration\nRMSE\nMg ha$^{-1}$', 'cal:val\nRMSE ratio', 'validation\ngradient', 'calibration\ngradient']

sns.set()
fig,axes = plt.subplots(nrows=5,ncols=5,figsize=[8,9], sharex='col',sharey='row')
for ii,axes_row in enumerate(axes):
    for jj, ax in enumerate(axes_row):
        ax.plot(cv_res[col_vars[jj]],cv_res[row_vars[ii]],'.',ms=1)
        if jj==0:
            ax.set_ylabel(row_labels[ii],fontsize=8)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)
for jj, ax in enumerate(axes[-1]):
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
        ax.set_xlabel(col_labels[jj],fontsize=8)

fig.tight_layout()
fig.show()
fig.savefig('../figures/manuscript/figS2_optimisation_summary.png')
