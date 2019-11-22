"""
23/3/17 - JFE
updated to get biomass a function of all
LU classes

22/3/17 - JFE
This script reconstructs ABC as a function of climate
in regions where LUH database indicates a certain level
of primary land in 2001
"""
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
import pylab as pl
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from scipy.stats import gaussian_kde
import seaborn as sns
import sys
sys.path.append('../')
from useful import *

# A function for plotting the regression and associated stats
def plot_OLS(ax,target,Y,ss,mode='unicolor'):

    X = target
    X = sm.add_constant(X)

    model = sm.OLS(Y,X)

    results = model.fit()

    st, data, ss2 = summary_table(results, alpha=0.05)

    fittedvalues = data[:,2]
    predict_mean_se  = data[:,3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
    predict_ci_low, predict_ci_upp = data[:,6:8].T

    if mode == 'unicolor':
        ax.scatter(target,Y,c='silver',linewidths=0, s =4)
    else:
        xy = np.row_stack([target,Y])
        z = gaussian_kde(xy)(xy)
        idx=z.argsort()
        x,y,z = xy[0][idx],xy[1][idx],z[idx]
        ax.scatter(x,y,c=z,s=4,cmap=pl.cm.inferno_r)

    ax.plot(target,fittedvalues,'r-',label='Least Square Regression',lw=2)

    idx = np.argsort(predict_ci_low)
    ax.plot(target[idx],predict_ci_low[idx],'r--',lw=2,label='95% confidence interval')

    idx = np.argsort(predict_ci_upp)
    ax.plot(target[idx],predict_ci_upp[idx],'r--',lw=2)

    mx = np.ceil(max(target.max(),fittedvalues.max()))
    ax.plot([0,mx],[0,mx],'k-')

    ax.set_xlim(0,mx)
    ax.set_ylim(0,mx)

    ax.set_aspect(1)
    if ss == 0:
        ax.legend(loc='upper left')
    ax.set_xlabel('AGB from Avitabile et al.(2016) [Mg C ha$^{-1}$]')

    ax.set_ylabel('Reconstructed AGB [Mg C ha$^{-1}$]')

    nse = 1-((Y-target)**2).sum()/((target-target.mean())**2).sum()
    rmse = np.sqrt(((Y-target)**2).mean())

    ax.text(0.98,0.02,'y = %4.2fx + %4.2f\nR$^2$ = %4.2f; p < 0.001\nrmse = %4.1f Mg C ha$^{-1}$ ; NSE = %4.2f' % (results.params[1],results.params[0],results.rsquared,rmse,nse),va='bottom',ha='right',transform=ax.transAxes)


agb_to_C = 0.48
# load random forest algorithm
rf = joblib.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rfbc_mean.pkl')
# load predictors and fit pca
pca = joblib.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/pca_pipeline.pkl')
predictors,landmask = get_predictors(y0=2000,y1=2009)
continents = get_continents(landmask)
continents = continents[landmask].reshape(landmask.sum(),1)

#transform the data
X = pca.transform(predictors)
X = np.hstack((X,continents))

# load in target data
y = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Map_0.25d.tif')[0].values[landmask]

#split train and test subset, specifying random seed
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=26)
#X_train_resampled,y_train_resampled = balance_training_data(X_train,y_train,n_bins=10,random_state=31)
#rf.fit(X_train_resampled,y_train_resampled)
rf1,rf2=rfbc_fit(rf['rf1'],X_train,y_train)

#create some pandas df
df_train = pd.DataFrame({'obs':y_train*agb_to_C,'sim':rfbc_predict(rf1,rf2,X_train)*agb_to_C})
df_test =  pd.DataFrame({'obs':y_test*agb_to_C,'sim':rfbc_predict(rf1,rf2,X_test)*agb_to_C})

print('Cal-val\n---------\nNcal = %i; Nval = %i' % (y_train.size,y_test.size))

figcalval = pl.figure('validation',figsize=(15,15));figcalval.clf()
titles = ['Calibration','Validation']
for ss,df in enumerate([df_train,df_test]):
    ax=figval.add_subplot(1,2,ss+1)
    plot_OLS(ax,df['obs'],df['sim'],ss,'density')
    ax.set_title(titles[ss])

figcalval.show()
figcalval.savefig('../figures/manuscript/figS3_calval_scatter.png',bbox_inches='tight')

figval = pl.figure('validation',figsize=(6,6));figval.clf()
ax=figval.add_subplot(111)
plot_OLS(ax,df_test['obs'],df_test['sim'],0,'density')
figval.show()
figval.savefig('../figures/manuscript/figS3_val_scatter.png',bbox_inches='tight')


# full model
rf1,rf2=rfbc_fit(rf['rf1'],X,y)
print('Final fit\n---------\nN = %i' % (y.size))
df_final = pd.DataFrame({'obs':y*agb_to_C,'sim':rfbc_predict(rf1,rf2,X)*agb_to_C})
figfinal = pl.figure(figsize=(8,6));figfinal.clf()
ax = figfinal.add_subplot(111)
plot_OLS(ax,df_final['obs'],df_final['sim'],0,'density')
figfinal.show()
figfinal.savefig('../figures/manuscript/figS4_cal_final_scatter.png',bbox_inches='tight')
