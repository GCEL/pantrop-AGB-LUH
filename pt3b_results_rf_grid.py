"""
14/11/2018 - JFE
This files analyzes the output of the randomized search to create the input for
the gridsearch
"""

from useful import *
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load the fitted rf_grid
rf_grid = np.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/s_AGB_potential_RFR_worldclim_soilgrids.npy')[()]

# create a pandas dataframe storing parameters and results of the cv
#cv_res = pd.DataFrame(rf_grid.cv_results_['params'])
cv_res = pd.DataFrame(rf_grid.['params'])
params = cv_res.columns #save parameter names for later
#get the scores as RMSE
#cv_res['mean_train_score'] = .5*(-rf_grid.cv_results_['mean_train_score'])**.5
#cv_res['mean_test_score'] = .5*(-rf_grid.cv_results_['mean_test_score'])**.5
cv_res['mean_train_score'] = rf_grid['mean_train_score']
cv_res['mean_test_score'] = rf_grid['mean_test_score']
cv_res['ratio_score'] = cv_res['mean_test_score'] / cv_res['mean_train_score']

# Construct best-fitting random forest model
idx = np.argmin(cv_res['mean_test_score'])
rf_best = RandomForestRegressor(bootstrap=cv_res['bootstrap'][idx],
            max_depth= cv_res['max_depth'][idx],
            max_features=cv_res['max_features'][idx],
            min_samples_leaf=cv_res['min_samples_leaf'][idx],
            n_estimators=cv_res['n_estimators'][idx],
            n_jobs=30,
            oob_score=True,
            random_state=26,
            )

#do some plots
pca = joblib.load('/disk/scratch/local.2/dmilodow/pantrop-AGB-LUH/saved_algorithms/pca_pipeline.pkl')
predictors,landmask = get_predictors(y0=2000,y1=2009)

#transform the data
X = pca.transform(predictors)
#get the agb data
y = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Map_0.25d.tif')[0].values[landmask]
#split train and test subset, specifying random seed
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=26)

#create some pandas df
df_train = pd.DataFrame({'obs':y_train,'sim':rf_best.predict(X_train)})
df_train.sim[df_train.sim<0] = 0.

df_test =  pd.DataFrame({'obs':y_test,'sim':rf_best.predict(X_test)})
df_test.sim[df_test.sim<0] = 0.
#plot
sns.set()
fig = plt.figure('cal/val grid',figsize=(10,6))
fig.clf()
#first ax
titles = ['a) Calibration','b) Validation']
for dd, df in enumerate([df_train,df_test]):
    ax = fig.add_subplot(1,2,dd+1,aspect='equal')
    sns.regplot(x='obs',y='sim',data=df,scatter_kws={'s':1},line_kws={'color':'k'},ax=ax)

    #adjust style
    ax.set_title(titles[dd]+' (n = %05i)' % df.shape[0])
    plt.xlim(0,550);plt.ylim(0,550)
    plt.xlabel('AGB from Avitabile et al. (2016) [Mg ha $^{-1}$]')
    plt.ylabel('Reconstructed AGB [Mg ha $^{-1}$]')

#show / save
fig.show()
#plt.show()
