"""
16/11/2018 - JFE
This files fits the best model from GridSearch to the whole dataset and saves
it for quick use in production files

11/10/2019 - DTM
Minor alterations to account for changes in the previous scripts
 """

from useful import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import pandas as pd

pca = joblib.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/pca_pipeline.pkl')

#load the fitted rf_grid
rf_grid = np.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rf_grid.npz')['arr_0'][()]
# Construct best-fitting random forest model
idx = np.argmin(rf_grid['mean_test_score'])
rf_best = RandomForestRegressor(bootstrap=True,
            max_depth= None,
            max_features=rf_grid['params'][idx]['max_features'],
            min_samples_leaf=rf_grid['params'][idx]['min_samples_leaf'],
            n_estimators=1000,
            n_jobs=30,
            oob_score=True,
            random_state=26,
            )

#refit to whole dataset - get predictors and targets
predictors,landmask = get_predictors(y0=2000,y1=2009)
X = pca.transform(predictors)
med = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Map_0.25d.tif')[0].values[landmask]
unc = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Uncertainty_0.25d.tif')[0].values[landmask]

lvls = ['mean','upper','lower']
for aa, y in enumerate([med,med+unc,med-unc]):
    y[y<0] = 0
    X_resampled,y_resampled = balance_training_data(X,y,n_bins=10,random_state=31)
    rf_best.fit(X_resampled,y_resampled)
    print(lvls[aa])
    print('MSE: %.03f' % mean_squared_error(rf_best.predict(X),y))
    joblib.dump(rf_best,'/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rf_%s.pkl' % lvls[aa])
