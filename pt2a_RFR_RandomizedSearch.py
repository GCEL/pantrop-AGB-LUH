"""
13/11/2018 - JFE
This files performs a randomized search on random forest hyperparameters to fit
AGB data as a function of climate, soil properties and land use.
The output of the randomized search will be used to inform a gridsearch.
"""

from useful import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

pca = joblib.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/pca_pipeline.pkl')

predictors,landmask = get_predictors(y0=2000,y1=2009)

#transform the data
X = pca.transform(predictors)

#get the agb data
y = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Map_0.25d.tif')[0].values[landmask]

#split train and test subset, specifying random seed
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=26)

#define the parameters for the gridsearch
random_grid = { "bootstrap":[True,False],
                "n_estimators": np.linspace(200,2000,10,dtype='i'),
                "max_depth": list(np.linspace(5,100,20,dtype='i'))+[None],
                "max_features": np.linspace(.1,1.,10),
                "min_samples_leaf": np.linspace(5,50,10,dtype='i') }

#create the random forest object and fit it out of the box
rf = RandomForestRegressor(n_jobs=20,random_state=26)
# rebalance the training set
X_train_resampled,y_train_resampled = balance_training_data(X_train,y_train,n_bins=10,random_state=31)
# fit model
rf.fit(X_train_resampled,y_train_resampled)
#save the mse for the cal / val
oob_cal = mean_squared_error(y_train,rf.predict(X_train))
oob_val = mean_squared_error(y_test,rf.predict(X_test))
print('Out of the box RMSE:\n\tcalibration score = %f\n\tvalidation_score = %f' % (oob_cal,oob_val))

# perform a randomized search on hyper parameters using training subset of data
n_iter = 100
fold=3
RandomizedSearchResults = {}
RandomizedSearchResults['params']=[]
RandomizedSearchResults['scores']=[]
RandomizedSearchResults['mean_scores']=[]
best_score = np.inf
for ii in range(0,n_iter):
    params={}
    for pp,param in enumerate(random_grid.keys()):
        params[param] = np.random.choice(random_grid[param])
    RandomizedSearchResults['params'].append(params)
    scores = balanced_cv(params,X_train,y_train,cv=fold,random_state=2097)
    RandomizedSearchResults['scores'].append(scores)
    RandomizedSearchResults['mean_scores'].append(np.mean(scores))
    if RandomizedSearchResults['mean_scores'][ii]<best_score:
        best_score = RandomizedSearchResults['mean_scores'][ii]
        print('\tNew Best RMSE: %.06f' % (best_score))
        print(params)

np.savez(RandomizedSearchResults,'/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rf_random.npy')

"""
# commented out previous iteration of the randomized search
rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,cv=3,
                            verbose = 3,scoring = 'neg_mean_squared_error',
                            random_state=26, n_iter=100, n_jobs=1)

rf_random.fit(X_train_resampled,y_train_resampled)

#save the fitted rf_random
joblib.dump(rf_random,'/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rf_random.pkl')
"""
