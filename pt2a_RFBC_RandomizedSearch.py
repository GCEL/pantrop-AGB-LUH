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

#create the random forest object and fit it out of the box
rf = RandomForestRegressor(n_jobs=20,random_state=26,oob_score=True,bootstrap=True)
# fit model
rf1_,rf2_=rfbc_fit(rf,X_train,y_train)
#save the mse for the cal / val
oob_cal = np.sqrt(mean_squared_error(y_train,rfbc_predict(rf1_,rf2_,X_train)))
oob_val = np.sqrt(mean_squared_error(y_test,rfbc_predict(rf1_,rf2_,X_test)))
print('Out of the box RMSE:\n\tcalibration score = %f\n\tvalidation_score = %f' % (oob_cal,oob_val))

# perform a randomized search on hyper parameters using training subset of data

#define the parameters for the search
random_grid = { "n_estimators": np.linspace(200,2000,10,dtype='i'),
                "max_depth": list(np.linspace(5,100,20,dtype='i'))+[None],
                "max_features": np.linspace(.1,1.,10),
                "min_samples_leaf": list(np.linspace(5,50,10,dtype='i'))}
n_iter = 100
fold=3
RandomizedSearchResults = {}
RandomizedSearchResults['params']=[]
RandomizedSearchResults['scores']=[]
RandomizedSearchResults['mean_train_score']=[]
RandomizedSearchResults['mean_test_score']=[]
best_score = np.inf
print('Starting randomised search')
for ii in range(0,n_iter):
    print('{0}\r'.format(ii),end='\r')
    params={}
    for pp,param in enumerate(random_grid.keys()):
        params[param] = np.random.choice(random_grid[param])
    params['n_jobs'] = 30
    params['oob_score'] =True
    params['bootstrap'] =True
    RandomizedSearchResults['params'].append(params)
    scores = rfbc_cv(params,X_train,y_train,cv=fold,random_state=112358)
    RandomizedSearchResults['scores'].append(scores)
    RandomizedSearchResults['mean_test_score'].append(np.mean(scores['test']))
    RandomizedSearchResults['mean_train_score'].append(np.mean(scores['train']))
    if RandomizedSearchResults['mean_test_score'][ii]<best_score:
        best_score = RandomizedSearchResults['mean_test_score'][ii]
        print('\tNew Best RMSE: %.06f' % (best_score))
        print(params)

np.savez('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rf_random.npz',RandomizedSearchResults)
