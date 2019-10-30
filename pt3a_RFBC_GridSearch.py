"""
14/11/2018 - JFE
This files performs a grid search on random forest hyperparameters to fit
AGB data as a function of climate, soil properties and land use.
Parameters have been chosen after an initial RandomizedSearch highlighted
the most sensitive parameters.
"""

from useful import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
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
param_grid = { "max_features": np.linspace(.15,.45,7),
                "min_samples_leaf": np.linspace(1,10,10,dtype='i'),
                "max_depth": list(np.linspace(200,500,4,dtype='i'))+[None]}

#create the random forest object with predefined parameters
rf = RandomForestRegressor(n_jobs=30,random_state=26,oob_score=True,
                            n_estimators = 1800,bootstrap=True)

fold=3
GridSearchResults = {}
GridSearchResults['params']=[]
GridSearchResults['scores']=[]
GridSearchResults['mean_train_score']=[]
GridSearchResults['mean_test_score']=[]
GridSearchResults['gradient_train']=[]
GridSearchResults['gradient_test']=[]
best_score = np.inf
np1 = param_grid['max_features'].size
np2 = param_grid['min_samples_leaf'].size
np3 = len(param_grid['max_depth'])
for ii,p1 in enumerate(param_grid['max_features']):
    for jj,p2 in enumerate(param_grid['min_samples_leaf']):
        for kk,p3 in enumerate(param_grid['max_depth']):
            print('{0:.3f}%\r'.format(float(ii*np2*np3+jj*np3+kk)/float((np1*np2*np3))*100),end='\r')
            params={'max_features':p1,'min_samples_leaf':p2,'max_depth':p3,'n_estimators':1800,
                    'random_state':26,'n_jobs':30,'bootstrap':True,'oob_score':True}
            GridSearchResults['params'].append(params)
            # run balanced cross validation
            scores = rfbc_cv(params,X_train,y_train,cv=fold,random_state=112358)
            GridSearchResults['scores'].append(scores)
            GridSearchResults['mean_test_score'].append(np.mean(scores['test']))
            GridSearchResults['mean_train_score'].append(np.mean(scores['train']))
            GridSearchResults['gradient_train'].append(np.mean(scores['gradient_train']))
            GridSearchResults['gradient_test'].append(np.mean(scores['gradient_test']))
            if np.mean(scores['test']) < best_score:
                best_score = np.mean(scores['test'])
                print('\tNew Best RMSE: %.06f' % (best_score))
                print(params)

np.savez('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rfbc_grid.npz',GridSearchResults)

"""
#perform a grid search on hyper parameters using training subset of data
rf_grid = GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,
                            verbose = 3,scoring = 'neg_mean_squared_error', n_jobs=1)

rf_grid.fit(X_train,y_train)

#save the fitted rf_grid
joblib.dump(rf_grid,'/disk/scratch/local.2/jexbraya/pantrop-AGB-LUH/saved_algorithms/rf_grid.pkl',compress=1)
"""
