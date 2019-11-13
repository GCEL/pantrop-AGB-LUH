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


load = True
pca = joblib.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/pca_pipeline.pkl')

predictors,landmask = get_predictors(y0=2000,y1=2009)
continents = get_continents(landmask)
continents = continents[landmask].reshape(landmask.sum(),1)

#transform the data
X = pca.transform(predictors)
X = np.hstack((X,continents))

#get the agb data
y = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Map_0.25d.tif')[0].values[landmask]

#split train and test subset, specifying random seed
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=26)

#define the parameters for the gridsearch
"""
# hyperparams from first grid search
param_grid = { "max_features": np.linspace(.15,.45,7),
                "min_samples_leaf": np.linspace(1,10,10,dtype='i'),
                "max_depth": [50,100,150,200,300,400,500,None]}

# hyperparams for second grid search
param_grid = { "max_features": np.linspace(.2,.3,3),
                "min_samples_leaf": np.linspace(2,4,3,dtype='i'),
                "max_depth": [30,40,60,70,80]}
# hyperparams for third grid search
param_grid = { "max_features": np.linspace(.2,.3,3),
                "min_samples_leaf": np.linspace(2,4,3,dtype='i'),
                "max_depth": [5,10,15,20,25]}
"""
# hyperparams for fourth grid search
param_grid = { "max_features": np.linspace(.2,.25,2),
                "min_samples_leaf": np.linspace(2,3,2,dtype='i'),
                "max_depth": [18,21,22,23,24,26,27]}

#run the grid search - append results to existing search if load = True
fold=3
if load:
    GridSearchResults = np.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rfbc_grid.npz')['arr_0'][()]
    best_score = np.min(GridSearchResults['mean_test_score'])
    print('Continuing previous grid search\n\tBest RMSE: %.06f' % (best_score))
else:
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
            params={'max_features':p1,'min_samples_leaf':p2,'max_depth':p3,'n_estimators':1500,
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

#save grid search results to file
np.savez('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rfbc_grid.npz',GridSearchResults)
