"""
12/11/2018 - JFE
This file contains the definition of some useful functions
for the pantrop-AGB-LUH work

08/10/2019 - DTM
Added new method for balancing training data prior to model fitting.
"""

import xarray as xr #xarray to read all types of formats
import glob
import numpy as np
import sys
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

def get_predictors(y0=2000,y1=None,luh_file='/disk/scratch/local.2/jexbraya/LUH2/states.nc',return_landmask = True):

    #check that both years are defined, if not assume only one year of data needed
    if y1 == None:
        y1 = y0
    #check that first year is before the last, otherwise invert
    if y0 > y1:
        y0,y1 = y1,y0

    print('Getting data for years %4i - %4i' % (y0,y1))

    ### first get datasets
    #LUH2 data provided by Uni of Maryland
    luh = xr.open_dataset(luh_file,decode_times=False)
    luh_mask = ~luh.primf[0].isnull()
    print('Loaded LUH data')

    #define time - start in 850 for the historical, 2015 for the SSPs
    if luh_file == '/disk/scratch/local.2/jexbraya/LUH2/states.nc':
        luh_time = 850+luh.time.values
    else:
        luh_time = 2015+luh.time.values

    #worldclim2 data regridded to 0.25x0.25
    wc2 = xr.concat([xr.open_rasterio(f) for f in sorted(glob.glob('/disk/scratch/local.2/jexbraya/WorldClim2/0.25deg/*tif'))],dim='band')
    wc2_mask = wc2[0]!=wc2[0,0,0]
    for ii in range(wc2.shape[0]):
        wc2_mask = wc2_mask & (wc2[ii]!=wc2[ii,0,0])
    print('Loaded WC2 data')

    #soilgrids data regridded to 0.25x0.25
    soil= xr.concat([xr.open_rasterio(f) for f in sorted(glob.glob('/disk/scratch/local.2/jexbraya/soilgrids/0.25deg/*tif'))],dim='band')
    soil_mask = soil[0]!=soil[0,0,0]
    for ii in range(soil.shape[0]):
        soil_mask = soil_mask & (soil[ii]!=soil[ii,0,0])
    print('Loaded SOILGRIDS data')

    #also load the AGB data to only perform the PCA for places where there is both AGB and uncertainty
    agb = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Map_0.25d.tif')
    agb_mask = agb[0]!=agb.nodatavals[0]

    unc = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Uncertainty_0.25d.tif')
    unc_mask = unc[0]!=unc.nodatavals[0]

    #create the land mask knowing that top left pixels (NW) are all empty
    landmask = (luh_mask.values & wc2_mask.values & soil_mask.values & agb_mask.values & unc_mask.values)

    #define the LUH variables of interest
    luh_pred = ['primf','primn','secdf','secdn','urban','c3ann','c4ann','c3per','c4per','c3nfx','pastr','range','secma']
    #create the empty array to store the predictors
    predictors = np.zeros([landmask.sum(),len(luh_pred)+soil.shape[0]+wc2.shape[0]])

    #iterate over variables to create the large array with data
    counter = 0

    #first LUH
    for landuse in luh_pred:
        #get average over the period
        if y0 != y1:
            predictors[:,counter] = luh[landuse][(luh_time>=y0) & (luh_time<=y1)].mean(axis=0).values[landmask]
        else:
            predictors[:,counter] = luh[landuse][(luh_time==y0)].values[0][landmask]
        counter += 1
    print('Extracted LUH data')
    #then wc2
    for bi in wc2:
        predictors[:,counter] = bi.values[landmask]
        counter += 1
    print('Extracted WC2 data')
    #then soil properties
    for sp in soil:
        predictors[:,counter] = sp.values[landmask]
        counter += 1
    print('Extracted SOILGRIDS data')

    if return_landmask:
        return(predictors,landmask)
    else:
        return(predictors)


# get areas for a global grid with 0.25x0.25 res
def get_areas():
    latorig = np.arange(90-1/8.,-90.,-1/4.)
    lonorig = np.arange(-180+1/8.,180.,1/4.)
    areas = np.zeros([latorig.size,lonorig.size])
    res = np.abs(latorig[1]-latorig[0])
    for la,latval in enumerate(latorig):
        areas[la]= (6371e3)**2 * ( np.deg2rad(0+res/2.)-np.deg2rad(0-res/2.) ) * (np.sin(np.deg2rad(latval+res/2.))-np.sin(np.deg2rad(latval-res/2.)))

    return areas

"""
balance_training_data
-----------------------
Balance training data for random forest regression by binning y into evenly
spaced bins (n=n_bins; default 10) then using naive random upsampling from
imbalanced learn
"""
def balance_training_data(X,y,n_bins=10,target=None,random_state=None):
    # bin data
    y = y[np.isfinite(y)]; X=X[np.isfinite(y)]
    ymin=np.min(y);ymax=np.max(y);yrange=ymax-ymin;width=yrange/n_bins
    bins = np.arange(ymin,ymax,width)
    label = np.zeros(y.size).astype('str')
    for ii,margin in enumerate(bins):
        label[y>=margin]=str(ii)
    labels,counts = np.unique(label,return_counts=True)
    # balance data
    idx=np.arange(0,y.size,dtype='int')
    if target is None:
        if random_state is None:
            ros = RandomOverSampler()
        else:
            ros = RandomOverSampler(random_state=random_state)
    else:
        target_dict = {}
        for ll, lab in enumerate(labels):
            if counts[ll]<target:
                target_dict[lab]=target
            else:
                target_dict[lab]=counts[ll]
        if random_state is None:
            ros = RandomOverSampler(sampling_strategy=target_dict)
        else:
            ros = RandomOverSampler(sampling_strategy=target_dict,random_state=random_state)

    idx_resampled,label_resampled = ros.fit_resample(idx.reshape(y.size,1),label)
    # Apply resampling index
    X_resampled=X[idx_resampled.reshape(idx_resampled.size),:]
    y_resampled=y[idx_resampled.reshape(idx_resampled.size)]
    return X_resampled, y_resampled


"""
balanced_cv
----------------------
Need a cross validation procedure for balanced trees that maintains the
independence of train and test sets between folds.

"""
def balanced_cv(params,X,y,cv=3,target=None,random_state=None):
    # create RandomForestRegressor object
    rf = RandomForestRegressor(**params)
    # create Kfold object
    kf = KFold(n_splits=cv,shuffle=True,random_state=random_state)
    ii=0
    scores = {'test':np.zeros(cv),'train':np.zeros(cv)}
    # loop through the folds
    for train_index, test_index in kf.split(X):
        # obtain train-test split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # balance training set for fold
        X_resampled,y_resampled = balance_training_data(X_train,y_train,n_bins=10,target=target,random_state=random_state+ii)
        # fit random forest model for fold
        rf.fit(X_resampled,y_resampled)
        # get prediction for test set
        y_rf_train = rf.predict(X_train)
        y_rf_test = rf.predict(X_test)
        # calculate rmse
        scores['train'][ii]=np.sqrt(np.mean((y_train-y_rf_train)**2))
        scores['test'][ii]=np.sqrt(np.mean((y_test-y_rf_test)**2))
        ii+=1
    return scores
