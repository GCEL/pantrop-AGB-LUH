"""
14/11/2018 - JFE
This files loads the fitted models and produces annual AGB maps for the 21st
century
"""

from useful import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import pandas as pd
import glob

pca = joblib.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/pca_pipeline.pkl')

#load the fitted rfs
rf_med = joblib.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rfbc_mean.pkl')
rf_upp = joblib.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rfbc_upper.pkl')
rf_low = joblib.load('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/saved_algorithms/rfbc_lower.pkl')

#iterate over years
years = np.arange(2015,2101)
lat = np.arange(90-0.125,-90,-0.25)
lon = np.arange(-180+0.125,180,0.25)

# get tropics mask (20N-30S)
lon2d,lat2d = np.meshgrid(lon,lat)
tropics_mask = np.all((lat2d>=-30,lat2d<=30),axis=0)

#get areas
areas = get_areas()

luh_files = sorted(glob.glob('/disk/scratch/local.2/jexbraya/LUH2/*ssp*'))
for luh_file in luh_files:
    ssp = luh_file.split('-')[-5]
    print(luh_file,ssp)

    for yy, year in enumerate(years):
        predictors,landmask = get_predictors(y0=year,luh_file=luh_file)
        # get region masks
        continents = get_continents(landmask)
        continents = continents[landmask].reshape(landmask.sum(),1)

        #transform the data
        X = pca.transform(predictors)
        X = np.hstack((X,continents))

        if yy == 0:
            #create coordinates
            coords = {'time': (['time'],years.astype('d'),{'units':'year'}),
                      'lat': (['lat'],lat,{'units':'degrees_north','long_name':'latitude'}),
                      'lon': (['lon'],lon,{'units':'degrees_east','long_name':'longitude'})}

            #create empty variable to store results
            attrs={'_FillValue':-9999.,'units':'Mg ha-1'}
            data_vars = {}
            for lvl in ['mean','upper','lower']:
                data_vars['AGB_'+lvl] = (['time','lat','lon'],np.zeros([years.size,lat.size,lon.size])*np.nan,attrs)
                data_vars['ts_' + lvl] = (['time'],np.zeros(years.size),{'units': 'Pg','long_name': 'total AGB'})
            agb_ssp = xr.Dataset(data_vars=data_vars,coords=coords)

        agb_ssp.AGB_mean[yy].values[landmask]  = rfbc_predict(rf_med['rf1'],rf_med['rf2'],X)
        agb_ssp.AGB_upper[yy].values[landmask] = rfbc_predict(rf_upp['rf1'],rf_upp['rf2'],X)
        agb_ssp.AGB_lower[yy].values[landmask] = rfbc_predict(rf_low['rf1'],rf_low['rf2'],X)

        # clip to tropics
        agb_ssp.AGB_mean[yy].values[~tropics_mask]  = np.nan
        agb_ssp.AGB_upper[yy].values[~tropics_mask] = np.nan
        agb_ssp.AGB_lower[yy].values[~tropics_mask] = np.nan

        # set negative estimates to zero
        agb_ssp.AGB_mean[yy].values[agb_ssp.AGB_mean[yy].values<0]  = 0
        agb_ssp.AGB_upper[yy].values[agb_ssp.AGB_upper[yy].values<0] = 0
        agb_ssp.AGB_lower[yy].values[agb_ssp.AGB_lower[yy].values<0] = 0

        #get time series
        agb_ssp.ts_mean[yy] = (agb_ssp.AGB_mean[yy].values*areas)[landmask*tropics_mask].sum()*1e-13
        agb_ssp.ts_upper[yy] = (agb_ssp.AGB_upper[yy].values*areas)[landmask*tropics_mask].sum()*1e-13
        agb_ssp.ts_lower[yy] = (agb_ssp.AGB_lower[yy].values*areas)[landmask*tropics_mask].sum()*1e-13

    #save to a nc file
    encoding = {'AGB_mean':{'zlib':True,'complevel':1},
                'AGB_upper':{'zlib':True,'complevel':1},
                'AGB_lower':{'zlib':True,'complevel':1},}

    agb_ssp.to_netcdf('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/output/AGB_%s.nc' % ssp,encoding=encoding)
