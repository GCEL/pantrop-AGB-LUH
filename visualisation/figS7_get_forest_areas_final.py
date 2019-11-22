"""
23/3/17 - JFE
updated to get biomass a function of all
LU classes

22/3/17 - JFE
This script reconstructs ABC as a function of climate
in regions where LUH database indicates a certain level
of primary land in 2001
"""

from sklearn.ensemble import RandomForestRegressor as RF
import numpy as np
from netCDF4 import Dataset
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import sys
import pylab as pl
import gdal
import pandas as pd
from sklearn.externals import joblib

import sys
sys.path.append('../')
from useful import *

path2prj = '/disk/scratch/local.2/jexbraya/potABC_avitabile/'

#primlim = int(sys.argv[2])/100.
#colorblind friendly figures
cols = [[230,159,0],
        [86,180,233],
        [0,158,115],
        [240,228,66],
        [0,114,178],
        [213,94,0],
        [204,121,167]]

cols = np.array(cols)/255.
#define coordinates and calculate areas
latorig = np.arange(90-1/8.,-90.,-1/4.)
lonorig = np.arange(-180+1/8.,180.,1/4.)
areas = np.zeros([latorig.size,lonorig.size])
res = np.abs(latorig[1]-latorig[0])
for la,latval in enumerate(latorig):
    areas[la]= (6371e3)**2 * ( np.deg2rad(0+res/2.)-np.deg2rad(0-res/2.) ) * (np.sin(np.deg2rad(latval+res/2.))-np.sin(np.deg2rad(latval-res/2.)))

lon2d,lat2d = np.meshgrid(lonorig,latorig)

mask_tropics = np.all((lat2d>=-30,lat2d<=30),axis=0)
mask_tropics[403:,1100:]=False # mask out Australia
mask_america = (lon2d<-25)*mask_tropics
mask_africa = (lon2d>-25)*(lon2d<58)*mask_tropics
mask_asia  = (lon2d>58)*mask_tropics
"""
slc_yrs = (np.arange(850,2016)>=1860)*(np.arange(850,2016)<=1869)
hist_primf = Dataset(path2prj+'/LUH2/states.nc').variables['primf'][slc_yrs].mean(0)
hist_secdf = Dataset(path2prj+'/LUH2/states.nc').variables['secdf'][slc_yrs].mean(0)
hist_primn = Dataset(path2prj+'/LUH2/states.nc').variables['primn'][slc_yrs].mean(0)

slc_yrs = (np.arange(850,2016)>=2000)*(np.arange(850,2016)<=2009)
pres_primf = Dataset(path2prj+'/LUH2/states.nc').variables['primf'][slc_yrs].mean(0)
pres_secdf = Dataset(path2prj+'/LUH2/states.nc').variables['secdf'][slc_yrs].mean(0)
pres_primn = Dataset(path2prj+'/LUH2/states.nc').variables['primn'][slc_yrs].mean(0)
"""
slc_yrs = (np.arange(850,2016)==1850)
hist_primf = Dataset(path2prj+'/LUH2/states.nc').variables['primf'][slc_yrs].mean(0)
hist_secdf = Dataset(path2prj+'/LUH2/states.nc').variables['secdf'][slc_yrs].mean(0)
hist_primn = Dataset(path2prj+'/LUH2/states.nc').variables['primn'][slc_yrs].mean(0)

slc_yrs = (np.arange(850,2016)==2015)
pres_primf = Dataset(path2prj+'/LUH2/states.nc').variables['primf'][slc_yrs].mean(0)
pres_secdf = Dataset(path2prj+'/LUH2/states.nc').variables['secdf'][slc_yrs].mean(0)
pres_primn = Dataset(path2prj+'/LUH2/states.nc').variables['primn'][slc_yrs].mean(0)

forest_areas = np.zeros([2,4])
primn_areas = np.zeros([2,4])
primf_areas = np.zeros([2,4])

for mm,msk in enumerate([maskAGB,mask_america,mask_africa,mask_asia]):
    forest_areas[0,mm] = np.nansum(areas * msk * (hist_primf+hist_secdf))*1e-6*1e-6
    forest_areas[1,mm] = np.nansum(areas * msk * (pres_primf+pres_secdf))*1e-6*1e-6

    primn_areas[0,mm] = np.nansum(areas * msk * (hist_primn))*1e-6*1e-6
    primn_areas[1,mm] = np.nansum(areas * msk * (pres_primn))*1e-6*1e-6

    primf_areas[0,mm] = np.nansum(areas * msk * (hist_primf))*1e-6*1e-6
    primf_areas[1,mm] = np.nansum(areas * msk * (pres_primf))*1e-6*1e-6


forest_areas_rcp = np.zeros([6,4])
slc_yrs = (np.arange(2015,2101)>=2091)*(np.arange(2015,2101)<=2100)

scen = ['ssp126','ssp434','ssp245','ssp460','ssp370','ssp585']
#scen.sort()
scenlong = ['SSP1-2.6','SSP4-3.4','SSP2-4.5','SSP4-6.0','SSP3-7.0','SSP5-8.5']
for rr,rcp in enumerate(scen):
    fname = glob.glob(path2prj+'/../LUH2/*%s*.nc' % rcp)[0]
    rcp_primf = Dataset(fname).variables['primf'][slc_yrs].mean(0)
    rcp_secdf = Dataset(fname).variables['secdf'][slc_yrs].mean(0)

    for mm,msk in enumerate([maskAGB,mask_america,mask_africa,mask_asia]):
        forest_areas_rcp[rr,mm] = np.nansum(areas * msk * (rcp_primf+rcp_secdf))*1e-6*1e-6

cols = ['k','k']+list(cols)

fig = pl.figure('bars forest',figsize=(12,9));fig.clf()
titles = ['Pantropical','Americas','Africa','Asia']
for mm,mask in enumerate([mask_tropics,mask_america,mask_africa,mask_asia]):
    ax = fig.add_subplot(2,2,mm+1)

    past = forest_areas[0,mm]
    pres = forest_areas[1,mm]
    rcps = forest_areas_rcp[:,mm]

    frst = [past,pres]+list(rcps)

    ax.bar(np.arange(8),frst,color = cols, align = 'center',edgecolor='k')
    ax.set_ylabel('Forested area [million km$^2$]')
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels(['1850','2015']+scenlong,size='small',rotation = 45)

    ax.tick_params(bottom=False,top=False)
    ax.text(0.97,0.97,chr(ord('a')+mm)+') '+titles[mm],transform=ax.transAxes,weight='bold',va='top',ha='right')

    ax.hlines(pres,ax.get_xlim()[0],ax.get_xlim()[1],linestyles='dashed',colors='gray',linewidths=2)

fig.show()
fig.savefig('../figures/manuscript/figS7_barplots_forests_final.png',bbox_inches='tight')
