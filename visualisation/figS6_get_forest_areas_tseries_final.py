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

#define coordinates and calculate areas
latorig = np.arange(90-1/8.,-90.,-1/4.)
lonorig = np.arange(-180+1/8.,180.,1/4.)
areas = np.zeros([latorig.size,lonorig.size])
res = np.abs(latorig[1]-latorig[0])
for la,latval in enumerate(latorig):
    areas[la]= (6371e3)**2 * ( np.deg2rad(0+res/2.)-np.deg2rad(0-res/2.) ) * (np.sin(np.deg2rad(latval+res/2.))-np.sin(np.deg2rad(latval-res/2.)))

lon2d,lat2d = np.meshgrid(lonorig,latorig)

mask_america = lon2d<-25
mask_africa = (lon2d>-25)*(lon2d<58)
mask_asia  = (lon2d>58)

ext = [-180,180,-90,90]
obsAGB = np.zeros([3,latorig.size,lonorig.size])
potAGB = np.zeros([3,latorig.size,lonorig.size])

for la,lvl in enumerate(['lower','mean','upper']):
    obsAGB[la] = np.load(path2prj+'/npydata/avitabile_%s_secma.npz' % lvl)['obsAGB']
    potAGB[la] = np.load(path2prj+'/npydata/avitabile_%s_secma.npz' % lvl)['potAGB']

#BGB = 0.490*AGB**0.89 of biomass... bgb = (np.nansum(0.490*((2*potAGB[1])**0.89)*areas)*1e-15*1e2)/2.

obsAGB[obsAGB<0] = np.nan
potAGB[potAGB<0] = np.nan

maskAGB = np.isfinite(obsAGB[1])


hist_states = Dataset(path2prj+'/LUH2/states.nc')

years = np.arange(1860,2016)

forest_areas = np.zeros([2,years.size])

for yr,yval in enumerate(years):

    hist_primf = hist_states.variables['primf'][np.arange(850,2016)==yval]
    hist_secdf = hist_states.variables['secdf'][np.arange(850,2016)==yval]

    forest_areas[0,yr] = (areas * maskAGB * hist_primf).sum()*1e-6*1e-6
    forest_areas[1,yr] = (areas * maskAGB * hist_secdf).sum()*1e-6*1e-6

hist_states.close()


years_rcp = np.arange(2015,2101)
forest_areas_rcp = np.zeros([2,6,years_rcp.size])

import glob
scen = ['ssp126','ssp434','ssp245','ssp460','ssp370','ssp585']
scenlong = ['SSP1-2.6','SSP4-3.4','SSP2-4.5','SSP4-6.0','SSP3-7.0','SSP5-8.5']
#scen.sort()

for rr,rcp in enumerate(scen):
    fname = glob.glob(path2prj+'/../LUH2/*%s*' % rcp)[0]
    rcp_states = Dataset(fname)

    for yr,yval in enumerate(years_rcp):
        rcp_primf = rcp_states.variables['primf'][np.arange(2015,2101)==yval]
        rcp_secdf = rcp_states.variables['secdf'][np.arange(2015,2101)==yval]

        forest_areas_rcp[0,rr,yr] = (areas * maskAGB * rcp_primf).sum()*1e-6*1e-6
        forest_areas_rcp[1,rr,yr] = (areas * maskAGB * rcp_secdf).sum()*1e-6*1e-6

    rcp_states.close()




fig = pl.figure('tseries forest area');fig.clf()
pl.plot(years,forest_areas.sum(0),'k-',lw=2, label='Historical')

#colorblind friendly figures
cols = [[230,159,0],
        [86,180,233],
        [0,158,115],
        [240,228,66],
        [0,114,178],
        [213,94,0],
        [204,121,167]]

cols = np.array(cols)/255.

for rr,rcp in enumerate(scen):
    pl.plot(years_rcp,forest_areas_rcp.sum(0)[rr],label=scenlong[rr],color=cols[rr],lw=2)

pl.ylabel('Pantropical forest area [million km$^2$]')
pl.legend(loc='lower left')
pl.xlim(1855,2105)
pl.xlabel('Year')
pl.grid(True,ls=':')
pl.fill_between([2000,2009],[0,0],[24,24],color='silver',edgecolor='silver',zorder=-1)
pl.ylim(12,24)
#fig.show()
fig.savefig('../figures/manuscript/figS6_tseries_forestareas_final.png',bbox_inches='tight')

"""
for rr,rcp in enumerate(['2.6','3.4','4.5','6.0','7.0','8.5']):
    rcp_primf = Dataset(path2prj+'/LUH2/states_RCP%s.nc' % rcp).variables['primf'][slc_yrs].mean(0)
    rcp_secdf = Dataset(path2prj+'/LUH2/states_RCP%s.nc' % rcp).variables['secdf'][slc_yrs].mean(0)

    for mm,msk in enumerate([maskAGB,mask_africa,mask_america,mask_asia]):
        forest_areas_rcp[rr,mm] = (areas * maskAGB * msk * (rcp_primf+rcp_secdf)).sum()*1e-6*1e-6





years = np.arange(2015,2101)
rcp_states = glob.glob('*RCP*nc')
rcp_states.sort()

primf_area = np.zeros([len(rcp_states),years.size])
secdf_area = np.zeros([len(rcp_states),years.size])

for rr,rcp in enumerate(rcp_states):
    print rcp
    primf = Dataset(rcp).variables['primf']
    secdf = Dataset(rcp).variables['secdf']

    for yy, yr in enumerate(years):
        print '\r %04i' % yr
        primf_area[rr,yy] = (areas * maskAGB * primf[yy]).sum()
        secdf_area[rr,yy] = (areas * maskAGB * secdf[yy]).sum()



"""
