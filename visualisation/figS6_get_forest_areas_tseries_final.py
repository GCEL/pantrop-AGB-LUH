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
import xarray as xr
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
mask_america = (lon2d<-25.) & mask_tropics
mask_africa  = (lon2d>-25.) & (lon2d<58) & mask_tropics
mask_asia    = (lon2d>58.) & mask_tropics

hist_states = xr.open_dataset(path2prj+'/LUH2/states.nc',decode_times = False)

years = np.arange(1850,2016)

forest_areas = np.zeros([2,years.size])

for yr,yval in enumerate(years):

    hist_primf = hist_states['primf'].values[np.arange(850,2016)==yval]
    hist_primn = hist_states['primn'].values[np.arange(850,2016)==yval]
    hist_secdf = hist_states['secdf'].values[np.arange(850,2016)==yval]

    forest_areas[0,yr] = np.nansum(areas * mask_tropics * hist_primf)*1e-6*1e-6
    forest_areas[1,yr] = np.nansum(areas * mask_tropics * hist_secdf)*1e-6*1e-6

hist_states.close()


years_rcp = np.arange(2015,2101)
forest_areas_rcp = np.zeros([2,6,years_rcp.size])

scen = ['ssp126','ssp434','ssp245','ssp460','ssp370','ssp585']
scenlong = ['SSP1-2.6','SSP4-3.4','SSP2-4.5','SSP4-6.0','SSP3-7.0','SSP5-8.5']
#scen.sort()

for rr,rcp in enumerate(scen):
    fname = glob.glob(path2prj+'/../LUH2/*%s*' % rcp)[0]
    rcp_states = xr.open_dataset(fname,decode_times=False)

    for yr,yval in enumerate(years_rcp):
        rcp_primf = rcp_states['primf'].values[np.arange(2015,2101)==yval]
        rcp_secdf = rcp_states['secdf'].values[np.arange(2015,2101)==yval]

        forest_areas_rcp[0,rr,yr] = np.nansum(areas * mask_tropics * rcp_primf)*1e-6*1e-6
        forest_areas_rcp[1,rr,yr] = np.nansum(areas * mask_tropics * rcp_secdf)*1e-6*1e-6
    print ('%s\t primary forest change: %.2f\tsecondary forest change: %.2f\t total forest change: %.2f' %
        (rcp,forest_areas_rcp[0,rr,-1]-forest_areas_rcp[0,rr,0],forest_areas_rcp[1,rr,-1]-forest_areas_rcp[1,rr,0],np.sum(forest_areas_rcp[:,rr,-1]-forest_areas_rcp[:,rr,0])))
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
pl.xlim(1845,2105)
pl.xlabel('Year')
pl.grid(True,ls=':')
pl.fill_between([2000,2009],[0,0],[24,24],color='silver',edgecolor='silver',zorder=-1)
pl.ylim(10,22)
#fig.show()
fig.savefig('../figures/manuscript/figS6_tseries_forestareas_final.png',bbox_inches='tight')

"""
# Calculations for Bonn Challenge
"""
pot = xr.open_dataset('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/output/AGB_hist.nc')
forest2015 = forest_areas[0,years==2015]
agb2015 = pot['ts_mean'].values[pot.time==2015]*0.48
bonn_challenge = forest2015+3.5
comparison_year = years[np.argmin(np.abs(forest_areas[0]-bonn_challenge))]
bonn_challenge_carbon =  (pot.ts_mean.values[pot.time==comparison_year] - pot.ts_mean.values[pot.time==2015]) * 0.48
bonn_challenge_carbon_upper =  (pot.ts_upper.values[pot.time==comparison_year] - pot.ts_upper.values[pot.time==2015]) * 0.48
bonn_challenge_carbon_lower =  (pot.ts_lower.values[pot.time==comparison_year] - pot.ts_lower.values[pot.time==2015]) * 0.48

print('2015:\t\t\t\tprimary forest area (10^6 km): %.2f;\tAGB (Pg C): %.2f' % (forest2015,agb2015))
print('%i (Bonn Challenge ref):\tprimary forest area (10^6 km): %.2f;\tAGB (Pg C): %.2f' % (comparison_year,forest_areas[0,years==comparison_year],0.48*pot.ts_mean.values[pot.time==comparison_year]))
print('Bonn Challenge potential (Pg C): %.2f (%.2f,%.2f)' % (bonn_challenge_carbon,bonn_challenge_carbon_lower,bonn_challenge_carbon_upper))

"""
# Forest  area changes summary
"""
hist_states = xr.open_dataset(path2prj+'/LUH2/states.nc',decode_times=False)
years = np.array([1850,2015])#np.arange(1850,2016)
forest_areas = np.zeros([2,years.size])
masks = {'tropics':mask_tropics,'americas':mask_america,'africa  ':mask_africa,'asia   ':mask_asia}
for yr,yval in enumerate(years):
    print('forest extent in %i (10^6 km)' % yval)
    hist_primf = hist_states['primf'].values[np.arange(850,2016)==yval]
    hist_primn = hist_states['primn'].values[np.arange(850,2016)==yval]
    hist_secdf = hist_states['secdf'].values[np.arange(850,2016)==yval]
    for mask in masks:
        AGB_C = np.nansum(areas * masks[mask] * pot.AGB_mean.values[pot.time==yval])/10**13*0.48
        AGB_C_upper = np.nansum(areas * masks[mask] * pot.AGB_upper.values[pot.time==yval])/10**13*0.48
        AGB_C_lower = np.nansum(areas * masks[mask] * pot.AGB_lower.values[pot.time==yval])/10**13*0.48
        primf = np.nansum(areas * masks[mask] * hist_primf)*1e-6*1e-6
        primn = np.nansum(areas * masks[mask] * hist_primn)*1e-6*1e-6
        secdf = np.nansum(areas * masks[mask] * hist_secdf)*1e-6*1e-6
        print('%s;\tprimary forest: %.2f;\tsecondary forest: %.2f;\ttotal forest: %.2f;\tprimary nonforest: %.2f;\tAboveground C: %.2f (%.2f/%.2f)' % (mask,primf,secdf,primf+secdf,primn,AGB_C,AGB_C_lower,AGB_C_upper))

"""
# Correlation of AGB with forest cover in 2015 and 1850
"""
import scipy.stats as stats
years = np.array([1850,2015])
for yr,yval in enumerate(years):

    hist_forest = hist_states['primf'].values[np.arange(850,2016)==yval]+hist_states['secdf'].values[np.arange(850,2016)==yval]
    hist_forest=hist_forest[0][mask_tropics]
    hist_agb = pot['AGB_mean'].values[pot.time==yval][0][mask_tropics]
    mask = np.isfinite(hist_agb)
    print(stats.linregress(hist_agb[mask],hist_forest[mask]))

hist_states.close()
