"""
#Plot AGB distributions for 1850 and 2015, and reference data for 2000s
# DTM 20/11/19
"""
import numpy as np
import xarray as xr
from sklearn.neighbors import KernelDensity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#colorblind friendly figures
cols = [[86,180,233],
        [230,159,0]]
cols = np.array(cols)/255.

pot = xr.open_dataset('/disk/scratch/local.2/dmilodow/pantrop_AGB_LUH/output/AGB_hist.nc')
pot['AGB_mean'].values[pot['AGB_mean'].values==-9999]=np.nan
years = np.arange(1850,2016)
agb1850 = pot['AGB_mean'].values[pot.time==1850]*0.48
agb2015 = pot['AGB_mean'].values[pot.time==2015]*0.48
agbref = xr.open_rasterio('/disk/scratch/local.2/jexbraya/AGB/Avitable_AGB_Map_0.25d.tif').values
agbref[agbref==-9999]=np.nan
agbref*=0.48

lon2d,lat2d = np.meshgrid(pot.lon,pot.lat)

mask_tropics = np.all((lat2d>=-30,lat2d<=30),axis=0)
mask_tropics[403:,1100:]=False # mask out Australia
mask_america = (lon2d<-25.) & mask_tropics
mask_africa  = (lon2d>-25.) & (lon2d<58) & mask_tropics
mask_asia    = (lon2d>58.) & mask_tropics

dist2015={}
dist1850={}
distref={}
labels=['tr','am','af','as']
grid = np.linspace(-49.5,349.5,400)
for mm,mask in enumerate([mask_tropics,mask_america,mask_africa,mask_asia]):
    kde2015 = KernelDensity(bandwidth=2,kernel='gaussian')
    kde2015.fit(agb2015[np.isfinite(agb2015)*mask][:,None])
    dist2015[labels[mm]] = np.exp(kde2015.score_samples(grid[:,None]))
    kde1850 = KernelDensity(bandwidth=2,kernel='gaussian')
    kde1850.fit(agb1850[np.isfinite(agb1850)*mask][:,None])
    dist1850[labels[mm]] = np.exp(kde1850.score_samples(grid[:,None]))
    kderef = KernelDensity(bandwidth=2,kernel='gaussian')
    kderef.fit(agbref[np.isfinite(agbref)*mask][:,None])
    distref[labels[mm]] = np.exp(kderef.score_samples(grid[:,None]))

    # apply reflective lower boundary at 0
    n_neg = np.sum(grid<0)
    dist1850[labels[mm]][n_neg:2*n_neg]+=dist1850[labels[mm]][:n_neg][::-1]
    dist1850[labels[mm]][:n_neg]=np.nan
    dist2015[labels[mm]][n_neg:2*n_neg]+=dist2015[labels[mm]][:n_neg][::-1]
    dist2015[labels[mm]][:n_neg]=np.nan
    distref[labels[mm]][n_neg:2*n_neg]+=distref[labels[mm]][:n_neg][::-1]
    distref[labels[mm]][:n_neg]=np.nan

fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(6,5),sharex=True,sharey=True)

titles = ['Pantropical','Americas','Africa','Asia']
for mm,mask in enumerate([mask_tropics,mask_america,mask_africa,mask_asia]):

    if mm==0:
        ax = axes[0,0]
    elif mm==1:
        ax=axes[0,1]
    elif mm==2:
        ax=axes[1,0]
    elif mm==3:
        ax=axes[1,1]

    ax.fill_between(grid,0,distref[labels[mm]],color='silver',edgecolor='silver',zorder=-1)
    ax.plot(grid,dist2015[labels[mm]],'-',color=cols[1],linewidth=2,label='2015')
    ax.plot(grid,dist1850[labels[mm]],'--',color=cols[0],linewidth=2,label='1850')
    ax.set_ylim(bottom=0,top=0.04)
    ax.set_xlim(left=0,right=250)
    if mm == 2 or mm ==3:
        ax.set_xlabel('AGB [Mg C ha$^{-1}$]')
    if mm == 0 or mm ==2:
        ax.set_ylabel('Frequency density')
    ax.text(0.97,0.97,chr(ord('a')+mm)+') '+titles[mm],transform=ax.transAxes,weight='bold',va='top',ha='right')
    if mm==3:
        ax.legend(loc = 'right')
fig.tight_layout()
fig.show()
fig.savefig('../figures/manuscript/figS8_agb_distributions.png',bbox_inches='tight')
