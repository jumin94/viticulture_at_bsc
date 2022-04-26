#Define constrained areas in ERA5

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

#Open data
path_tas = '/esarchive/recon/ecmwf/era5land/daily_mean/tas_f1h'
path_pr = '/esarchive/obs/noaa/cpc-global-precip/daily_mean/prlr_s0-24h'

#Time
years = np.arange(1981,2016,1)
months = ['01','02','03','04','05','06','07','08','09','10','11','12']
#Temperature
tas_day = xr.open_dataset(path_tas+'/tas_198101.nc').sel(lat=slice(0,-90)).sel(lon=slice(260,330)).tas
for year in years:
    for month in months:
        tas_day_aux = xr.open_dataset(path_tas+'/tas_'+str(year)+month+'.nc').sel(lat=slice(0,-90)).sel(lon=slice(260,330)).tas
        tas_day = xr.merge([tas_day,tas_day_aux])

#Temperature Winter
years = np.arange(1982,2016,1)
tas_day_july = xr.open_dataset(path_tas+'/tas_198107.nc').sel(lat=slice(0,-90)).sel(lon=slice(260,330)).tas
for year in years:
    tas_day_aux = xr.open_dataset(path_tas+'/tas_'+str(year)+'07.nc').sel(lat=slice(0,-90)).sel(lon=slice(260,330)).tas
    tas_day_july = xr.merge([tas_day_july,tas_day_aux])


#Precipitation
prlr_201107.nc
pr_day = xr.open_dataset(path_pr+'/prlr_198101.nc').sel(lat=slice(0,-90)).sel(lon=slice(260,330))
for year in years:
    for month in months:
        pr_day_aux = xr.open_dataset(path_pr+'/prlr_'+str(year)+month+'.nc').sel(lat=slice(0,-90)).sel(lon=slice(260,330))
        pr_day = xr.merge([pr_day,pr_day_aux])

#Create masks for constraints in present climate - not suitable areas
#Mask based on temperatures will reject areas were temperatures in july reach less than -15
tas_min = tas_day_july.min(dim=['time']) 
tas_mask = tas_min.where(tas_min > -15)

#Genero un plot para tener alguna idea
fig = plt.figure()
plt.contourf(tas_mas)
plt.savefig('/home/Earth/jmindlin/mask_tas.png')
