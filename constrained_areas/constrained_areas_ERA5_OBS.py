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

#Temperature growing season
years = np.arange(1981,2016,1)
months = ['01','02','03','04','10','11','12']
#Temperature
tas_day = xr.open_dataset(path_tas+'/tas_198101.nc').sel(lat=slice(0,-90)).sel(lon=slice(260,330)).tas
for year in years:
    for month in months:
        tas_day_aux = xr.open_dataset(path_tas+'/tas_'+str(year)+month+'.nc').sel(lat=slice(0,-90)).sel(lon=slice(260,330)).tas
        tas_day = xr.merge([tas_day,tas_day_aux])


#Precipitation
months = ['01','02','03','04','05','06','07','08','09','10','11','12']
pr_month = xr.open_dataset(path_pr+'/prlr_198101.nc').sel(latitude=slice(0,-90)).sel(longitude=slice(260,330))*3600*1000
for year in years:
    for month in months:
        pr_month_aux = xr.open_dataset(path_pr+'/prlr_'+str(year)+month+'.nc').sel(latitude=slice(0,-90)).sel(longitude=slice(260,330))*1000*3600
        pr_month = xr.merge([pr_month,pr_month_aux])

#Create masks for constraints in present climate - not suitable areas
#Mask based on temperatures will reject areas were temperatures in july reach less than -15
tas_min = tas_day_july.min(dim=['time']) 
tas_mask_winter = tas_min.where(tas_min > -15)

#Create masks for constraints in present climate - not suitable areas
#Mask based on temperatures will reject areas were growing season are between thresholds
tas_min = tas_day.min(dim=['time']) 
tas_mask_min = tas_min.where(tas_min > 13.1)
tas_mask_gs = tas_mask_min.where(tas_min < 20.9)

#Create masks for constraints in present climate - not suitable areas
#Mask based on temperatures will reject areas were temperatures in july reach less than -15
pr_year = pr_month.groupby('time.year').sum()
pr_annual_mean = pr_year.mean(dim='year')
pr_mask = pr_annual_mean.where(pr_annual_mean >= 300)
pr_mask = pr_mask.where(pr_mask <= 3000)

#Genero un plot para tener alguna idea
fig = plt.figure()
plt.contourf(tas_mask_winter.lon,tas_mask_winter.lat,tas_mask_winter.values)
plt.colorbar()
plt.savefig('/home/Earth/jmindlin/viticulture_at_bsc/constrained_areas/mask_tas_winter.png')
plt.close()

#Genero un plot para tener alguna idea
fig = plt.figure()
plt.contourf(tas_mask_gs.lon,tas_mask_gs.lat,tas_mask_gs.values)
plt.colorbar()
plt.savefig('/home/Earth/jmindlin/viticulture_at_bsc/constrained_areas/mask_tas_growing_season.png')
plt.close()


#Genero un plot para tener alguna idea
fig = plt.figure()
plt.contourf(pr_mask.longitude,pr_mask.latitude,pr_mask.prlr.values)
plt.colorbar()
plt.savefig('/home/Earth/jmindlin/viticulture_at_bsc/constrained_areas/mask_pr.png')
plt.close()

