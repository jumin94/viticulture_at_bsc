# This code evaluates the indices associated with the remote driver responses for ON season

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from numpy import linalg as la
from sklearn.linear_model import LinearRegression
#matplotlib.rcParams['text.usetex'] = True
import utilities

#Open data
import numpy as np
import pandas as pd
import xarray as xr
import utilities.index_module as im
from utilities.index_module import cargo_todo
import os

models = ['ACCESS-CM2','ACCESS-ESM1-5','BCC-CSM2-MR','CAMS-CSM1-0','CanESM5','CESM2_','CESM2-WACCM','CMCC-CM2-SR5',
          'CNRM-CM6-1','CNRM-ESM2-1','EC-Earth3','FGOALS-g3','HadGEM3-GC31-LL','HadGEM3-GC31-MM','IITM-ESM','INM-CM4-8',
          'INM-CM5-0','KACE-1-0-G','MIROC6','MIROC-ES2L','MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0',
          'NESM3','NorESM2-LM','NorESM2-MM','TaiESM1','UKESM1-0-LL']

experiments = ['historical','ssp585']

path = '/datos/julia.mindlin/CMIP6_ensambles/preprocesados' #/historical/mon/tas/past'
os.chdir(path)
os.getcwd()

#October to November------------------------------------------------------------------------------

## Global mean temprature change 
var = 'mon/tas'
variables = ['tas']
dato = cargo_todo(path,experiments,models,var)
season = [10,11]; season_name = 'ON'

GW = im.global_warming_flex_season(dato,season,models,experiments)
GW = {season_name: GW}
GW = pd.DataFrame(GW)
GW.insert(0,"Modelo", models,True)
GW.to_csv('/home/julia.mindlin/Tesis/BSC/indices/GW_index_'+season_name+'.csv',float_format='%g')

del dato

## Tropical Warming
var = 'mon/ta'
variables = ['ta']
dato = cargo_todo(path,experiments,models,var)

TA = im.tropical_warming_flex_season(dato,season,models,experiments) 
TA = {season_name: TA}
TA = pd.DataFrame(TA)
TA.insert(0,"Modelo", models,True)
TA.to_csv('/home/julia.mindlin/Tesis/BSC/indices/TA_index_'+season_name+'.csv',float_format='%g')

del dato 

## Stratospheric Polar Vortex
var = 'mon/ua/50'
variables = ['ua']
dato = cargo_todo(path,experiments,models,var)

SV = im.stratospheric_vortex_flex_season(dato,season,models,experiments) 
SV = {season_name: SV}
SV = pd.DataFrame(SV)
SV.insert(0,"Modelo", models,True)
SV.to_csv('/home/julia.mindlin/Tesis/BSC/indices/SV_index_'+season_name+'.csv',float_format='%g')

## Asymmetric SST changes
#SST changes
var = 'mon/tos'
variables = ['tos']
dato = cargo_todo(path,experiments,models,var)

#Eastern STD 
box = [0,-10,260,290]
ssts = im.changes_list_flex_season(dato,experiments,models,season)
#components, 0: full, 1: symmetric, 2: asymmetric
D_SST_E_std = im.sst_index_asym_change(models,box,ssts)
D_SST_E_std = {season_name: D_SST_E_std}
D_SST_E_std = pd.DataFrame(D_SST_E_std)
D_SST_E_std.insert(0,"Modelo", models,True)
D_SST_E_std.to_csv('/home/julia.mindlin/Tesis/BSC/indices/E_std_asym_index_'+season_name+'2.csv',float_format='%g')

#Central STD
box = [5,-5,180,250]
ssts = im.changes_list_flex_season(dato,experiments,models,season)
D_SST_C_std = im.sst_index_asym_change(models,box,ssts)
D_SST_C_std = {season_name: D_SST_C_std}
D_SST_C_std = pd.DataFrame(D_SST_C_std)
D_SST_C_std.insert(0,"Modelo", models,True)
D_SST_C_std.to_csv('/home/julia.mindlin/Tesis/BSC/indices/C_std_asym_index_'+season_name+'2.csv',float_format='%g')


### Calculamos primero el campo asimétrico de SST y luego calculamos el cambio
## Asymmetric SST changes
#SST changes
var = 'mon/tos'
variables = ['tos']
dato = cargo_todo(path,experiments,models,var)

#Eastern STD 
box = [0,-10,260,290]
ssts = im.changes_list_asym_flex_season(dato,experiments,models,season)
#components, 0: full, 1: symmetric, 2: asymmetric
D_SST_E_std = im.sst_index_asym(models,box,ssts)
D_SST_E_std = {season_name: D_SST_E_std}
D_SST_E_std = pd.DataFrame(D_SST_E_std)
D_SST_E_std.insert(0,"Modelo", models,True)
D_SST_E_std.to_csv('/home/julia.mindlin/Tesis/BSC/indices/E_std_asym_index_'+season_name+'.csv',float_format='%g')

#Central STD
box = [5,-5,180,250]
ssts = im.changes_list_asym_flex_season(dato,experiments,models,season)
D_SST_C_std = im.sst_index_asym(models,box,ssts)
D_SST_C_std = {season_name: D_SST_C_std}
D_SST_C_std = pd.DataFrame(D_SST_C_std)
D_SST_C_std.insert(0,"Modelo", models,True)
D_SST_C_std.to_csv('/home/julia.mindlin/Tesis/BSC/indices/C_std_asym_index_'+season_name+'.csv',float_format='%g')

### To evaluate error in indices------------------------------------------------------------------------------
path = '/datos/julia.mindlin/CMIP6_ensambles/preprocesados' #/historical/mon/tas/past'
os.chdir(path)
os.getcwd()

#Global Warming 
var = 'mon/tas'
variables = ['tas']
dato = cargo_todo(ruta,experiments,models,var)

season = [10,11]; season_name = 'ON'
#ESTOS NUMEROS CONSIDERAN IPSL!!!!!!!!!!!
#ens_members_hist = np.array([2,10,3,2,25,11,3,2,10,5,2,6,4,4,1,1,10,32,1,50,10,10,10,5,5,3,1,1,14])
#ens_members_rcp = np.array([1,3,1,2,25,2,5,1,6,5,4,4,4,4,1,1,1,6,3,3,1,10,10,1,2,1,1,1,2])
ens_members_hist = np.array([2,10,3,2,25,11,3,2,10,5,2,6,4,4,1,1,10,1,50,10,10,10,5,5,3,1,1,14])
ens_members_rcp = np.array([1,3,1,2,25,2,5,1,6,5,4,4,4,4,1,1,1,3,3,1,10,10,1,2,1,1,1,2])

ts_hist, ts_rcp = im.global_warming_time_series_flex_season(dato,season,models,experiments)
gw_mean = np.array([])
gw_error = np.array([])
for i in range(len(models)):
    model = models[i]
    factor_hist = np.sqrt(len(ts_hist[model])*ens_members_hist[i])
    factor_rcp = np.sqrt(len(ts_rcp[model])*ens_members_rcp[i])
    std_hist = np.std(ts_hist[model].values)/factor_hist
    std_rcp = np.std(ts_rcp[model].values)/factor_rcp
    mean = np.mean(ts_rcp[model].values) - np.mean(ts_hist[model].values)
    error = np.sqrt(std_hist**2 + std_rcp**2)
    gw_mean = np.append(gw_mean,mean)
    gw_error = np.append(gw_error,error)

GWerr = {season_name+'_mean':gw_mean,season_name+'_error': gw_error}
GWerr = pd.DataFrame(GWerr)
GWerr.insert(0,"Modelo", models,True)
GWerr.to_csv('/home/julia.mindlin/Tesis/BSC/indices/GW_index_errors_'+season_name+'.csv',float_format='%g')
    
#Tropical warming - timeseries
var = 'mon/ta'
variables = ['ta']
dato = cargo_todo(path,experiments,models,var)

ts_hist, ts_rcp = im.tropical_warming_time_series_flex_season(dato,season,models,experiments)
tw_mean = np.array([])
tw_error = np.array([])
for i in range(len(models)):
    model = models[i]
    factor_hist = np.sqrt(len(ts_hist[model])*ens_members_hist[i])
    factor_rcp = np.sqrt(len(ts_rcp[model])*ens_members_rcp[i])
    std_hist = np.std(ts_hist[model].values)/factor_hist
    std_rcp = np.std(ts_rcp[model].values)/factor_rcp
    mean = np.mean(ts_rcp[model].values) - np.mean(ts_hist[model].values)
    error = np.sqrt(std_hist**2 + std_rcp**2)
    tw_mean = np.append(tw_mean,mean)
    tw_error = np.append(tw_error,error)

TWerr = {season_name+'_mean':tw_mean,season_name+'_error': tw_error}
TWerr = pd.DataFrame(TWerr)
TWerr.insert(0,"Modelo", models,True)
TWerr.to_csv('/home/julia.mindlin/Tesis/BSC/indices/TW_index_errors_'+season_name+'.csv',float_format='%g')


#Sea Surface Temperature changes - timeseries
var = 'mon/tos'
variables = ['tos']
dato = cargo_todo(path,experiments,models,var)

#Open indices
gloW  = pd.read_csv(path_indices+'/GW_index_'+season_name+'.csv')
gw_index = gloW.iloc[:,2].values

#Central Pacific CMIP6 ensemble standard deviation maximum 
box = [5,-5,180,250]
ts_hist, ts_rcp = im.asym_sst_box_time_series_flex_season(dato,season,models,experiments,box)
sst_mean = np.array([])
sst_error = np.array([])
for i in range(len(models)):
    model = models[i]
    factor_hist = np.sqrt(len(ts_hist[model])*ens_members_hist[i])
    factor_rcp = np.sqrt(len(ts_rcp[model])*ens_members_rcp[i])
    std_hist = np.std(ts_hist[model].values)/factor_hist
    std_rcp = np.std(ts_rcp[model].values)/factor_rcp
    mean = np.mean(ts_rcp[model].values) - np.mean(ts_hist[model].values)
    error = np.sqrt(std_hist**2 + std_rcp**2)
    sst_mean = np.append(sst_mean,mean)
    sst_error = np.append(sst_error,error)

SSTerr = {season_name+'_mean':sst_mean,season_name+'_error': sst_error,season_name+'_/K':sst_mean/gw_index}
SSTerr = pd.DataFrame(SSTerr)
SSTerr.insert(0,"Modelo", models,True)
SSTerr.to_csv('/home/julia.mindlin/Tesis/BSC/indices/C_std_asym_index_errors_'+season_name+'.csv',float_format='%g')


#Sea Surface Temperature changes - timeseries
var = 'mon/tos'
variables = ['tos']
dato = cargo_todo(path,experiments,models,var)
#Open indices
gloW  = pd.read_csv(path_indices+'/GW_index_'+season_name+'.csv')
gw_index = gloW.iloc[:,2].values

#Eastern Pacific CMIP6 ensemble standard deviation maximum 
box = [0,-10,260,290]
ts_hist, ts_rcp = im.asym_sst_box_time_series_flex_season(dato,season,models,experiments,box)
sst_mean = np.array([])
sst_error = np.array([])
for i in range(len(models)):
    model = models[i]
    factor_hist = np.sqrt(len(ts_hist[model])*ens_members_hist[i])
    factor_rcp = np.sqrt(len(ts_rcp[model])*ens_members_rcp[i])
    std_hist = np.std(ts_hist[model].values)/factor_hist
    std_rcp = np.std(ts_rcp[model].values)/factor_rcp
    mean = np.mean(ts_rcp[model].values) - np.mean(ts_hist[model].values)
    error = np.sqrt(std_hist**2 + std_rcp**2)
    sst_mean = np.append(sst_mean,mean)
    sst_error = np.append(sst_error,error)

SSTerr = {season_name+'_mean':sst_mean,season_name+'_error': sst_error,season_name+'_/K':sst_mean/gw_index}
SSTerr = pd.DataFrame(SSTerr)
SSTerr.insert(0,"Modelo", models,True)
SSTerr.to_csv('/home/julia.mindlin/Tesis/BSC/indices/E_std_asym_index_errors_'+season_name+'.csv',float_format='%g')

#Stratospheric Polar Vortex - timeseries
var = 'mon/ua/50'
variables = ['ua']
dato = cargo_todo(path,experiments,models,var)

ts_hist, ts_rcp = im.stratospheric_vortex_time_series_flex_season(dato,season,models,experiments)
sv_mean = np.array([])
sv_error = np.array([])
for i in range(len(models)):
    model = models[i]
    factor_hist = np.sqrt(len(ts_hist[model])*ens_members_hist[i])
    factor_rcp = np.sqrt(len(ts_rcp[model])*ens_members_rcp[i])
    std_hist = np.std(ts_hist[model].values)/factor_hist
    std_rcp = np.std(ts_rcp[model].values)/factor_rcp
    mean = np.mean(ts_rcp[model].values) - np.mean(ts_hist[model].values)
    error = np.sqrt(std_hist**2 + std_rcp**2)
    sv_mean = np.append(sv_mean,mean)
    sv_error = np.append(sv_error,error)

SVerr = {season_name+'_mean':sv_mean,season_name+'_error': sv_error}
SVerr = pd.DataFrame(SVerr)
SVerr.insert(0,"Modelo", models,True)
SVerr.to_csv('/home/julia.mindlin/Tesis/BSC/indices/SV_index_errors_'+season_name+'.csv',float_format='%g')

