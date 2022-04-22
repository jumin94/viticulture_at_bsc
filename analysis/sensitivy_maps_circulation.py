# Circulation response to remote drivers

# Imports ------------------------
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from numpy import linalg as la
from sklearn.linear_model import LinearRegression
#matplotlib.rcParams['text.usetex'] = True
import utilities
import numpy as np
import pandas as pd
import xarray as xr
import utilities.index_module as im
from utilities.index_module import cargo_todo
import os
# My functions-----------------------
import open_data
import regression
import csv2nc

# Data loading --------------------
models = [
    'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0',
    'CanESM5', 'CESM2_', 'CESM2-WACCM','CMCC-CM2-SR5','CNRM-CM6-1',
    'CNRM-ESM2-1','EC-Earth3', 'FGOALS-g3', 'HadGEM3-GC31-LL','HadGEM3-GC31-MM',
    'IITM-ESM','INM-CM4-8','INM-CM5-0','KACE-1-0-G',
    'MIROC6','MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
    'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'TaiESM1','UKESM1-0-LL'
    ]

ruta = '/pikachu/datos/julia.mindlin/CMIP6_ensambles/preprocesados'
scenarios = ['historical','ssp585']
os.chdir(ruta)
os.getcwd()

import fnmatch
def cargo_todo(scenarios,models,ruta,var):
    os.chdir(ruta)
    os.getcwd()
    dic = {}
    dic['historical'] = {}
    dic['ssp585'] = {}
    for scenario in dic.keys():
        listOfFiles = os.listdir(ruta+'/'+scenario+'/'+var)
        for model in models:
            dic[scenario][model] = {}
            if scenario == 'ssp585':
                periods = ['2070-2099']
            else:
                periods = ['1940-1969'] #1940-1969
            for period in periods:
                dic[scenario][model][period] = []
                pattern1 = "*"+model+"*"+scenario+"*"+"1950-1979"+"*T42*"
                pattern2 = "*"+model+"*"+scenario+"*"+"2070-2099"+"*T42*"
                for entry in listOfFiles:
                    if fnmatch.fnmatch(entry,pattern1):
                        dato = xr.open_dataset(ruta+'/'+scenario+'/'+var+'/'+entry)
                        dic[scenario][model][period].append(dato)
                    elif fnmatch.fnmatch(entry,pattern2):
                        dato = xr.open_dataset(ruta+'/'+scenario+'/'+var+'/'+entry)
                        dic[scenario][model][period].append(dato)
    return dic


#October - November-------------------------------------------
#Remote driver index loading
path_indices = '/home/julia.mindlin/Tesis/BSC/indices'
#Open indices for ON season
season_name = 'ON'
GW  = pd.read_csv(path_indices+'/GW_index_'+season_name+'.csv')
gw_ON = GW.iloc[:,2].values
E_index  = pd.read_csv(path_indices+'/E_std_asym_index_'+season_name+'.csv')
e_index_ON = E_index.iloc[:,2]
C_index = pd.read_csv(path_indices+'/C_std_asym_index_'+season_name+'.csv')
c_index_ON = C_index.iloc[:,2]
TW = pd.read_csv(path_indices+'/TA_index_'+season_name+'.csv')
tw_ON = TW.iloc[:,2]
SV = pd.read_csv(path_indices+'/SV_index_'+season_name+'.csv')
sv_ON = SV.iloc[:,2]

#Wind at 850hPa
var = 'mon/ua/850'
dato_ua = open_data.cargo_todo(scenarios,models,ruta,var)
indices = [tw_ON,sv_ON,c_index_ON,e_index_ON] 
indices_names = ['TA','VB','SST_central','SST_east'] 
var = 'ua'
#Create regression class
reg = across_models()
#Generate regression data
# If season is DJF,MAM,JJA,SON than full_season = 'yes', if not , full_season = 'no'
season = [10,11]
reg.regression_data(dato_ua,scenarios,models,gw_ON,season,var,'yes','no')
#Create sensitivity maps
path_maps = '/home/julia.mindlin/Tesis/BSC/sensitivity_maps/ON'
os.chdir(path_maps)
os.getcwd()
os.makedirs(var,exist_ok=True)
os.chdir(path_maps+'/'+var)
reg.perform_regression(indices,indices_names,gw_ON,path_maps+'/'+var)
file_list = csv2nc.csv_to_nc(path_maps+'/'+var)

#Geopotential height
var = 'mon/zg'
dato_ua = cargo_todo(scenarios,models,ruta,var)
indices = [tw_ON,sv_ON,c_index_ON,e_index_ON] 
indices_names = ['TA','VB','SST_central','SST_east'] 
var = 'zg'
#Create regression class
reg = across_models()
#Generate regression data
season = [10,11]
reg.regression_data(dato_ua,scenarios,models,gw_ON,season,var,'yes','no')
#Create sensitivity maps
path_maps = '/home/julia.mindlin/Tesis/BSC/sensitivity_maps/ON'
os.chdir(path_maps)
os.getcwd()
os.makedirs(var,exist_ok=True)
os.chdir(path_maps+'/'+var)
reg.perform_regression(indices,indices_names,gw_ON,path_maps+'/'+var)
file_list = csv2nc.csv_to_nc(path_maps+'/'+var)


# December - Febuary -----------------------------------------
#Remote driver index loading
path_indices = '/home/julia.mindlin/Tesis/BSC/indices'
#Open indices for MA season
season_name = 'DJF'
GW  = pd.read_csv(path_indices+'/GW_index_'+season_name+'.csv')
gw_DJF = GW.iloc[:,2].values
E_index  = pd.read_csv(path_indices+'/E_std_asym_index_'+season_name+'.csv')
e_index_DJF = E_index.iloc[:,2]
C_index = pd.read_csv(path_indices+'/C_std_asym_index_'+season_name+'.csv')
c_index_DJF = C_index.iloc[:,2]
TW = pd.read_csv(path_indices+'/TW_index_'+season_name+'.csv')
tw_DJF = TW.iloc[:,2]
VB = pd.read_csv(path_indices+'/VB_regresion_coef_all_models.csv')
vb =  VB.iloc[:,2]
#Open index errors for ON season
GW  = pd.read_csv(path_indices+'/GW_index_errors_'+season_name+'.csv')
gw_index_errors_DJF = GW.iloc[:,3].values
E_index  = pd.read_csv(path_indices+'/E_std_asym_index_errors_'+season_name+'.csv')
e_index_errors_DJF = E_index.iloc[:,3]
C_index = pd.read_csv(path_indices+'/C_std_asym_index_errors_'+season_name+'.csv')
c_index_errors_DJF = C_index.iloc[:,3]
TW = pd.read_csv(path_indices+'/TW_index_errors_'+season_name+'.csv')
tw_errors_DJF = TW.iloc[:,3]
VB = pd.read_csv(path_indices+'/VB_regresion_coef_all_models.csv')
vb_errors =  VB.iloc[:,4]

# Geopotential Height
var = 'mon/ua/850'
dato_ua = open_data.cargo_todo(scenarios,models,ruta,var)
indices = [tw_DJF,vb,c_index_DJF,e_index_DJF] 
indices_names = ['TA','VB','SST_central','SST_east'] 
var = 'ua'
#Create regression class
reg = across_models()
#Generate regression data
season = 'DJF'
reg.regression_data(dato_ua,scenarios,models,gw_DJF,season,var,'yes','yes')
#Create sensitivity maps
path_maps = '/home/julia.mindlin/Tesis/BSC/sensitivity_maps/DJF'
os.chdir(path_maps)
os.getcwd()
os.makedirs(var,exist_ok=True)
os.chdir(path_maps+'/'+var)
reg.perform_regression(indices,indices_names,gw_DJF,path_maps+'/'+var)
file_list = csv2nc.csv_to_nc(path_maps+'/'+var)

# Wind at 850 hPa
#Create dictionary
var = 'mon/zg'
dato_ua = open_data.cargo_todo_zg(scenarios,models,ruta,var)
indices = [tw_DJF,vb,c_index_DJF,e_index_DJF] 
indices_names = ['TA','VB','SST_central','SST_east'] 
var = 'zg'
#Create regression class
reg = across_models()
#Generate regression data
season = 'DJF'
reg.regression_data(dato_ua,scenarios,models,gw_DJF,season,var,'yes','yes')
#Create sensitivity maps
path_maps = '/home/julia.mindlin/Tesis/BSC/sensitivity_maps/DJF'
os.chdir(path_maps)
os.getcwd()
os.makedirs(var,exist_ok=True)
os.chdir(path_maps+'/'+var)
reg.perform_regression(indices,indices_names,gw_DJF,path_maps+'/'+var)
file_list = csv2nc.csv_to_nc(path_maps+'/'+var)


#March - April -----------------------------------------------------
#Remote Driver index loading
path_indices = '/home/julia.mindlin/Tesis/BSC/indices'
#Open indices for MA season
season_name = 'MA'
GW  = pd.read_csv(path_indices+'/GW_index_'+season_name+'.csv')
gw_MA = GW.iloc[:,2].values
E_index  = pd.read_csv(path_indices+'/E_std_asym_index_'+season_name+'.csv')
e_index_MA = E_index.iloc[:,2]
C_index = pd.read_csv(path_indices+'/C_std_asym_index_'+season_name+'.csv')
c_index_MA = C_index.iloc[:,2]
TW = pd.read_csv(path_indices+'/TA_index_'+season_name+'.csv')
tw_MA = TW.iloc[:,2]
SV = pd.read_csv(path_indices+'/SV_index_'+season_name+'.csv')
sv_MA = SV.iloc[:,2]
#Open index errors for ON season
GW  = pd.read_csv(path_indices+'/GW_index_errors_'+season_name+'.csv')
gw_index_errors_MA = GW.iloc[:,3].values
E_index  = pd.read_csv(path_indices+'/E_std_asym_index_errors_'+season_name+'.csv')
e_index_errors_MA = E_index.iloc[:,3]
C_index = pd.read_csv(path_indices+'/C_std_asym_index_errors_'+season_name+'.csv')
c_index_errors_MA = C_index.iloc[:,3]
TW = pd.read_csv(path_indices+'/TW_index_errors_'+season_name+'.csv')
tw_errors_MA = TW.iloc[:,3]
SV = pd.read_csv(path_indices+'/SV_index_errors_'+season_name+'.csv')
sv_errors_MA = SV.iloc[:,3]

#Wind 850hPa
var = 'mon/ua/850'
dato_ua = open_data.cargo_todo(scenarios,models,ruta,var)
indices = [tw_MA,sv_MA,c_index_MA,e_index_MA] 
indices_names = ['TA','VB','SST_central','SST_east'] 
var = 'ua'
#Create regression class
reg = across_models()
#Generate regression data
season = [3,4]
reg.regression_data(dato_ua,scenarios,models,gw_MA,season,var,'yes','no')
#Create sensitivity maps
path_maps = '/home/julia.mindlin/Tesis/BSC/sensitivity_maps/MA'
os.chdir(path_maps)
os.getcwd()
os.makedirs(var,exist_ok=True)
os.chdir(path_maps+'/'+var)
reg.perform_regression(indices,indices_names,gw_MA,path_maps+'/'+var)
file_list = csv2nc.csv_to_nc(path_maps+'/'+var)


#Geopotential Height 500hPa
var = 'mon/zg'
dato_ua = open_data.cargo_todo_zg(scenarios,models,ruta,var)
indices = [tw_MA,sv_MA,c_index_MA,e_index_MA] 
indices_names = ['TA','VB','SST_central','SST_east'] 
var = 'zg'
#Create regression class
reg = across_models()
#Generate regression data
season = [3,4]
reg.regression_data(dato_ua,scenarios,models,gw_MA,season,var,'yes','no')
#Create sensitivity maps
path_maps = '/home/julia.mindlin/Tesis/BSC/sensitivity_maps/MA'
os.chdir(path_maps)
os.getcwd()
os.makedirs(var,exist_ok=True)
os.chdir(path_maps+'/'+var)
reg.perform_regression(indices,indices_names,gw_MA,path_maps+'/'+var)
file_list = csv2nc.csv_to_nc(path_maps+'/'+var)