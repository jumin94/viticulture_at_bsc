# Genera la clase across_models que genera una regresion across models
#Imports
import numpy as np
import pandas as pd
import xarray as xr
import math
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import os, fnmatch
import glob
import csv2nc
import metpy

def cross_year_season(month,season):
    #Season is a list with two values, begining and endig season
    return (month >= season[0]) & (month <= season[1])


#Across models regression class

class across_models(object):
    def __init__(self):
        self.what_is_this = 'This performs a regression across models and plots everything'
    
    def regression_data(self,data_dic,scenarios,models,gw_index,season,var,levs='no',full_season='yes'):

        #----------------------------------------
        variable = []

        for i in range(len(models)):
            print(models[i])
            hist = data_dic[scenarios[0]][models[i]]['1940-1969'][0]
            rcp585 = data_dic[scenarios[1]][models[i]]['2070-2099'][0]
            if var == 'pr':
                h = hist.pr
                h.attrs = hist.pr.attrs
                rcp = rcp585.pr 
                rcp.attrs = rcp585.pr.attrs
            elif var == 'tas':
                h = hist.tas
                h.attrs = hist.tas.attrs
                rcp = rcp585.tas 
                rcp.attrs = rcp585.tas.attrs
            elif var == 'ua':
                h = hist.ua
                h.attrs = hist.ua.attrs
                rcp = rcp585.ua
                rcp.attrs = rcp585.ua.attrs
            elif var == 'psl':
                h = hist.psl
                h.attrs = hist.psl.attrs
                rcp = rcp585.psl
                rcp.attrs = rcp585.psl.attrs
            elif var == 'zg':
                h = hist.zg
                h.attrs = hist.zg.attrs
                rcp = rcp585.zg
                rcp.attrs = rcp585.zg.attrs
            
            if full_season == 'yes':
                seasonal_h = h.groupby('time.season').mean(dim='time')
                seasonal_h = seasonal_h.sel(season=season) #.isel(season=0)
            else:
                seasonal_h = h.sel(time=cross_year_season(h['time.month'],season)).mean(dim='time')
            h1_seasonal = seasonal_h.sel(lat=slice(90,-90))
            if levs == 'no':
                a = 0
            else:
                h1_seasonal = h1_seasonal.isel(plev=0)
            h1_seasonal.attrs = h.attrs
            if full_season == 'yes':
                seasonal_r = rcp.groupby('time.season').mean(dim='time')
                seasonal_r = seasonal_r.sel(season=season)#.isel(season=0)
            else:
                seasonal_r = rcp.sel(time=cross_year_season(rcp['time.month'],season)).mean(dim='time')
            r1_seasonal = seasonal_r.sel(lat=slice(90,-90))
            if levs == 'no':
                a = 0
            else:
                r1_seasonal = r1_seasonal.isel(plev=0)
            r1_seasonal.attrs = rcp.attrs
            var_change = (r1_seasonal - h1_seasonal)/gw_index[i]
            variable.append(var_change)
            
        self.psl_change = variable

    def regression_data_div(self,data_dic_ua,data_dic_va,scenarios,models,gw_index,season,var,levs='yes',full_season='yes'):

        #----------------------------------------
        psl = []

        for i in range(len(models)):
            print(models[i])
            hist_ua = data_dic_ua[scenarios[0]][models[i]]['1940-1969'][0]
            rcp585_ua = data_dic_ua[scenarios[1]][models[i]]['2070-2099'][0]
            hist_va = data_dic_va[scenarios[0]][models[i]]['1940-1969'][0]
            rcp585_va = data_dic_va[scenarios[1]][models[i]]['2070-2099'][0]
            h_ua = hist_ua.ua
            h_ua.attrs = hist_ua.ua.attrs
            rcp_ua = rcp585_ua.ua
            rcp_ua.attrs = rcp585_ua.ua.attrs
            h_va = hist_va.va
            h_va.attrs = hist_va.va.attrs
            rcp_va = rcp585_va.va
            rcp_va.attrs = rcp585_va.va.attrs

            #Calculo divergencia climatologica en pasado
            seasonal_h_ua = h_ua.groupby('time.season').mean(dim='time')
            DJF_h_ua = seasonal_h_ua.sel(season=season)
            h1DJF_ua = DJF_h_ua.sel(lat=slice(90,-90))
            if levs == 'no':
                a = 0
            else:
                h1DJF_ua = h1DJF_ua.isel(plev=0)
            h1DJF_ua.attrs = h_ua.attrs
            seasonal_h_va = h_va.groupby('time.season').mean(dim='time')
            DJF_h_va = seasonal_h_va.sel(season=season)
            h1DJF_va = DJF_h_va.sel(lat=slice(90,-90))
            if levs == 'no':
                a = 0
            else:
                h1DJF_va = h1DJF_va.isel(plev=0)
            h1DJF_va.attrs = h_va.attrs
            h1DJF_div = metpy.calc.divergence(h1DJF_ua,h1DJF_va)
            
            #Calculo divergencia climatologica en futuro 
            seasonal_r_ua = rcp_ua.groupby('time.season').mean(dim='time')
            DJF_r_ua = seasonal_r_ua.sel(season=season)
            r1DJF_ua = DJF_r_ua.sel(lat=slice(90,-90))
            if levs == 'no':
                a = 0
            else:
                r1DJF_ua = r1DJF_ua.isel(plev=0)
            r1DJF_ua.attrs = rcp_ua.attrs
            seasonal_r_va = rcp_va.groupby('time.season').mean(dim='time')
            DJF_r_va = seasonal_r_va.sel(season=season)
            r1DJF_va = DJF_r_va.sel(lat=slice(90,-90))
            if levs == 'no':
                a = 0
            else:
                r1DJF_va = r1DJF_va.isel(plev=0)
            r1DJF_va.attrs = rcp_va.attrs
            r1DJF_div = metpy.calc.divergence(r1DJF_ua,r1DJF_va)
            
            #Calculo el cambio en el campo de divergencia total
            div_change = (r1DJF_div -h1DJF_div)/gw_index[i]
            psl.append(div_change)
            
        self.psl_change = psl

    
    
    def perform_regression(self,regressors,rd_names,gw_index,path): 
        
        #regressors  es un dataframe
        #rd_names es una lista de nombres para los archivos ['Aij','VBij',...]
        rd_num = len(rd_names)
        sensitivity_maps = {}
        pvalue_maps = {}
        for i in range(rd_num):
            sensitivity_maps[i] = pd.DataFrame(columns=['a','lat','lon'])
            pvalue_maps[i] = pd.DataFrame(columns=['at','lat','lon'])
        R2ij = pd.DataFrame(columns=['r2','lat','lon'])
        x = np.array([])
    
        #Regresion lineal
        y = sm.add_constant(regressors.values)
        lat = self.psl_change[0].lat
        lon = self.psl_change[0].lon
        reg = linear_model.LinearRegression()
        
        pr = self.psl_change
        for i in range(len(lat)):
            for j in range(len(lon)):
                if np.isnan(pr[0][i-1,j-1].values) or np.isnan(pr[1][i-1,j-1].values) or np.isnan(pr[2][i-1,j-1].values) or np.isnan(pr[3][i-1,j-1].values) or np.isnan(pr[4][i-1,j-1].values) or np.isnan(pr[5][i-1,j-1].values) or np.isnan(pr[6][i-1,j-1].values) or np.isnan(pr[7][i-1,j-1].values) or np.isnan(pr[8][i-1,j-1].values) or np.isnan(pr[9][i-1,j-1].values) or np.isnan(pr[10][i-1,j-1].values) or np.isnan(pr[11][i-1,j-1].values) or np.isnan(pr[12][i-1,j-1].values) or np.isnan(pr[13][i-1,j-1].values) or np.isnan(pr[14][i-1,j-1].values) or np.isnan(pr[15][i-1,j-1].values) or np.isnan(pr[16][i-1,j-1].values) or np.isnan(pr[17][i-1,j-1].values) or np.isnan(pr[18][i-1,j-1].values) or np.isnan(pr[19][i-1,j-1].values) or np.isnan(pr[20][i-1,j-1].values) or np.isnan(pr[21][i-1,j-1].values) or np.isnan(pr[22][i-1,j-1].values) or np.isnan(pr[23][i-1,j-1].values) or np.isnan(pr[24][i-1,j-1].values) or np.isnan(pr[25][i-1,j-1].values) or np.isnan(pr[26][i-1,j-1].values) or np.isnan(pr[27][i-1,j-1].values):
                    for k in range(rd_num):
                        aux = pd.DataFrame({'a':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                        sensitivity_maps[k] = pd.concat([sensitivity_maps[i],aux],axis=0)
                        aux_t = pd.DataFrame({'at':[np.nan],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                        pvalue_maps[i] = pd.concat([pvalue_maps[k],aux_t],axis=0)
                    del aux, aux_t
                    x = np.array([])
                    continue
                x = self.create_x(i,j,pr)
                res = sm.OLS(x,y).fit()
                r2 = res.rsquared
                mse = res.conf_int(alpha=0.05, cols=None) #mse_model
                for l in range(len(rd_num)):
                    aux = pd.DataFrame({'a':res.params[l],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    sensitivity_maps[l] = pd.concat([sensitivity_maps[l] ,aux],axis=0)
                    aux_t = pd.DataFrame({'at':res.pvalues[l],'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                    pvalues_maps[l] = pd.concat([pvalues_maps[l] ,aux_t],axis=0)

                r2  = pd.DataFrame({'r2':r2,'lat':[lat[i-1].values.tolist()],'lon':[lon[j-1].values.tolist()]})
                R2ij = pd.concat([R2ij,r2],axis=0)
                del r2, aux, aux_t
                x = np.array([])
                
        #Guardo resultados en DataFrames
        for k in range(rd_num):
            DF = {'coef':sensitivity_maps[k].iloc[:,0],'lat':sensitivity_maps[k].iloc[:,1],'lon':sensitivity_maps[k].iloc[:,2]}
            DFij = pd.DataFrame(DF).fillna(0)
            DFij.to_csv(path+'/'+rd_names[k]+'.csv', float_format='%g')
            DFp = {'coef':pvalues_maps[k].iloc[:,0],'lat':pvalues_maps[k].iloc[:,1],'lon':pvalues_maps[k].iloc[:,2]}
            DFijp = pd.DataFrame(DFp).fillna(0)
            DFijp.to_csv(path+'/'+rd_names[k]+'p.csv', float_format='%g')

        R2ij.to_csv(path+'/R2ij.csv', float_format='%g')
                
        
    def convert_csv_files(self,path):
        file_names_nc = csv2nc.csv_to_nc(path)
        return file_names_nc
    
    
    def create_x(self,i,j,pr):
        x = np.array([])
        for k in range(len(pr)):
            aux = pr[k]
            x = np.append(x,aux[i-1,j-1].values)
            x_mean = np.mean(x); x_anom = (x - x_mean)
            x_new = x
        return x_new
    