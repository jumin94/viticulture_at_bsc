#Winkler index: cumulated sum from October to April of all the degrees of daily mean temperature above 10ºC
#To calculate Growing Degree Days, subtract the grapevine’s threshold temperature of 50°F (10°C) from the mean daily air temperature in any 24-hour period (the mean daily temperature adds together the high and low temperature for the day and divides that value by two). However, if the mean temperature is at or below the base temperature for a crop or pest of interest, the GDD value is zero. If the mean temperature is above the base temperature, then the GDD equals the value of the mean temperature minus the base temperature. GDD values are accumulated during a growing season. 

#Imports 
import xarray as xr
import numpy as np
import pandas as pd

#Open ERA5 data
path = '/datos/ERA5_aux/troposphere/temperature'
years = np.arange(1979,2019,1)
temp_day = xr.open_dataset(path+'/t_01_1979_day.nc')
months = ['01','02','03','04','05','06','07','08','09','10','11','12']
for k in range(len(years)):
    for j in months:
        temp_day_aux = xr.open_dataset(path+'/t_'+j+'_'+str(years[k])+'_day.nc')
        temp_day = xr.merge([temp_day,temp_day_aux])


#Winkler index
def cross_year_season(month,season):
    #Season is a list with two values, begining and endig season
    return (month >= season[0]) & (month <= season[1])

#Temperatures for summing
winkler_temp = winkler_temp.where(winkler_temp.t > 10)
winkler_index_ON = winkler_temp.sel(time=cross_year_season(winkler_temp['time.month'],[10,11])).sum(dim='time')
wnkler_index_DJF = winkler_temp.sel(time=cross_year_season(winkler_temp['time.month'],[12,2])).sum(dim='time')
winkler_index_MA = winkler_temp.sel(time=cross_year_season(winkler_temp['time.month'],[3,4])).sum(dim='time')

#Winkler 1940-1969
mean_temp_change = 0.2
winkler_temp = temp_day - 0.2
winkler_temp = winkler_temp.where(winkler_temp.t > 10)
winkler_index_ON = winkler_temp.sel(time=cross_year_season(winkler_temp['time.month'],[10,11])).sum(dim='time')
wnkler_index_DJF = winkler_temp.sel(time=cross_year_season(winkler_temp['time.month'],[12,2])).sum(dim='time')
winkler_index_MA = winkler_temp.sel(time=cross_year_season(winkler_temp['time.month'],[3,4])).sum(dim='time')

#Winkler Storylines
winkler_temp_storyline1_ON = temp_day.sel(time=cross_year_season(winkler_temp['time.month'],[10,11])).sum(dim='time') + storyline1_ON*2
winkler_temp_storyline1_DJF = temp_day.sel(time=cross_year_season(winkler_temp['time.month'],[12,2])).sum(dim='time') + storyline1_DJF*2
winkler_temp_storyline1_MA = temp_day.sel(time=cross_year_season(winkler_temp['time.month'],[3,4])).sum(dim='time') + storyline1_MA*2

winkler_temp_storyline2_ON = temp_day.sel(time=cross_year_season(winkler_temp['time.month'],[10,11])).sum(dim='time') + storyline2_ON*2
winkler_temp_storyline2_DJF = temp_day.sel(time=cross_year_season(winkler_temp['time.month'],[12,2])).sum(dim='time') + storyline2_DJF*2
winkler_temp_storyline2_MA = temp_day.sel(time=cross_year_season(winkler_temp['time.month'],[3,4])).sum(dim='time') + storyline2_MA*2

winkler_temp_storyline3_ON = temp_day.sel(time=cross_year_season(winkler_temp['time.month'],[10,11])).sum(dim='time') + storyline3_ON*2
winkler_temp_storyline3_DJF = temp_day.sel(time=cross_year_season(winkler_temp['time.month'],[12,2])).sum(dim='time') + storyline3_DJF*2
winkler_temp_storyline3_MA = temp_day.sel(time=cross_year_season(winkler_temp['time.month'],[3,4])).sum(dim='time') + storyline3_MA*2

winkler_temp_storyline4_ON = temp_day.sel(time=cross_year_season(winkler_temp['time.month'],[10,11])).sum(dim='time') + storyline4_ON*2
winkler_temp_storyline4_DJF = temp_day.sel(time=cross_year_season(winkler_temp['time.month'],[12,2])).sum(dim='time') + storyline4_DJF*2
winkler_temp_storyline4_MA = temp_day.sel(time=cross_year_season(winkler_temp['time.month'],[3,4])).sum(dim='time') + storyline4_MA*2

past_winkler = winkler_index_ON + winkler_index_DJF + winkler_index_MA

winkler_index_storyline1 = winkler_temp_storyline1_ON.sum(dim='time') + winkler_temp_storyline1_DJF.sum(dim='time') + winkler_temp_storyline1_MA.sum(dim='time')
winkler_index_storyline2 = winkler_temp_storyline1_ON.sum(dim='time') + winkler_temp_storyline1_DJF.sum(dim='time') + winkler_temp_storyline1_MA.sum(dim='time')
winkler_index_storyline3 = winkler_temp_storyline1_ON.sum(dim='time') + winkler_temp_storyline1_DJF.sum(dim='time') + winkler_temp_storyline1_MA.sum(dim='time')
winkler_index_storyline4 = winkler_temp_storyline1_ON.sum(dim='time') + winkler_temp_storyline1_DJF.sum(dim='time') + winkler_temp_storyline1_MA.sum(dim='time')


winkler_change_storyline1 = winkler_index_storyline1 - past_winkler
winkler_change_storyline2 = winkler_index_storyline2 - past_winkler
winkler_change_storyline3 = winkler_index_storyline3 - past_winkler
winkler_change_storyline4 = winkler_index_storyline4 - past_winkler




