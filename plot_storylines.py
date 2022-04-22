# Subplot number three for mean changes and other figures
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import os
import glob
import pandas as pd
import xarray as xr
import netCDF4
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth
import cartopy.util as cutil

def cross_year_season(month,season):
    #Season is a list with two values, begining and endig season
    return (month >= season[0]) & (month <= season[1])

#Este programa hace las figuras para high/low impact regional storylines JJA. 
import cartopy.crs as ccrs
import numpy as np
import cartopy.feature
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath
import os
import glob
import pandas as pd
import xarray as xr
import netCDF4
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth
import cartopy.util as cutil
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.gridspec as gridspec

def plot_storylines_ua(indices,indices_pval,frac_var,path_fig,title,season,full_season='yes'):
    
#Funciones------------------
    def split_title_line(title_text, split_on='(', max_words=4):  # , max_words=None):
        """
        A function that splits any string based on specific character
        (returning it with the string), with maximum number of words on it
        """
        split_at = title_text.find (split_on)
        ti = title_text
        if split_at > 1:
            ti = ti.split (split_on)
            for i, tx in enumerate (ti[1:]):
                ti[i + 1] = split_on + tx
        if type (ti) == type ('text'):
            ti = [ti]
        for j, td in enumerate (ti):
            if td.find (split_on) > 0:
                pass
            else:
                tw = td.split ()
                t2 = []
                for i in range (0, len (tw), max_words):
                    t2.append (' '.join (tw[i:max_words + i]))
                ti[j] = t2
        ti = [item for sublist in ti for item in sublist]
        ret_tex = []
        for j in range (len (ti)):
            for i in range(0, len(ti)-1, 2):
                if len (ti[i].split()) + len (ti[i+1].split ()) <= max_words:
                    mrg = " ".join ([ti[i], ti[i+1]])
                    ti = [mrg] + ti[2:]
                    break

        if len (ti[-2].split ()) + len (ti[-1].split ()) <= max_words:
            mrg = " ".join ([ti[-2], ti[-1]])
            ti = ti[:-2] + [mrg]
        return '\n'.join (ti)



    #Variables-----------------------------------------

    GlobalWarming = indices[0]
    TropicalWarming = indices[1]
    VorBreak_GW = indices[2]
    SeaSurfaceTemperature = indices[3]
    SeaSurfaceTemperature2 = indices[4]
    FracVar = frac_var
    
    latr = FracVar.lat 
    lat = GlobalWarming.lat
    lon = np.arange(0,357.188,2.81)
    gw, lon_c = add_cyclic_point(GlobalWarming.coef,lon)
    ta, lon_c = add_cyclic_point(TropicalWarming.coef,lon)
    sv, lon_c = add_cyclic_point(VorBreak_GW.coef,lon)
    sst1, lon_c = add_cyclic_point(SeaSurfaceTemperature.coef,lon)
    sst2, lon_c = add_cyclic_point(SeaSurfaceTemperature2.coef,lon)
    fv, lon_c  = add_cyclic_point(FracVar.coef,lon)

    t = 1.26
    mmm = gw
    story1 = (mmm + t*ta + t*sv) #+ r*t*sv_ta
    story2 = (mmm - t*ta - t*sv) #- r*t*sv_ta
    story3 = (mmm - t*ta + t*sv) #- r*t*sv_ta
    story4 = (mmm + t*ta - t*sv)  #+ r*t*sv_ta

    #Plot---------------------------------------------------
    cmapU850 = mpl.colors.ListedColormap(['darkblue','navy','steelblue','lightblue',
                                              'lightsteelblue','white','white','mistyrose',
                                              'lightcoral','indianred','brown','firebrick'])
    cmapU850.set_over('maroon')
    cmapU850.set_under('midnightblue')

    path_era = '/datos/ERA5/mon'
    u_ERA = xr.open_dataset(path_era+'/era5.mon.mean.nc')
    if full_season == 'yes':
        u_ERA = u_ERA.u.sel(lev=850).sel(time=slice('1979','2010'))
        u_ERA = u_ERA.groupby('time.season').mean(dim='time').sel(season=season)
    else:
        u_ERA = u_ERA.u.sel(lev=850).sel(time=slice('1979','2010')).sel(time=cross_year_season(u_ERA.sel(time=slice('1979','2010'))['time.month'],season)).mean(dim='time')

    fig = plt.figure(figsize=(15, 15),dpi=300,constrained_layout=True)
    widths = [1, 1, 1]
    heights = [1, 1, 1]
    projection = ccrs.SouthPolarStereo(central_longitude=300)
    data_crs = ccrs.PlateCarree()
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    spec9 = fig.add_gridspec(ncols=3, nrows=3,width_ratios=widths,height_ratios=heights)
    ax1 = plt.subplot(spec9[0,0],projection=projection)
    ax1.set_extent([0,359.9, -90, 0], crs=data_crs)
    ax1.set_boundary(circle, transform=ax1.transAxes)
    clevels = np.arange(-1,1.25,.25)
    im1=ax1.contourf(lon_c, lat, story3,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax1.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.title(split_title_line(r'low tropical warm + late vortex breakdown',max_words=4),fontsize=20)
    ax1.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax1.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax1.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax1.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
    plt1_ax = plt.gca()
    left_1, bottom_1, width_1, height_1 = plt1_ax.get_position().bounds


    ax2 = plt.subplot(spec9[0,2],projection=projection)
    ax2.set_boundary(circle, transform=ax2.transAxes)
    im2=ax2.contourf(lon_c, lat, story1,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax2.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    ax2.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax2.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax2.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax2.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
    plt2_ax = plt.gca()
    left_2, bottom_2, width_2, height_2 = plt2_ax.get_position().bounds
    plt.title(split_title_line(r'high tropical warm + late vortex breakdown',max_words=4),fontsize=20)

    ax3 = plt.subplot(spec9[1,1],projection=projection)
    ax3.set_boundary(circle, transform=ax3.transAxes)
    im3=ax3.contourf(lon_c, lat, mmm,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax3.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    ax3.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax3.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax3.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax3.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
    plt3_ax = plt.gca()
    left_3, bottom_3, width_3, height_3 = plt3_ax.get_position().bounds
    plt.title(split_title_line(r'multimodel ensemble mean change',max_words=2),fontsize=20)


    ax4 = plt.subplot(spec9[2,0],projection=projection)
    ax4.set_boundary(circle, transform=ax4.transAxes)
    im4=ax4.contourf(lon_c, lat, story2,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax4.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    ax4.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax4.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax4.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax4.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
    plt4_ax = plt.gca()
    left_4, bottom_4, width_4, height_4 = plt4_ax.get_position().bounds
    plt.title(split_title_line(r'low tropical warm + early vortex breakdown',max_words=4),fontsize=20)

    ax5 = plt.subplot(spec9[2,2],projection=projection)
    ax5.set_boundary(circle, transform=ax5.transAxes)
    im5=ax5.contourf(lon_c, lat, story4,clevels,transform=data_crs,cmap=cmapU850,extend='both')
    cnt=ax5.contour(u_ERA.lon,u_ERA.lat, u_ERA.values,levels=[8],transform=data_crs,linewidths=1.2, colors='black', linestyles='-')
    plt.clabel(cnt,inline=True,fmt='%1.0f',fontsize=8)
    ax5.add_feature(cartopy.feature.COASTLINE,alpha=.5)
    ax5.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax5.gridlines(crs=data_crs, linewidth=0.3, linestyle='-')
    ax5.set_extent([-180, 180, -90, -25], ccrs.PlateCarree())
    plt5_ax = plt.gca()
    left_5, bottom_5, width_5, height_5 = plt5_ax.get_position().bounds
    plt.title(split_title_line(r'high tropical warm + early vortex breakdown',max_words=4),fontsize=20)


    plt5_ax = plt.gca()
    left5, bottom5, width5, height5 = plt5_ax.get_position().bounds
    fourth_plot_left = plt5_ax.get_position().bounds[0]

    #Utilizo las coordenadas para definir la posición de la colorbar
    colorbar_axes = fig.add_axes([fourth_plot_left - 2.2*width5, bottom5 -0.15, 3*width5, 0.02])

    # Add the colour bar
    cbar = plt.colorbar(im2, colorbar_axes, fraction=0.05, pad=0.04,aspect=16, orientation='horizontal')
    ticklabs = cbar.ax.get_xticklabels()
    print('ticklabs',ticklabs)
    #cbar.ax.set_xticklabels(ticklabs, fontsize=20)
    cbar.set_label('ms$^{-1}$K$^{-1}$',fontsize=22)

    #Save figure
    #plt.savefig(path_fig+'/'+title,bbox_inches='tight')
    #plt.clf
    return fig

