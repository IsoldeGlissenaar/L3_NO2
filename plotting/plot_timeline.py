#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:16:30 2024

@author: glissena
"""


import os
import calendar
import scipy
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

#
location ='Bologna'
min_lat = 44.3
max_lat =44.8
min_lon = 11.1
max_lon = 11.6


#Load TROPOMI monthly means
dates_all = np.concatenate(
    [
        np.arange(201801, 201813, 1),
        np.arange(201901, 201913, 1),
        np.arange(202001, 202013, 1),
        np.arange(202101, 202113, 1),
    ]
).astype(str)

data_trop = np.full((len(dates_all), 900, 1800), np.nan)
trop_uncer = np.full((len(dates_all), 900, 1800), np.nan)
for i, date in enumerate(dates_all):
    f = f"/nobackup/users/glissena/data/TROPOMI/out_L3/02x02/CCIp-L3-NO2_TC-TROPOMI_S5P_v020301-KNMI-{date}-fv0110.nc"
    if os.path.isfile(f):
        ds_trop = xr.open_dataset(f)
        data_trop[i, :, :] = ds_trop.tropospheric_NO2_column_number_density.values
        data_trop[i,:,:][ds_trop.qa_L3.values[0,:,:]==0] = np.nan
        trop_uncer[i, :, :] = ds_trop.tropospheric_NO2_column_number_density_total_uncertainty.values
        trop_uncer[i,:,:][ds_trop.qa_L3.values[0,:,:]==0] = np.nan

dates_float = np.round(dates_all.astype(int),-2)/100 + (abs(dates_all.astype(int))%100-1)/12

#Get limits of region
trop_lat_min = np.where((ds_trop.latitude_bounds[:,0]>(min_lat-0.01)))[0][0]
trop_lat_max = np.where((ds_trop.latitude_bounds[:,1]<(max_lat+0.01)))[0][-1]+1
if trop_lat_max==trop_lat_min:
    trop_lat_max += 1
trop_lon_min = np.where((ds_trop.longitude_bounds[:,0]>(min_lon-0.01)))[0][0]
trop_lon_max = np.where((ds_trop.longitude_bounds[:,1]<(max_lon+0.01)))[0][-1]+1
if trop_lon_max==trop_lon_min:
    trop_lon_max += 1


x = dates_float[4:]
y = np.nanmean(data_trop[4:,trop_lat_min:trop_lat_max,trop_lon_min:trop_lon_max],axis=(1,2))/1e15
x = x[~np.isnan(y)]
y = y[~np.isnan(y)]
y = y[~np.isnan(x)]
x = x[~np.isnan(x)]
regr = scipy.stats.linregress(x,y)

#Plot timeline
trop_mean = np.nanmean(data_trop[:,trop_lat_min:trop_lat_max,trop_lon_min:trop_lon_max],axis=(1,2))/1e15
trop_uncer = np.nanmean(trop_uncer[:,trop_lat_min:trop_lat_max,trop_lon_min:trop_lon_max],axis=(1,2))/1e15

fig,ax = plt.subplots(dpi=500,figsize=(10,3))
tick_idx = np.arange(2018,2023,1) 
ax.plot(dates_float,
          trop_mean,c='C1',linewidth=1,label="TROPOMI")
ax.fill_between(dates_float,
                trop_mean-trop_uncer,
                trop_mean+trop_uncer,
                color='C1',edgecolors=None,
                alpha=0.3
                )
ax.set_xticks(tick_idx)
ax.set_xticklabels(np.arange(2018,2023,1).astype(str))
ax.tick_params(axis='both', which='major', labelsize=10)
# ax.xaxis.set_minor_locator(MultipleLocator(1/12))
ax.set_ylim(ymin=0)
ax.set_ylabel("Mean tropospheric column\n[10$^{15}$ molecules/cm$^2$]",fontsize=12)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.grid(linestyle=':',linewidth=0.5) 
ax.set_xlim([2018.2,2022.2])
plt.title(f"{location} - TROPOMI NO$_2$ Tropospheric VCD")


#Plot map Tropomi
min_lat2 = ds_trop.latitude_bounds.values[trop_lat_min,0]
max_lat2 = ds_trop.latitude_bounds.values[trop_lat_max,0]
min_lon2 = ds_trop.longitude_bounds.values[trop_lon_min,0]
max_lon2 = ds_trop.longitude_bounds.values[trop_lon_max,0]

fig = plt.figure(dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution="10m", linewidth=0.3)
ax.set_extent([min_lon-2,max_lon+2,min_lat-2,max_lat+2],crs=ccrs.PlateCarree())
im = ax.pcolormesh(
    ds_trop.longitude,
    ds_trop.latitude,
    data_trop[4, :, :] / 1e15,
    vmin=0,
    vmax=10,
    cmap="Spectral_r",
    transform=ccrs.PlateCarree(),
) 
ax.plot([min_lon2, max_lon2, max_lon2, min_lon2, min_lon2], 
        [min_lat2, min_lat2, max_lat2, max_lat2, min_lat2],
         color='black', linewidth=1, #remove this line to get straight lines
         )
cbar = plt.colorbar(im, ax=ax, shrink=0.8, extend="both")
cbar.ax.tick_params(labelsize=8)
cbar.set_label("10$^{15}$ molecules/cm$^2$", size=8)

