#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 09:07:37 2023

@author: glissena
"""

import numpy as np
import calendar
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from mapplot_func import world_plot

date = "201901"
# f = f"/nobackup/users/glissena/data/TROPOMI/out_L3/NO2_TROPOMI_{date}.nc"

f = "/nobackup/users/glissena/data/TROPOMI/out_L3/02x02/ESACCI-PREC-L3-NO2_TC-TROPOMI_S5P-KNMI-1M-20190101_20190131-fv0121.nc"
ds = xr.open_dataset(f)

# ds.tropospheric_NO2_column_number_density.values[ds.qa_L3==0] = np.nan

# Superobservation - weighted
world_plot(
    (ds.tropospheric_NO2_column_number_density / 1e15)[0, :, :],
    ds.longitude,
    ds.latitude,
    cbar_label="10$^{15}$ molecules/cm$^2$",
    extend="both",
    title=f"{calendar.month_name[int(date[4:6])]} {date[0:4]} - Tropospheric VCD TROPOMI (S5P)",
)

# STD1
world_plot(
    (ds.tropospheric_NO2_column_number_density_temporal_std / 1e15)[0, :, :],
    ds.longitude,
    ds.latitude,
    vmin=0,
    vmax=1,
    cmap="YlOrRd",
    extend="max",
    cbar_label="10$^{15}$ molecules/cm$^2$",
    title="temporal uncertainty",
)

# STD2
world_plot(
    (ds.tropospheric_NO2_column_number_density_measurement_uncertainty / 1e15)[0, :, :],
    ds.longitude,
    ds.latitude,
    vmin=0,
    vmax=1,
    cmap="YlOrRd",
    extend="max",
    cbar_label="10$^{15}$ molecules/cm$^2$",
    title="measurement uncertainty",
)

# total uncertainty
world_plot(
    (ds.tropospheric_NO2_column_number_density_total_uncertainty / 1e15)[0, :, :],
    ds.longitude,
    ds.latitude,
    vmin=0,
    vmax=1,
    cmap="YlOrRd",
    extend="max",
    cbar_label="10$^{15}$ molecules/cm$^2$",
    title="total uncertainty",
)

#Temporal representativity uncertainty
temp_rep = np.sqrt(ds.tropospheric_NO2_column_number_density_total_uncertainty**2 - ds.tropospheric_NO2_column_number_density_measurement_uncertainty**2)
world_plot(
    temp_rep[0,:,:] / 1e15,
    ds.longitude,
    ds.latitude,
    vmin=0,
    vmax=1,
    cmap="YlOrRd",
    extend="max",
    cbar_label="10$^{15}$ molecules/cm$^2$",
    title="temporal representativity uncertainty",
)


# Difference in prop uncer and total
world_plot(
    (
        (
            ds.tropospheric_NO2_column_number_density_total_uncertainty
            - ds.tropospheric_NO2_column_number_density_measurement_uncertainty
        )
        / 1e15
    )[0, :, :],
    ds.longitude,
    ds.latitude,
    vmin=-0.5,
    vmax=0.5,
    cmap="RdBu",
    extend="both",
    cbar_label="10$^{15}$ molecules/cm$^2$",
    title="total - propegated measurement ",
)


# Difference in STD1 and STD2
world_plot(
    (
        (
            ds.tropospheric_NO2_column_number_density_temporal_std
            - ds.tropospheric_NO2_column_number_density_measurement_uncertainty
        )
        / 1e15
    )[0, :, :],
    ds.longitude,
    ds.latitude,
    vmin=-0.5,
    vmax=0.5,
    cmap="RdBu",
    extend="both",
    cbar_label="10$^{15}$ molecules/cm$^2$",
    title="std1 - std2",
)


# Relative uncertainty - STD1
world_plot(
    (
        ds.tropospheric_NO2_column_number_density_temporal_std
        / ds.tropospheric_NO2_column_number_density
    )[0, :, :]
    * 100,
    ds.longitude,
    ds.latitude,
    vmin=0,
    vmax=100,
    cmap="YlOrRd",
    extend="max",
    cbar_label="%",
    title="Relative - temporal uncertainty",
)

# Relative uncertainty - STD2
world_plot(
    (ds.tropospheric_NO2_column_number_density_measurement_uncertainty 
     / ds.tropospheric_NO2_column_number_density)[0, :, :] * 100,
    ds.longitude,
    ds.latitude,
    vmin=0,
    vmax=100,
    cmap="YlOrRd",
    extend="max",
    cbar_label="%",
    title="Relative STD3 - measurement uncertainty",
)

# Relative uncertainty - total
world_plot(
    (ds.tropospheric_NO2_column_number_density_total_uncertainty 
     / ds.tropospheric_NO2_column_number_density)[0, :, :] * 100,
    ds.longitude,
    ds.latitude,
    vmin=0,
    vmax=100,
    cmap="YlOrRd",
    extend="max",
    cbar_label="%",
    title="Relative total uncertainty",
)


# Relative uncertainty - total increase
plot_data = ((ds.tropospheric_NO2_column_number_density_total_uncertainty 
              / ds.tropospheric_NO2_column_number_density)[0, :, :] * 100) - ((ds.tropospheric_NO2_column_number_density_measurement_uncertainty    
            / ds.tropospheric_NO2_column_number_density)[0, :, :] * 100)

plot_data.values[ds.tropospheric_NO2_column_number_density.values[0,:,:]<=0] = np.nan                                                                              

world_plot(
    plot_data,
    ds.longitude,
    ds.latitude,
    vmin=-5,
    vmax=5,
    # region="europe",
    cmap="RdBu",
    extend="both",
    cbar_label="%",
    title="Increase in relative uncertainty by adding temporal representativity",
)


# GCOS requirements:
# max(20% ; 1e15). The uncertainty of 1x10^15 molec/cm2 holds for tropospheric 
# NO2 columns up to 4.0x10^15 molec/cm2. For larger column values the relative 
# uncertainty of 20% holds.
# https://gcos.wmo.int/en/essential-climate-variables/precursors

lim_high = 0.2 #20%
lim_low = 1 #e15 molec/cm2

def gcos(lim_high, lim_low):
    gcos_req_g = np.zeros(ds.tropospheric_NO2_column_number_density.values.shape)
    low = (ds.tropospheric_NO2_column_number_density.values/1e15 < 4)
    gcos_req_g[~low] = (2 * ds.tropospheric_NO2_column_number_density_total_uncertainty.values < 
                ds.tropospheric_NO2_column_number_density.values * lim_high)[~low]
    gcos_req_g[low] = (2 * ds.tropospheric_NO2_column_number_density_total_uncertainty.values/1e15 < 
                     lim_low)[low]
    return gcos_req_g

gcos_req_g = gcos(0.2,1)
gcos_req_b = gcos(0.4,2)
gcos_req_t = gcos(1,5)


gcos_req = np.ones(gcos_req_t.shape)
gcos_req[gcos_req_t.astype(int)==1] = 2
gcos_req[gcos_req_b.astype(int)==1] = 3
gcos_req[gcos_req_g.astype(int)==1] = 4

gcos_req[np.isnan(ds.tropospheric_NO2_column_number_density.values)] = np.nan

gcos_req[ds.qa_L3.values==0] = 0

import matplotlib as mpl
cmap = mpl.colors.LinearSegmentedColormap.from_list("lcc", ['lightgrey','red','#fdc086','#beaed4','#7fc97f'])
norm = mpl.colors.BoundaryNorm([0,1,2,3,4,5], cmap.N)

fig = plt.figure(dpi=400)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution="50m", linewidth=0.3)
# ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
# ax.set_extent([-15, 40, 30, 70], crs=ccrs.PlateCarree())
# ax.set_extent([100, 142, 17, 49], crs=ccrs.PlateCarree())
plt.pcolormesh(
    ds.longitude, ds.latitude, gcos_req[0,:,:], cmap=cmap, norm=norm,transform=ccrs.PlateCarree()
)
cbar = plt.colorbar(shrink=0.6)
cbar.ax.tick_params(labelsize=6)
cbar.set_ticks([0.5,1.5,2.5,3.5,4.5])
cbar.set_ticklabels(['qa==0','Too high','Threshold','Breakthrough','Goal'])

plt.title("GCOS requirements", fontsize=8)
fig.show()


