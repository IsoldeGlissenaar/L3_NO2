# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:32:07 2023

@author: glissena
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

f = "/nobackup/users/glissena/data/TROPOMI/out_L3/02x02/CCI+p-L3-NO2_TC-TROPOMI_S5P_v020301-KNMI-201901-fv0100.nc"
ds = xr.open_dataset(f)
dates = [f[76:84], f[85:93]]

proj = ccrs.PlateCarree()

fig, axs = plt.subplots(
    3,
    2,
    figsize=(7.5, 5),
    dpi=400,
    constrained_layout=True,
    subplot_kw={"projection": proj},
)

# Superobservation - weighted
axs[0, 0].coastlines(resolution="50m", linewidth=0.3)
im = axs[0, 0].pcolormesh(
    ds.longitude,
    ds.latitude,
    ds.tropospheric_NO2_column_number_density[0,:,:] / 1e15,
    vmin=0,
    vmax=10,
    cmap="Spectral_r",
    transform=ccrs.PlateCarree(),
)
axs[0, 0].text(0.01, 0.92, "(a)", fontsize=8, transform=axs[0, 0].transAxes)

# STD1
axs[1, 0].coastlines(resolution="50m", linewidth=0.3)
im = axs[1, 0].pcolormesh(
    ds.longitude,
    ds.latitude,
    ds.tropospheric_NO2_column_number_density_temporal_std[0,:,:] / 1e15,
    vmin=0,
    vmax=1,
    cmap="YlOrRd",
    transform=ccrs.PlateCarree(),
)
axs[1, 0].text(0.01, 0.92, "(b)", fontsize=8, transform=axs[1, 0].transAxes)

# STD2
axs[2, 0].coastlines(resolution="50m", linewidth=0.3)
im = axs[2, 0].pcolormesh(
    ds.longitude,
    ds.latitude,
    ds.tropospheric_NO2_column_number_density_uncertainty_kernel[0,:,:] / 1e15,
    vmin=0,
    vmax=1,
    cmap="YlOrRd",
    transform=ccrs.PlateCarree(),
)
axs[2, 0].text(0.01, 0.92, "(c)", fontsize=8, transform=axs[2, 0].transAxes)

# #Relative uncertainty - STD2
# rel_uncer = (ds.std2/ds.weighted_mean_no2)*100
# rel_uncer.values[(ds.weighted_mean_no2/1e15)<0.1] = np.nan
# axs[2,0].coastlines(resolution='50m',linewidth=0.3)
# im=axs[2,0].pcolormesh(ds.longitude,ds.latitude,rel_uncer,
#                 vmin=0,vmax=100,cmap='YlOrRd',transform=ccrs.PlateCarree())
# axs[2,0].text(0.01,0.92,'(c)',fontsize=8,transform=axs[2,0].transAxes)


f = "/nobackup/users/glissena/data/TROPOMI/out_L3/02x02/CCI+p-L3-NO2_TC-TROPOMI_S5P_v020301-KNMI-201906-fv0100.nc"
ds = xr.open_dataset(f)
dates = [f[76:84], f[85:93]]

# Superobservation - weighted
axs[0, 1].coastlines(resolution="50m", linewidth=0.3)
im = axs[0, 1].pcolormesh(
    ds.longitude,
    ds.latitude,
    ds.tropospheric_NO2_column_number_density[0,:,:] / 1e15,
    vmin=0,
    vmax=10,
    cmap="Spectral_r",
    transform=ccrs.PlateCarree(),
)
cbar = plt.colorbar(im, ax=axs[0, 1], extend="both")
cbar.ax.tick_params(labelsize=6)
cbar.set_label("10$^{15}$ molecules/cm$^2$", size=6)
axs[0, 1].text(0.01, 0.92, "(d)", fontsize=8, transform=axs[0, 1].transAxes)

# STD1
axs[1, 1].coastlines(resolution="50m", linewidth=0.3)
im = axs[1, 1].pcolormesh(
    ds.longitude,
    ds.latitude,
    ds.tropospheric_NO2_column_number_density_temporal_std[0,:,:] / 1e15,
    vmin=0,
    vmax=1,
    cmap="YlOrRd",
    transform=ccrs.PlateCarree(),
)
cbar = plt.colorbar(im, ax=axs[1, 1], extend="max")
cbar.ax.tick_params(labelsize=6)
cbar.set_label("10$^{15}$ molecules/cm$^2$", size=6)
axs[1, 1].text(0.01, 0.92, "(e)", fontsize=8, transform=axs[1, 1].transAxes)

# STD2
axs[2, 1].coastlines(resolution="50m", linewidth=0.3)
im = axs[2, 1].pcolormesh(
    ds.longitude,
    ds.latitude,
    ds.tropospheric_NO2_column_number_density_uncertainty_kernel[0,:,:] / 1e15,
    vmin=0,
    vmax=1,
    cmap="YlOrRd",
    transform=ccrs.PlateCarree(),
)
cbar = plt.colorbar(im, ax=axs[2, 1], extend="max")
cbar.ax.tick_params(labelsize=6)
cbar.set_label("10$^{15}$ molecules/cm$^2$", size=6)
axs[2, 1].text(0.01, 0.92, "(f)", fontsize=8, transform=axs[2, 1].transAxes)

# #Relative uncertainty - STD2
# rel_uncer = (ds.std2/ds.weighted_mean_no2)*100
# rel_uncer.values[(ds.weighted_mean_no2/1e15)<0.1] = np.nan
# axs[2,1].coastlines(resolution='50m',linewidth=0.3)
# im=axs[2,1].pcolormesh(ds.longitude,ds.latitude,rel_uncer,
#                 vmin=0,vmax=100,cmap='YlOrRd',transform=ccrs.PlateCarree())
# cbar = plt.colorbar(im, ax=axs[2,1], extend='max')
# cbar.ax.tick_params(labelsize=6)
# cbar.set_label('%', size=6)
# axs[2,1].text(0.01,0.92,'(f)',fontsize=8,transform=axs[2,1].transAxes)


# fig.savefig('C:/Users/glissena/OneDrive - KNMI/Documents/figures/monthly_means.png')

#%%

print("January 2019")
f = "/nobackup/users/glissena/data/TROPOMI/out_L3/02x02/CCI+p-L3-NO2_TC-TROPOMI_S5P_v020301-KNMI-201901-fv0100.nc"
ds = xr.open_dataset(f)

print("--")
ams = ds.tropospheric_NO2_column_number_density.values[0, 711, 924] / 1e15
print(f"Amsterdam mean: {np.round(ams,2)} 1e15 molecules/cm2")
ams = ds.tropospheric_NO2_column_number_density_temporal_std.values[0, 711, 924] / 1e15
print(f"Amsterdam std1: {np.round(ams,2)} 1e15 molecules/cm2")
ams = (
    ds.tropospheric_NO2_column_number_density_uncertainty_kernel.values[0, 711, 924] / 1e15
)
print(f"Amsterdam std2: {np.round(ams,2)} 1e15 molecules/cm2")
print("--")
beij = ds.tropospheric_NO2_column_number_density.values[0, 649, 1481] / 1e15
print(f"Beijing mean: {np.round(beij,2)} 1e15 molecules/cm2")
beij = ds.tropospheric_NO2_column_number_density_temporal_std.values[0, 649, 1481] / 1e15
print(f"Beijing std1: {np.round(beij,2)} 1e15 molecules/cm2")
beij = (
    ds.tropospheric_NO2_column_number_density_uncertainty_kernel.values[0, 649, 1481]
    / 1e15
)
print(f"Beijing std2: {np.round(beij,2)} 1e15 molecules/cm2")
print("--")
afr = (
    np.nanmax(ds.tropospheric_NO2_column_number_density.values[0, 462:515, 797:1093])
    / 1e15
)
print(f"Africa mean: {np.round(afr,2)} 1e15 molecules/cm2")
afr = (
    np.nanmax(
        ds.tropospheric_NO2_column_number_density_temporal_std.values[0, 462:515, 797:1093]
    )
    / 1e15
)
print(f"Africa std1: {np.round(afr,2)} 1e15 molecules/cm2")
afr = (
    np.nanmax(
        ds.tropospheric_NO2_column_number_density_uncertainty_kernel.values[
            0, 462:515, 797:1093
        ]
    )
    / 1e15
)
print(f"Africa std2: {np.round(afr,2)} 1e15 molecules/cm2")

print("---------------")


print("June 2019")
f = "/nobackup/users/glissena/data/TROPOMI/out_L3/02x02/CCI+p-L3-NO2_TC-TROPOMI_S5P_v020301-KNMI-201906-fv0100.nc"
ds = xr.open_dataset(f)

print("--")
ams = ds.tropospheric_NO2_column_number_density.values[0, 711, 924] / 1e15
print(f"Amsterdam mean: {np.round(ams,2)} 1e15 molecules/cm2")
ams = ds.tropospheric_NO2_column_number_density_temporal_std.values[0, 711, 924] / 1e15
print(f"Amsterdam std1: {np.round(ams,2)} 1e15 molecules/cm2")
ams = (
    ds.tropospheric_NO2_column_number_density_uncertainty_kernel.values[0, 711, 924] / 1e15
)
print(f"Amsterdam std2: {np.round(ams,2)} 1e15 molecules/cm2")
print("--")
beij = ds.tropospheric_NO2_column_number_density.values[0, 649, 1481] / 1e15
print(f"Beijing mean: {np.round(beij,2)} 1e15 molecules/cm2")
beij = ds.tropospheric_NO2_column_number_density_temporal_std.values[0, 649, 1481] / 1e15
print(f"Beijing std1: {np.round(beij,2)} 1e15 molecules/cm2")
beij = (
    ds.tropospheric_NO2_column_number_density_uncertainty_kernel.values[0, 649, 1481]
    / 1e15
)
print(f"Beijing std2: {np.round(beij,2)} 1e15 molecules/cm2")
print("--")
afr = (
    np.nanmax(ds.tropospheric_NO2_column_number_density.values[0, 384:437, 952:1093])
    / 1e15
)
print(f"Africa mean: {np.round(afr,2)} 1e15 molecules/cm2")
afr = (
    np.nanmax(
        ds.tropospheric_NO2_column_number_density_temporal_std.values[0, 384:437, 952:1093]
    )
    / 1e15
)
print(f"Africa std1: {np.round(afr,2)} 1e15 molecules/cm2")
afr = (
    np.nanmax(
        ds.tropospheric_NO2_column_number_density_uncertainty_kernel.values[
            0, 384:437, 952:1093
        ]
    )
    / 1e15
)
print(f"Africa std2: {np.round(afr,2)} 1e15 molecules/cm2")
