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
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.2

f = "/nobackup/users/glissena/data/TROPOMI/out_L3/02x02/v1_2/ESACCI-PREC-L3-NO2_TC-TROPOMI_S5P-KNMI-1M-20190101_20190131-fv0120.nc"
ds = xr.open_dataset(f)
dates = [f[76:84], f[85:93]]

data = ds.tropospheric_NO2_column_number_density[0,:,:] / 1e15
data_m = np.ma.masked_where(ds.qa_L3.values[0,:,:]!=0, data)


proj = ccrs.PlateCarree()

fig, axs = plt.subplots(
    2,
    2,
    figsize=(7.5, 3.6),
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
axs[0, 0].pcolor(ds.longitude,ds.latitude,data_m,hatch='xxxxxxxx', alpha=0, transform=ccrs.PlateCarree())
axs[0, 0].text(0.01, 0.92, "(a)", fontsize=8, transform=axs[0, 0].transAxes)

# STD1
axs[1, 0].coastlines(resolution="50m", linewidth=0.3)
im = axs[1, 0].pcolormesh(
    ds.longitude,
    ds.latitude,
    ds.tropospheric_NO2_column_number_density_total_uncertainty_kernel[0,:,:] / 1e15,
    vmin=0,
    vmax=1,
    cmap="YlOrRd",
    transform=ccrs.PlateCarree(),
)
axs[1, 0].pcolor(ds.longitude,ds.latitude,data_m,hatch='xxxxxxxx', alpha=0, transform=ccrs.PlateCarree())
axs[1, 0].text(0.01, 0.92, "(b)", fontsize=8, transform=axs[1, 0].transAxes)



f = "/nobackup/users/glissena/data/TROPOMI/out_L3/02x02/v1_2/ESACCI-PREC-L3-NO2_TC-TROPOMI_S5P-KNMI-1M-20190601_20190630-fv0120.nc"
ds = xr.open_dataset(f)
dates = [f[76:84], f[85:93]]

data = ds.tropospheric_NO2_column_number_density[0,:,:] / 1e15
data_m = np.ma.masked_where(ds.qa_L3.values[0,:,:]!=0, data)


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
axs[0, 1].pcolor(ds.longitude,ds.latitude,data_m,hatch='xxxxxxxx', alpha=0, transform=ccrs.PlateCarree())
cbar = plt.colorbar(im, ax=axs[0, 1], extend="both")
cbar.ax.tick_params(labelsize=6)
cbar.set_label("10$^{15}$ molecules/cm$^2$", size=6)
axs[0, 1].text(0.01, 0.92, "(c)", fontsize=8, transform=axs[0, 1].transAxes)

# STD1
axs[1, 1].coastlines(resolution="50m", linewidth=0.3)
im = axs[1, 1].pcolormesh(
    ds.longitude,
    ds.latitude,
    ds.tropospheric_NO2_column_number_density_total_uncertainty_kernel[0,:,:] / 1e15,
    vmin=0,
    vmax=1,
    cmap="YlOrRd",
    transform=ccrs.PlateCarree(),
)
axs[1, 1].pcolor(ds.longitude,ds.latitude,data_m,hatch='xxxxxxxx', alpha=0, transform=ccrs.PlateCarree())
cbar = plt.colorbar(im, ax=axs[1, 1], extend="max")
cbar.ax.tick_params(labelsize=6)
cbar.set_label("10$^{15}$ molecules/cm$^2$", size=6)
axs[1, 1].text(0.01, 0.92, "(d)", fontsize=8, transform=axs[1, 1].transAxes)



# fig.savefig('/usr/people/glissena/Documents/projects/L3_NO2/figures/monthly_means.png')

#%%

print("January 2019")
f = "/nobackup/users/glissena/data/TROPOMI/out_L3/02x02/v1_2/ESACCI-PREC-L3-NO2_TC-TROPOMI_S5P-KNMI-1M-20190101_20190131-fv0120.nc"
ds = xr.open_dataset(f)

print("--")
ams = ds.tropospheric_NO2_column_number_density.values[0, 711, 924] / 1e15
print(f"Amsterdam mean: {np.round(ams,2)} 1e15 molecules/cm2")
ams1 = ds.tropospheric_NO2_column_number_density_temporal_std.values[0, 711, 924] / 1e15
print(f"Amsterdam std1: {np.round(ams1,2)} 1e15 molecules/cm2")
ams2 = (
    ds.tropospheric_NO2_column_number_density_measurement_uncertainty.values[0, 711, 924] / 1e15
)
print(f"Amsterdam std2: {np.round(ams2,2)} 1e15 molecules/cm2")
ams3 = (
    ds.tropospheric_NO2_column_number_density_total_uncertainty.values[0, 711, 924] / 1e15
)
print(f"Amsterdam total_uncer: {np.round(ams3,2)} 1e15 molecules/cm2")
perc = ams3/ams*100
print(f"Amsterdam total_uncer: {np.round(perc,2)}% of total")
print("--")
beij = ds.tropospheric_NO2_column_number_density.values[0, 649, 1481] / 1e15
print(f"Beijing mean: {np.round(beij,2)} 1e15 molecules/cm2")
beij1 = ds.tropospheric_NO2_column_number_density_temporal_std.values[0, 649, 1481] / 1e15
print(f"Beijing std1: {np.round(beij1,2)} 1e15 molecules/cm2")
beij2 = (
    ds.tropospheric_NO2_column_number_density_measurement_uncertainty.values[0, 649, 1481]
    / 1e15
)
print(f"Beijing std2: {np.round(beij2,2)} 1e15 molecules/cm2")
beij3 = (
    ds.tropospheric_NO2_column_number_density_total_uncertainty.values[0, 649, 1481]
    / 1e15
)
print(f"Beijing total_uncer: {np.round(beij3,2)} 1e15 molecules/cm2")
perc = beij3/beij*100
print(f"Beijing total_uncer: {np.round(perc,2)}% of total")
print("--")
afr = (
    np.nanmax(ds.tropospheric_NO2_column_number_density.values[0, 462:515, 797:1093])
    / 1e15
)
print(f"Africa mean: {np.round(afr,2)} 1e15 molecules/cm2")
afr1 = (
    np.nanmax(
        ds.tropospheric_NO2_column_number_density_temporal_std.values[0, 462:515, 797:1093]
    )
    / 1e15
)
print(f"Africa std1: {np.round(afr1,2)} 1e15 molecules/cm2")
afr2 = (
    np.nanmax(
        ds.tropospheric_NO2_column_number_density_measurement_uncertainty.values[
            0, 462:515, 797:1093
        ]
    )
    / 1e15
)
print(f"Africa std2: {np.round(afr2,2)} 1e15 molecules/cm2")
afr3 = (
    np.nanmax(
        ds.tropospheric_NO2_column_number_density_total_uncertainty.values[
            0, 462:515, 797:1093
        ]
    )
    / 1e15
)
print(f"Africa total_uncer: {np.round(afr3,2)} 1e15 molecules/cm2")
perc = afr3/afr*100
print(f"Africa total_uncer: {np.round(perc,2)}% of total")

print("---------------")


print("June 2019")
f = "/nobackup/users/glissena/data/TROPOMI/out_L3/02x02/v1_2/ESACCI-PREC-L3-NO2_TC-TROPOMI_S5P-KNMI-1M-20190601_20190630-fv0120.nc"
ds = xr.open_dataset(f)

print("--")
ams = ds.tropospheric_NO2_column_number_density.values[0, 711, 924] / 1e15
print(f"Amsterdam mean: {np.round(ams,2)} 1e15 molecules/cm2")
ams1 = ds.tropospheric_NO2_column_number_density_temporal_std.values[0, 711, 924] / 1e15
print(f"Amsterdam std1: {np.round(ams1,2)} 1e15 molecules/cm2")
ams2 = (
    ds.tropospheric_NO2_column_number_density_measurement_uncertainty.values[0, 711, 924] / 1e15
)
print(f"Amsterdam std2: {np.round(ams2,2)} 1e15 molecules/cm2")
ams3 = (
    ds.tropospheric_NO2_column_number_density_total_uncertainty.values[0, 711, 924] / 1e15
)
print(f"Amsterdam total_uncer: {np.round(ams3,2)} 1e15 molecules/cm2")
perc = ams3/ams*100
print(f"Amsterdam total_uncer: {np.round(perc,2)}% of total")
print("--")
beij = ds.tropospheric_NO2_column_number_density.values[0, 649, 1481] / 1e15
print(f"Beijing mean: {np.round(beij,2)} 1e15 molecules/cm2")
beij1 = ds.tropospheric_NO2_column_number_density_temporal_std.values[0, 649, 1481] / 1e15
print(f"Beijing std1: {np.round(beij1,2)} 1e15 molecules/cm2")
beij2 = (
    ds.tropospheric_NO2_column_number_density_measurement_uncertainty.values[0, 649, 1481]
    / 1e15
)
print(f"Beijing std2: {np.round(beij2,2)} 1e15 molecules/cm2")
beij3 = (
    ds.tropospheric_NO2_column_number_density_total_uncertainty.values[0, 649, 1481]
    / 1e15
)
print(f"Beijing total_uncer: {np.round(beij3,2)} 1e15 molecules/cm2")
perc = beij3/beij*100
print(f"Beijing total_uncer: {np.round(perc,2)}% of total")
print("--")
afr = (
    np.nanmax(ds.tropospheric_NO2_column_number_density.values[0, 384:437, 952:1093])
    / 1e15
)
print(f"Africa mean: {np.round(afr,2)} 1e15 molecules/cm2")
afr1 = (
    np.nanmax(
        ds.tropospheric_NO2_column_number_density_temporal_std.values[0, 384:437, 952:1093]
    )
    / 1e15
)
print(f"Africa std1: {np.round(afr1,2)} 1e15 molecules/cm2")
afr2 = (
    np.nanmax(
        ds.tropospheric_NO2_column_number_density_measurement_uncertainty.values[
            0, 384:437, 952:1093
        ]
    )
    / 1e15
)
print(f"Africa std2: {np.round(afr2,2)} 1e15 molecules/cm2")
afr3 = (
    np.nanmax(
        ds.tropospheric_NO2_column_number_density_total_uncertainty.values[
            0, 384:437, 952:1093
        ]
    )
    / 1e15
)
print(f"Africa total_uncer: {np.round(afr3,2)} 1e15 molecules/cm2")
perc = afr3/afr*100
print(f"Africa total_uncer: {np.round(perc,2)}% of total")
