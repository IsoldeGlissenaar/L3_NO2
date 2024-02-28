#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 09:07:37 2023

@author: glissena
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from mapplot_func import world_plot

date = "201901"
# f = f"/nobackup/users/glissena/data/TROPOMI/out_L3/NO2_TROPOMI_{date}.nc"

f = f"/nobackup/users/glissena/data/TROPOMI/out_L3/res_geos_chem/CCI+p-L3-NO2_TC-TROPOMI_S5P_v020301-KNMI-{date}-fv0100.nc"
ds = xr.open_dataset(f)


# Superobservation - weighted
world_plot(
    (ds.tropospheric_NO2_column_number_density / 1e15)[0, :, :],
    ds.longitude,
    ds.latitude,
    cbar_label="10$^{15}$ molecules/cm$^2$",
    extend="both",
    title="Weighted mean of superobservations: " + date,
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
    title="STD1 - temporal uncertainty",
)

# STD2
world_plot(
    (ds.tropospheric_NO2_column_number_density_uncertainty_kernel / 1e15)[0, :, :],
    ds.longitude,
    ds.latitude,
    vmin=0,
    vmax=1,
    cmap="YlOrRd",
    extend="max",
    cbar_label="10$^{15}$ molecules/cm$^2$",
    title="STD2 - measurement uncertainty",
)


# Difference in STD1 and STD2
world_plot(
    (
        (
            ds.tropospheric_NO2_column_number_density_temporal_std
            - ds.tropospheric_NO2_column_number_density_uncertainty_kernel
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
    title="Relative STD1 - temporal uncertainty",
)

# Relative uncertainty - STD2
world_plot(
    (ds.std2 / ds.weighted_mean_no2) * 100,
    ds.longitude,
    ds.latitude,
    vmin=0,
    vmax=100,
    cmap="YlOrRd",
    extend="max",
    cbar_label="%",
    title="Relative STD2 - measurement uncertainty",
)
