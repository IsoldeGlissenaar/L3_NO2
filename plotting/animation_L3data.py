#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:33:19 2024

@author: glissena
"""
import calendar
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation, FFMpegWriter


dates_all = np.concatenate(
    [
        np.arange(201805, 201813, 1),
        np.arange(201901, 201913, 1),
        np.arange(202001, 202013, 1),
        np.arange(202101, 202113, 1),
    ]
).astype(str)

data = np.full((len(dates_all), 900, 1800), np.nan)
for i, date in enumerate(dates_all):
    f = f"/nobackup/users/glissena/data/TROPOMI/out_L3/02x02/CCI+p-L3-NO2_TC-TROPOMI_S5P_v020301-KNMI-{date}-fv0100.nc"
    ds = xr.open_dataset(f)
    dates = f[57:63]

    data[i, :, :] = ds.tropospheric_NO2_column_number_density.values


fig, ax = plt.subplots(dpi=400, subplot_kw={"projection": ccrs.PlateCarree()})
ax.coastlines(resolution="50m", linewidth=0.3)
im = ax.pcolormesh(
    ds.longitude,
    ds.latitude,
    data[0, :, :] / 1e15,
    vmin=0,
    vmax=10,
    cmap="Spectral_r",
    transform=ccrs.PlateCarree(),
)
cbar = plt.colorbar(im, ax=ax, shrink=0.6, extend="both")
cbar.ax.tick_params(labelsize=6)
cbar.set_label("10$^{15}$ molecules/cm$^2$", size=6)


def animate(i, ds, data, dates_all):
    ax.clear()
    ax.coastlines(resolution="50m", linewidth=0.3)
    ax.text(
        0.93,
        0.035,
        f"{calendar.month_name[int(dates_all[i][4:])]} {dates_all[i][:4]}",
        horizontalalignment="right",
        transform=ax.transAxes,
    )
    ax.pcolormesh(
        ds.longitude,
        ds.latitude,
        data[i, :, :] / 1e15,
        vmin=0,
        vmax=10,
        cmap="Spectral_r",
        transform=ccrs.PlateCarree(),
    )


# set ani variable to call the
# function recursively
anim = FuncAnimation(
    fig,
    animate,
    interval=10000,
    frames=len(dates_all),
    fargs=[ds, data, dates_all],
    repeat=True,
)

mywriter = FFMpegWriter(fps=1)
anim.save("/usr/people/glissena/Documents/figures/animation_0_2.mp4", writer=mywriter)
