#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:17:57 2024

@author: glissena
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def world_plot(
    data,
    lon,
    lat,
    region="world",
    vmin=0,
    vmax=10,
    cmap="Spectral_r",
    extend="neither",
    cbar_label="",
    title="",
):
    fig = plt.figure(dpi=400)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=0.3)
    if region.lower() == "world":
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    if region.lower() == "europe":
        ax.set_extent([-15, 40, 30, 70], crs=ccrs.PlateCarree())
    if region.lower() == "eastasia":
        ax.set_extent([100, 142, 17, 49], crs=ccrs.PlateCarree())
    if region.lower() == "northamerica":
        ax.set_extent([-165, -49, 13.5, 65], crs=ccrs.PlateCarree())
    plt.pcolormesh(
        lon, lat, data, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree()
    )
    cbar = plt.colorbar(shrink=0.6, extend=extend)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(cbar_label, size=6)
    plt.title(title, fontsize=8)
    fig.show()
    return
