#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:02:41 2023

@author: Isolde Glissenaar

Functions to calculate extra variables for 
the temporal mean.
"""

import numpy as np
import xarray as xr
import os



def add_count(ds,files,date):
    """
    Add count to monthly mean.

    Parameters
    ----------
    ds : xr dataset
        Dataset with monthly mean values to add to.
    files : list
        List of filenames of superobservations.
    date : str
        Date to get monthly mean for in yyyymm

    Returns
    -------
    ds : xr dataset
        Same as input ds but now with added calculated variables.
    """
    cover = np.full((len(files),ds.sizes['latitude'],ds.sizes['longitude']),np.nan)
    no2 = np.full((len(files),ds.sizes['latitude'],ds.sizes['longitude']),np.nan)
    for i,file in enumerate(files):
        data = xr.open_dataset(file)
        cover[i,:,:] = data.covered_area_fraction.values
        no2[i,:,:] = data.no2_superobs.values
    cover[cover>1e36] = np.nan
    no2[no2>1e36] = np.nan
    

    #Get number of days in month
    if date[4:6] in ['01','03','05','07','08','10','12']:
        len_m = 31.
    elif date[4:6] in ['04','06','09','11']:
        len_m = 30.
    elif date[4:6]=='02':
        if np.mod(int(date[0:4]),4)==0:  #If leap year
            len_m = 29.
        else:
            len_m = 28.
           
    ds['no_observations'] = xr.DataArray(data=np.sum((~np.isnan(no2)),axis=0).astype('int32'),
                                     dims = ['latitude','longitude']
                                     )

    ds['tropospheric_NO2_column_number_density_count'] = xr.DataArray(data=np.nansum(cover,axis=0)/len_m,
                                                                      dims = ['latitude','longitude']
                                                                     )  
    
    return ds
    


def add_vars(ds, calc_vars):
    """
    Add variables that need to be calculated.

    Parameters
    ----------
    ds : xr dataset
        Dataset with monthly mean values to add to.
    calc_vars : dict
        Dictionary of variables that need to be calculated.

    Returns
    -------
    ds : xr dataset
        Same as input ds but now with added calculated variables.

    """
    for var in calc_vars:
        var_dict = calc_vars[var]
        if var_dict['do_func']:
            print(var)
            ds[var_dict['out_name']] = xr.DataArray(data=eval(var_dict['func']),
                                                    dims = ['latitude','longitude'])    
    return ds       



def add_time(ds,files,date,weights,split_hems=False):
    """
    Add effective date and effective time of day
    to monthly mean. 

    Parameters
    ----------
    ds : xr dataset
        Dataset with monthly mean values to add to.
    files : list
        List of filenames of superobservations.
    date : str
        Date to get monthly mean for in yyyymm.
    weights : float
        Weights used to take monthly mean.
    split_hems : bool
        True/False calculate with split hemispheres (only possible when
        latitude dimension is an even number), can be used to relieve working
        memory of hardware.

    Returns
    -------
    ds : xr dataset
        Same as input ds but now with added calculated variables.
    """

    split_lon=True
    
    if split_hems:
        regions = ['SH','NH']
        lat_idx = np.array([[0,ds.sizes['latitude']/2],[ds.sizes['latitude']/2,ds.sizes['latitude']]]).astype(int)
    else:
        regions = ['all']
        lat_idx = np.array([[0,ds.sizes['latitude']],[-9999,-9999]]).astype(int)
        
    if split_lon:
        regions_lon = ['WH','EH']
        lon_idx = np.array([[0,ds.sizes['longitude']/2],[ds.sizes['longitude']/2,ds.sizes['longitude']]]).astype(int)
    else:
        regions_lon = ['all']
        lon_idx = np.array([[0,ds.sizes['longitude']],[-9999,-9999]]).astype(int)

    time = np.full((ds.sizes['latitude'],ds.sizes['longitude']),np.nan)
    mean_local_time = np.full((ds.sizes['latitude'],ds.sizes['longitude']),np.nan)

    for d,region in enumerate(regions):
        for d2,region_lon in enumerate(regions_lon):
            #Load delta_time
            delta_time = np.full((len(files),(lat_idx[d,1]-lat_idx[d,0]),lon_idx[d2,1]-lon_idx[d2,0]),np.nan).astype('datetime64[s]')
            for i,file in enumerate(files):
                data = xr.open_dataset(file)
                if region=='SH':
                    data = data.sel(latitude=slice(-90,0))
                elif region=='NH':
                    data = data.sel(latitude=slice(0,90))  
                if region_lon=='WH':
                    data = data.sel(longitude=slice(-180,0))
                elif region_lon=='EH':
                    data = data.sel(longitude=slice(0,180))
                delta_time[i,:,:] = data.delta_time.values.astype('datetime64[s]')
                lon = data.longitude.values
            del data
    
            #Add effective day of month
            day = np.full(delta_time.shape,np.nan).astype('float32')
            nonan_time = ~np.isnat(delta_time)
            day[nonan_time] = np.array([d[8:10] for d in delta_time[nonan_time].astype('datetime64[D]').astype(str)]).astype('float32')
            
            weights_r = weights[:,lat_idx[d,0]:lat_idx[d,1],lon_idx[d2,0]:lon_idx[d2,1]]
            mean_day = np.full((weights_r.shape[1],weights_r.shape[2]),np.nan)
            zero_weight = (np.nansum(weights_r,axis=0)==0)
            mean_day[~zero_weight] = np.nansum(weights_r*day,axis=0)[~zero_weight]/np.nansum(weights_r,axis=0)[~zero_weight]
            for i in range(lat_idx[d,0],lat_idx[d,1]):
                for j in range(lon_idx[d2,0],lon_idx[d2,1]):
                    i2 = i-lat_idx[d,0]
                    j2 = j-lon_idx[d2,0]
                    if ~np.isnan(mean_day[i2,j2]):
                        time[i,j] = np.round(mean_day[i2,j2])
            del day,mean_day
    
            #Add effective time of day 
            time_of_day = np.full(delta_time.shape,np.nan)
            time_of_day[nonan_time] = np.array([float(t[11:13])*60*60+float(t[14:16])*60+float(t[17:19]) for t in delta_time[nonan_time].astype(str)])
            time_of_day[time_of_day>86400] = time_of_day[time_of_day>86400]-86400
            local_time = time_of_day+lon/180*12*60*60
            local_time[local_time<0] = 86400+local_time[local_time<0]
            local_time[local_time>86400] = local_time[local_time>86400]-86400
            mean_local_time[lat_idx[d,0]:lat_idx[d,1],lon_idx[d2,0]:lon_idx[d2,1]][~zero_weight] = np.nansum(weights_r*local_time,axis=0)[~zero_weight]/np.nansum(weights_r,axis=0)[~zero_weight]
            del time_of_day,local_time,weights_r,zero_weight,nonan_time,lon
            
    t = np.array([np.datetime64(f"{date[0:4]}-{date[4:6]}-01 00:00:00.000000000")])
    t = t[0].astype('datetime64[D]')-np.datetime64('1995-01-01')
    t = t.astype('double')
                 
    ds['eff_date'] = xr.DataArray(data=time.astype('double'),
                              dims = ['latitude','longitude']
                              )
    ds['eff_frac_day'] = xr.DataArray(data=mean_local_time/86400.,
                              dims = ['latitude','longitude']
                              )
    ds['time'] = xr.DataArray(data = [t],
                              dims = ['time']
                              )
    return ds



def add_nontime_vars(ds,files,var,var_dict):
    """
    Add non-time dependent variables, e.g. tm5_sigma_a
    and tm5_sigma_b.

    Parameters
    ----------
    ds : xr Dataset
        Dataset with monthly mean values to add to.
    files : list
        List of superobservation files.
    var : str
        Name of variable to add.
    var_dict : dict
        Settings dictionary of added variable.
    
    Returns
    -------
    ds : float32
        Same as input ds but with added variable.
    """
    f = os.path.join(files[-1])
    data = xr.open_dataset(f)       
    ds[var_dict['out_name']] = xr.DataArray(data = data[var].values*var_dict['conversion'],
                                            dims = ['vertical','independent_2'],
                                            attrs = var_dict['attrs']
                                            )
    return ds



def add_land_water_mask(ds,attrs):
    """
    Load auxiliary land_water_mask and add to output 
    dataset. For now only available for 0.2x0.2 degree
    and 1x1 degree grids. More can be generated with
    land_water_mask.py.
    
    Parameters
    ----------
    ds : xr Dataset
        xarray Dataset output without land water mask.
    attrs : dictionary
        Dictionary with attributes to add to output.
    
    Returns
    -------
    ds : float32
        Dataset output including land_water_mask.
    """
    resolution = attrs['geospatial_lat_resolution'].astype('float64')
    resolution = np.round(resolution,1)
    if resolution == 0.2:
        f = 'aux/land_water_classification_02x02.nc'
    elif resolution == 1.:
        f = 'aux/land_water_classification_1x1.nc'
    elif resolution == 2.:
        f = 'aux/land_water_classification_20x25.nc'
    else:
        print(f"resolution: {resolution}")
        print('WARNING: No land_water_mask file available for this resolution')
    lc = xr.open_dataset(f)
    ds['land_water_mask'] = xr.DataArray(data = lc.land_water_mask.values.astype('int32'),
                                          dims = ['time','latitude','longitude'],
                                          attrs = {'description' : '0:water, 1:land, 2:land water transition',
                                                   'long_name' : 'land sea mask',
                                                   'units' : '1'}
                                          )
    return ds



def add_latlon_bnds(ds):
    """
    Add latitude bounds and longitude bounds
    to the dataset. Works for regular grids.

    Parameters
    ----------
    ds : xr Dataset
        xarray Dataset output without lat and lon bounds.

    Returns
    -------
    ds : xr Dataset
        xarray Dataset output with lat and lon bounds.

    """
    lat_bnds_1 = [-90]
    for i in range(len(ds.latitude.values)-1):
        lat_bnds_1.append(np.round(ds.latitude.values[i]*2-lat_bnds_1[i],2))
    lat_bnds_2 = lat_bnds_1[1:]
    lat_bnds_2.append(90)
    ds['latitude_bounds'] = xr.DataArray(data = np.array([lat_bnds_1,lat_bnds_2]).transpose(),
                                          dims = ['latitude','independent_2'],
                                          attrs = {'long_name' : 'grid latitude bounds',
                                                   'units' : 'degree_north'}
                                          )
    # Regular grid
    lon_bnds_1 = [-180]
    for i in range(len(ds.longitude.values)-1):
        lon_bnds_1.append(np.round(ds.longitude.values[i]*2-lon_bnds_1[i],2))
    lon_bnds_2 = lon_bnds_1[1:]
    lon_bnds_2.append(180)
    
    # Res-geoschem (2x2.5)
    # lon_bnds_1 = [178.75,-178.75]
    # for i in range(1,len(ds.longitude.values)-1):
    #     lon_bnds_1.append(np.round(ds.longitude.values[i]*2-lon_bnds_1[i],2))
    # lon_bnds_2 = lon_bnds_1[1:]
    # lon_bnds_2.append(178.75)
        
    ds['longitude_bounds'] = xr.DataArray(data = np.array([lon_bnds_1,lon_bnds_2]).transpose(),
                                          dims = ['longitude','independent_2'],
                                          attrs = {'long_name' : 'grid longitude bounds',
                                                   'units' : 'degree_east'}
                                          )
    return ds
