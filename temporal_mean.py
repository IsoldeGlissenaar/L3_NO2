#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:02:41 2023

@author: Isolde Glissenaar

Functions to calculate the temporal mean of variables.
"""

import numpy as np
import xarray as xr
import os
from glob import glob
import netCDF4 as nc

def get_list_of_files(date: str, main_sets) -> list:
    """
    Get list of filenames of superorbits.

    Parameters
    ----------
    date : str
        Date in yyyymm.
    main_sets : dict
        Dictionary with main configuration. 

    Returns
    -------
    files : list
        List of filenames of superorbits to run.

    """
    in_folder = f"{main_sets['path_in']}/{main_sets['dataset']}/{date[0:4]}_{date[4:6]}/"
    glob_pattern = os.path.join(in_folder, '*')
    files = sorted(glob(glob_pattern))
    return files


def get_mean_all_vars(variables_2d: dict[str, dict[str, str|bool|float]],
                      files: list,
                      dataset: str,
                      split_hems=True,
                      ):
    """
    Get monthly mean of all superorbits for variables in assigned dictionary.
    
    Parameters
    ----------
    variables_2d : dict
        Dictionary that defines variables to get the monthly mean from.
    files: list
        List of files with relevant superobservations.
    dataset: str
        Dataset of superobservations. 
    split_hems: bool
        True/False calculate with split hemispheres (only possible when
        latitude dimension is an even number), can be used to relieve working
        memory of hardware.
        
    Returns
    -------
    ds_out : xarray Dataset
        Dataset with monthly mean of superobservations for given month.
    weights : float array
        Weights used to take monthly mean, still needed for uncertainty
        calculations.
    
    """
    ds_out = xr.Dataset(data_vars = {})
    for var in variables_2d:
        print(var)
        var_dict = variables_2d[var]
        if var_dict['dimension']=='2d':
            if split_hems==True:
                ##Load variables in two parts to avoid RAM overload.
                ds_SH = get_superobs(files,var,var_dict,dataset=dataset,region='SH')
                lat_SH,lon = ds_SH.latitude,ds_SH.longitude
                weights_SH, weighted_mean_SH = weighted_mean_func(ds_SH,var_dict)
                del ds_SH
                ds_NH = get_superobs(files,var,var_dict,dataset=dataset,region='NH')
                lat_NH = ds_NH.latitude
                weights_NH, weighted_mean_NH = weighted_mean_func(ds_NH,var_dict)
                del ds_NH
                ds_out['latitude'] = xr.concat([lat_SH,lat_NH], dim="latitude")
                ds_out['longitude'] = lon
                ds_out[var_dict['out_name']] = xr.concat([weighted_mean_SH,weighted_mean_NH], dim="latitude")
                del lat_SH,lat_NH,lon,weighted_mean_SH,weighted_mean_NH
            elif split_hems==False:
                ds = get_superobs(files,var,var_dict,dataset=dataset,region='all')
                weights, ds_out[var_dict['out_name']] = weighted_mean_func(ds,var_dict)
                ds_out['latitude'], ds_out['longitude'] = ds.latitude,ds.longitude
        elif var_dict['dimension']=='3d':
            #3D arrays are to big to load in RAM so need to be split up
            #in slices.
            data = xr.open_dataset(files[-1])
            ds_out[var_dict['out_name']] = xr.DataArray(data = np.full((data.sizes['layer'],
                                                                    data.sizes['latitude'],data.sizes['longitude']),
                                                                    np.nan).astype('float32'),
                                                        dims = ['vertical','latitude','longitude'])
            slice_len = int(25)
            while np.mod(data.sizes['latitude'],slice_len)!=0:
                #While this is the case latitude cannot be split up
                #equally with this slice_len, so decrease.
                slice_len=int(slice_len-1)
            for lat_idx in range(int(data.sizes['latitude']/slice_len)):
                print(lat_idx)
                ds = get_superobs(files,var,var_dict,dataset=dataset,region='all',idx=[lat_idx],slice_len=slice_len)
                __, ds_out[var_dict['out_name']].values[:,lat_idx*slice_len:lat_idx*slice_len+slice_len,:] = weighted_mean_func(ds,var_dict,idx=[lat_idx],slice_len=slice_len)
    if split_hems==True:
        weights = np.concatenate([weights_SH,weights_NH],axis=1)
    return(ds_out,weights)



def get_superobs(files: list,
                 var: str,
                 var_dict: dir,
                 dataset: str,
                 region='all',
                 idx=[],
                 slice_len=0):
    """
    Load superobservations for separate orbits for given month and put in 
    shared xarray Dataset.
    
    Parameters
    ----------
    files : list
        List of filenames to get superobservations from.
    var : str
        Name of variable to load superobservations for.
    var_dict : dict
        Settings dictionary for var.
    dataset : str
        Dataset of superobservations.
    region : str
        Hemisphere (SH or NH) to get dataset for. Defaults to 'all'.    
    idx : list
        When dimension of variable is 3d, idx gives
        the idx where the slice starts. Defaults to empty 
        list.
    slice_len : int
        When dimension of variable is 3d, gives the
        length of the slices. Defaults to zero.

    Returns
    -------
    ds : xarray Dataset
        Dataset with all superobservations for given month.
    
    """        
    #Get array size
    data = xr.open_dataset(files[-1])            

    #Create empty Dataset to fill
    ds = xr.Dataset(data_vars = {})
    if region=='all':
        lat_dim = int(data.sizes['latitude'])
    elif region in ['SH','NH']:
        lat_dim = int(data.sizes['latitude']/2)
    
    if var_dict['dimension']=='2d':
        ds[var_dict['out_name']] = xr.DataArray(data = np.full((len(files),lat_dim,data.sizes['longitude']),
                                                            np.nan).astype('float32'),
                                                dims = ['time','latitude','longitude'])
    elif var_dict['dimension']=='3d':
        layer_dim = int(data.sizes['layer'])
        ds[var_dict['out_name']] = xr.DataArray(data = np.full((len(files),layer_dim,slice_len,data.sizes['longitude']),
                                                            np.nan).astype('float32'),
                                                dims = ['time','vertical','latitude2','longitude'])
        
    ds['re_rel'] = xr.DataArray(data = np.full((len(files),lat_dim,data.sizes['longitude']),
                                                            np.nan).astype('float32'),
                                dims = ['time','latitude','longitude'])
        
    ## Fill dataset with orbits
    c1=0
    for f in files:
        file = nc.Dataset(f)
        valid = (file['/covered_area_fraction'][()]<=1.1)
        rerel = file['/no2_superobs_re_rel'][()]
        data = file[f'/{var}'][()]
        if region=='SH':
            lat = file['/latitude'][()]
            idxlat = np.where(lat<=0)[0][-1]+1
            valid = valid[:idxlat,:]; rerel=rerel[:idxlat,:]
            data = data[:idxlat,:]
        if region=='NH':
            lat = file['/latitude'][()]
            idxlat = np.where(lat>0)[0][0]
            valid = valid[idxlat:,:]; rerel=rerel[idxlat:,:]
            data = data[idxlat:,:]
        if var_dict['dimension']=='2d':
            ds[var_dict['out_name']].values[c1,:,:][valid] = data[valid]*var_dict['conversion']
            ds['re_rel'].values[c1,:,:][valid] = rerel[valid]
        elif var_dict['dimension']=='3d':
            ds[var_dict['out_name']].values[c1,:,:,:] = data[:,idx[0]*slice_len:idx[0]*slice_len+slice_len,:]*var_dict['conversion']
            ds['re_rel'].values[c1,:,:][valid] = rerel[valid]
        c1=c1+1
        
    
    #Add longitude and latitude
    lat = file['/latitude'][()]
    lon = file['/longitude'][()]
    if region=='SH':
        idxlat = np.where(lat<=0)[0][-1]+1
        lat = lat[:idxlat]
    if region=='NH':
        idxlat = np.where(lat>0)[0][0]
        lat = lat[idxlat:]
    
    file.close()

    ds['latitude'] = xr.DataArray(data = lat,
                                     dims = ['latitude']
                                     )
    ds['longitude'] = xr.DataArray(data = lon,
                                     dims = ['longitude']
                                     )

    return ds



def weighted_mean_func(ds,var_dict,idx=[],slice_len=0):
    """
    Calculate weighted mean.
    
    Parameters
    ----------
    ds : xr Dataset
        xarray Dataset with superobservation.
        orbits.
    var_dict : dict
        Settings dictionary for variable.    
    idx : list
        When dimension of variable is 3d, idx gives
        the idx where the slice starts.
    slice_len : int
        When dimension of variable is 3d, gives the
        length of the slices.

    Returns
    -------
    weights : float32
        weights used for calculating weighted
        mean.
    weighted_mean : float32
        weighted mean.
        
    """
    weights = (1-ds['re_rel'].values).astype('float32')
    if var_dict['dimension']=='2d':
        weighted_mean = np.full((weights.shape[1],weights.shape[2]),np.nan)
        zero_weight = (np.nansum(weights,axis=0)==0)
        weighted_mean[~zero_weight] = np.nansum(weights*ds[var_dict['out_name']].values,axis=0)[~zero_weight]/np.nansum(weights,axis=0)[~zero_weight]
        weighted_mean = xr.DataArray(data = weighted_mean, dims = ['latitude','longitude'])
    elif var_dict['dimension']=='3d':
        weights_3d = np.transpose(np.tile(np.expand_dims(weights[:,idx[0]*slice_len:idx[0]*slice_len+slice_len,:],axis=3),34),(0,3,1,2))
        weighted_mean = np.full((weights_3d.shape[1],weights_3d.shape[2],weights_3d.shape[3]),np.nan)
        zero_weight = (np.nansum(weights_3d,axis=0)==0)
        weighted_mean[~zero_weight] = np.nansum(weights_3d*ds[var_dict['out_name']].values,axis=0)[~zero_weight]/np.nansum(weights_3d,axis=0)[~zero_weight]

    return weights, weighted_mean


