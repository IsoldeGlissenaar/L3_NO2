#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:53:19 2024

@author: Isolde Glissenaar; isolde.glissenaar@knmi.nl

Functions to retrieve temporal mean in L3 generation 
of NO2 columns.
"""

import calendar
import numpy as np
import xarray as xr
import os
from glob import glob
import datetime
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
            ds_out[var_dict['out_name']] = xr.DataArray(data = np.full((data.sizes['vertical'],
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
        layer_dim = int(data.sizes['vertical'])
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



def get_uncertainty(ds,weights,files,uncertainty_vars,corr_coef_uncer,split_hems=True):
    """
    Get standard deviation and propagated measurement uncertainty for 
    temporal mean of L3 NO2 columns.

    Parameters
    ----------
    ds : xr Dataset
        xarray dataset with loaded no2 superobservations.
    weights : float32
        weights used for calculating weighted
        mean.
    files : list
        List of filenames of superobservations.
    uncertainty_vars : dict
        Dictionary with variables needed to determine uncertainty.
    corr_coef_uncer : dict
        Settings dictionary with correlation coefficients.
    split_hems : bool
        True/False calculate with split hemispheres (only possible when
        latitude dimension is an even number), can be used to relieve working
        memory of hardware.

    Returns
    -------
    ds : xr Dataset
        Same as input ds but now with added variables for uncertainty.

    """
    if split_hems==True:
        ds_SH = get_uncertainty_superobs(files,uncertainty_vars,region='SH')
        ds_SH['weighted_mean'] = ds.sel(latitude=slice(-90,0))['no2']
        ds_SH['std1'], ds_SH['temporal_rep'] = standev1(ds_SH,weights[:,:450,:],files)
        ds_SH = ds_SH.drop_vars(["no2","no_superobs","weighted_mean"])
        ds_SH['std2'], ds_SH['std3'], ds_SH['scd_uncer'] = standev2(ds_SH,ds.sel(latitude=slice(-90,0)),weights[:,:450,:],corr_coef_uncer)
        # ds_SH['random'], ds_SH['systematic'] = random_sys(ds_SH,ds.sel(latitude=slice(-90,0)),weights[:,:450,:],corr_coef_uncer)
        
        ds_NH = get_uncertainty_superobs(files,uncertainty_vars,region='NH')
        ds_NH['weighted_mean'] = ds.sel(latitude=slice(0,90))['no2']
        ds_NH['std1'], ds_NH['temporal_rep'] = standev1(ds_NH,weights[:,450:,:],files)
        ds_NH = ds_NH.drop_vars(["no2","no_superobs","weighted_mean"])
        ds_NH['std2'], ds_NH['std3'], ds_NH['scd_uncer'] = standev2(ds_NH,ds.sel(latitude=slice(0,90)),weights[:,450:,:],corr_coef_uncer)
        # ds_NH['random'], ds_NH['systematic'] = random_sys(ds_NH,ds.sel(latitude=slice(0,90)),weights[:,450:,:],corr_coef_uncer)
        
        ds_uncer = xr.concat([ds_SH,ds_NH], dim="latitude")
    elif split_hems==False:
        ds_uncer = get_uncertainty_superobs(files,uncertainty_vars,region='all')
        ds_uncer['weighted_mean'] = ds.no2
        ds_uncer['std1'], ds_uncer['temporal_rep'] = standev1(ds_uncer,weights,files)
        ds_uncer = ds_uncer.drop_vars(["no2","no_superobs","weighted_mean"])
        ds_uncer['std2'], ds_uncer['std3'], ds_uncer['scd_uncer'] = standev2(ds_uncer,ds,weights,corr_coef_uncer)
        # ds_uncer['random'], ds_uncer['systematic'] = random_sys(ds_uncer,ds,weights,corr_coef_uncer)
        
    ds['std1'] = ds_uncer['std1']
    ds['std2'] = ds_uncer['std2']
    ds['std3'] = ds_uncer['std3']
    ds['scd_uncer'] = ds_uncer['scd_uncer']
    # ds['random'] = ds_uncer['random']
    # ds['systematic'] = ds_uncer['systematic'] 
    ds['std2_total'] = np.sqrt( ds_uncer['std2']**2 + ds_uncer['temporal_rep']**2 )
    ds['std3_total'] = np.sqrt( ds_uncer['std3']**2 + ds_uncer['temporal_rep']**2 )
    return ds


def get_uncertainty_superobs(files,uncertainty_vars,region='all'):
    """
    Load superobservations for separate orbits for given month and put in 
    shared xarray Dataset.
    
    Parameters
    ----------
    files : list
        List of filenames of superobservations.
    uncertainty_vars : dict
        Dictionary of variables needed to determine uncertainty.
    region : str
        Hemisphere (SH or NH) to get dataset for. Defaults to 'all'.

    Returns
    -------
    ds : xarray Dataset
        Dataset with all superobservations for given month.

    """

    ##Create empty Dataset to fill
    #Get array size
    f = os.path.join(files[-1])
    data = xr.open_dataset(f)        
    if region=='all':
        empty_arr = np.full((len(files),int(data.sizes['latitude']),data.sizes['longitude']),np.nan).astype('float32')
    elif region in ['SH','NH']:
        empty_arr = np.full((len(files),int(data.sizes['latitude']/2),data.sizes['longitude']),np.nan).astype('float32')
    ds = xr.Dataset(data_vars = {})
    for var in uncertainty_vars:
        var_dict = uncertainty_vars[var]
        ds[var_dict['out_name']] = xr.DataArray(data=np.copy(empty_arr),
                                                dims = ['time','latitude','longitude'])
    
    ## Fill dataset with orbits
    c1=0
    for f in files:
        # checking if it is a file
        if os.path.isfile(f):
            data = xr.open_dataset(f)         
            if region=='SH':
                data = data.sel(latitude=slice(-90,0))
            elif region=='NH':
                data = data.sel(latitude=slice(0,90))                  
            valid = (data.covered_area_fraction.values<=1.1)
            for var in uncertainty_vars:
                var_dict = uncertainty_vars[var]
                ds[var_dict['out_name']].values[c1,:,:][valid] = data[var].values[valid]*var_dict['conversion']
            c1=c1+1
    
    #Get number of observations per superobs gridcell
    ds['no_superobs'] = xr.DataArray(data = np.sum((~np.isnan(ds[var_dict['out_name']])),axis=0).astype('int'),
                                     dims = ['latitude','longitude']
                                     )
    ds['latitude'] = xr.DataArray(data = data.latitude,
                                     dims = ['latitude']
                                     )
    ds['longitude'] = xr.DataArray(data = data.longitude,
                                     dims = ['longitude']
                                     )
    
    return ds



def calc_corr_uncorr_uncer(weights, sigma, c):
    """
    Calculate uncertainty propagation that is
    partly correlated with correlation fraction
    c.
    
    Parameters
    ----------
    weights : float32
        weights used for averaging.
    sigma : float32
        uncertainty to propagate.
    c : float
        correlation fraction.

    Returns
    -------
    total : float32
        propagated error.
        
    """
    uncor = 1/(np.nansum(weights,axis=0))**2*(np.nansum(((weights**2)*(sigma**2)),axis=0)) 
    cor = 1/(np.nansum(weights,axis=0))**2*(np.nansum((weights*sigma),axis=0)**2) 
    total = np.sqrt((1-c)*uncor+c*cor)
    return total


def standev1(ds,weights,files):
    """
    Calculate temporal uncertainty.
    
    Parameters
    ----------
    ds : xr Dataset
        xarray Dataset with superobservation
        orbits.
    weights : float32
        weights used for averaging.
    files : list of strings
        list of filenames.
    
    Returns
    -------
    std1 : float32
        temporal uncertainty.
    temporal_rep : xr Dataarray
        temporal representativity
        uncertainty. 
    """


    #Get number of days in month
    month = files[0].split('/')[-2][-2:]
    if month in ['01','03','05','07','08','10','12']:
        N = 31.
    elif month in ['04','06','09','11']:
        N = 30.
    elif month=='02':
        if np.mod(int(month),4)==0:  #If leap year
            N = 29.
        else:
            N = 28.
           
    #Get day of observation
    days = []
    for f in files:
        days.append(f.split('/')[-1][6:8])
    days = np.array(days)
    
    #Get number of days with valid observation
    values = (~np.isnan(ds['no2'].values))
    n = np.full((ds.sizes['latitude'],ds.sizes['longitude']),0)
    for i in range(values.shape[1]):
        for j in range(values.shape[2]):
            valid_days = days[values[:,i,j]]
            n[i,j] = len(np.unique(valid_days))
    
    #Fix obs-1 problem for only 1 observation
    no_superobs = ds['no_superobs'].values
    no_superobs[no_superobs==1] = 2
    std1 = np.sqrt(( np.nansum( weights*(ds['no2'].values-ds['weighted_mean'].values)**2,axis=0 ) )/
                   ( (no_superobs-1) * np.nansum(weights,axis=0)) )
    
    temporal_rep = std1/np.sqrt(n) * np.sqrt( (N-n)/(N-1) )
    # temporal_rep = std1 / np.sqrt(ds['no_superobs'].values)
    return xr.DataArray(data = std1.astype('float32'), dims = ['latitude','longitude']), xr.DataArray(data = temporal_rep.astype('float32'), dims = ['latitude','longitude'])


def standev2(ds,ds_in,weights,corr_coef_uncer):
    """
    Calculate propoagated measurement uncertainty.
    
    Parameters
    ----------
    ds : xr Dataset
        xarray Dataset with superobservation
        orbits.
    weights : array, float32
        weights used for averaging.
    corr_coef_uncer : list
        correlation factor of uncertainty components in temporal uncertainty 
        propagation.
    
    Returns
    -------
    std2 : array, float32
        measurement uncertainty.
    """
    sigma_amf_w = calc_corr_uncorr_uncer(weights, ds['sigma_amf'], corr_coef_uncer['c_amf'])
    sigma_sc_w = calc_corr_uncorr_uncer(weights, ds['sigma_sc'], corr_coef_uncer['c_scd'])
    sigma_strat_w = calc_corr_uncorr_uncer(weights, ds['sigma_strat'], corr_coef_uncer['c_strat'])
    sigma_re_w = calc_corr_uncorr_uncer(weights, ds['sigma_re'], corr_coef_uncer['c_strat'])
    std2 = np.sqrt(sigma_amf_w**2+sigma_sc_w**2+sigma_strat_w**2+sigma_re_w**2)
    std3 = np.sqrt(sigma_amf_w**2+sigma_sc_w**2+sigma_strat_w**2+sigma_re_w**2+
                   (0.1*ds_in.no2.values)**2)
    return (xr.DataArray(data = std2, dims = ['latitude','longitude']), 
            xr.DataArray(data = std3, dims = ['latitude','longitude']),
            xr.DataArray(data = sigma_sc_w, dims=["latitude","longitude"])
            )


def random_sys(ds,ds_in,weights,corr_coef_uncer):
    """
    Calculate random and systematic components of uncertainty. Random component
    includes the slant column uncertainty, systematic component includes the 
    stratospheric error and the AMF error.
    
    Parameters
    ----------
    ds : xr Dataset
        xarray Dataset with superobservation
        orbits.
    weights : array, float32
        weights used for averaging.
    corr_coef_uncer : list
        correlation factor of uncertainty components in temporal uncertainty 
        propagation.
    
    Returns
    -------
    random : array, float32
        random component of uncertainty.
    systematic : array, float32
        systematic component of uncertainty.
    """
    sigma_amf_w = calc_corr_uncorr_uncer(weights, ds['sigma_amf'], corr_coef_uncer['c_amf'])
    sigma_sc_w = calc_corr_uncorr_uncer(weights, ds['sigma_sc'], corr_coef_uncer['c_scd'])
    sigma_strat_w = calc_corr_uncorr_uncer(weights, ds['sigma_strat'], corr_coef_uncer['c_strat'])
    
    random = np.sqrt(sigma_sc_w**2)
    systematic = np.sqrt(sigma_strat_w**2+sigma_amf_w**2)
    return xr.DataArray(data = random, dims = ['latitude','longitude']), xr.DataArray(data = systematic, dims = ['latitude','longitude'])


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

    time = np.full((ds.sizes['latitude'],ds.sizes['longitude']),np.nan).astype('datetime64[D]')
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
                        time[i,j] = f'{date[0:4]}-{date[4:6]}-{"{:02d}".format(int(np.round(mean_day[i2,j2])))}'
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

    ds['eff_time'] = xr.DataArray(data=time,
                              dims = ['latitude','longitude']
                              )
    ds['local_time'] = xr.DataArray(data=mean_local_time/86400.,
                              dims = ['latitude','longitude']
                              )

    return ds


def get_attrs(date,ds_out,main_sets):
    """
    Get attributes needed for output dataset

    Parameters
    ----------
    date : str
        Date used to retrieve L3 data for.
    ds_out : xarray Dataset
        Dataset with monthly mean values.

    Returns
    -------
    attrs : dict
        Dictionary filled with information for the attributes for the L3 dataset.

    """
    #Get geospatial resolution
    geospatial_lat_res = np.round(ds_out.latitude.values[5]-ds_out.latitude.values[4],1)
    geospatial_lon_res = np.round(ds_out.longitude.values[5]-ds_out.longitude.values[4],1)

    #Get array of available dates in month
    year = int(date[:4])
    month = int(date[4:6])
    last_day = str(calendar.monthrange(year, month)[1]+1)
    dates = np.arange(int(date+'01'),int(date+last_day),1).astype(str)
    
    #Get list of superobservation orbit files
    files = []
    for date in dates:
        in_folder = f'/net/pc200252/nobackup_1/users/gomenrt/no2_tropomi/PAL_reduced/{date[:6]}/'
        files.extend(sorted(os.listdir(in_folder)))

    time_coverage_list = " ".join([file[20:35] for file in files])
    datetime_start = datetime.datetime.strptime(files[0][20:35],'%Y%m%dT%H%M%S') - datetime.datetime(1995,1,1)
    datetime_start = datetime_start.days + datetime_start.seconds/60/60/24
    datetime_stop = datetime.datetime.strptime(files[-1][36:51],'%Y%m%dT%H%M%S') - datetime.datetime(1995,1,1)
    datetime_stop = datetime_stop.days + datetime_stop.seconds/60/60/24
    date_created = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")+'Z'
    
    if int(date)>202106:
        L1_version = '2.0'
    else:
        L1_version = '1.0'
    
    attrs = {'files': files,
             'datetime_start': datetime_start,
             'datetime_stop': datetime_stop,
             'time_coverage_list': time_coverage_list,
             'date_created': date_created,
             'geospatial_lat_resolution': geospatial_lat_res,
             'geospatial_lon_resolution': geospatial_lon_res,
             'L1_version': L1_version,
             'L3_out_version': main_sets['L3_out_version']
             }
    return attrs


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
        xarray Dataset with superobservation
        orbits.
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
        f = '/nobackup/users/glissena/data/TROPOMI/aux/land_water_classification_02x02.nc'
    elif resolution == 1.:
        f = '/nobackup/users/glissena/data/TROPOMI/aux/land_water_classification_1x1.nc'
    elif resolution == 2.:
        f = '/nobackup/users/glissena/data/TROPOMI/aux/land_water_classification_20x25.nc'
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
    return(ds)


def output_dataset(ds,attrs,variables_2d,variables_1d,corr_coef_uncer,files,out_filename):
    """
    Create output dataset.
    
    Parameters
    ----------
    ds : xr Dataset
        xarray Dataset with superobservation
        orbits.
    attrs : dictionary
        Dictionary with attributes to add to output.
    variables_2d : dictionary
        Settings dictionary for 2D variables.
    variables_1d : dictionary
        Settings dictionary for 1D variables.
    corr_coef_uncer : dictionary
        Settings dictionary with correlation 
        coefficient values for uncertainty 
        calculation.
    files : list
        List with filenames for superobservations.
    
    Returns
    -------
    ds2 : float32
        output dataset for saving.
    """

    #Create dataset with main values and attributes
    ds2 = xr.Dataset(data_vars = {
        'tropospheric_NO2_column_number_density' : 
                              xr.DataArray(data = np.expand_dims(ds.no2.values,axis=0),
                                           dims = ['time','latitude','longitude'],
                                           attrs = {'description':'NO2 troposhperic vertical column number density',
                                                    'long_name':'NO2 VCD',
                                                    'standard_name':'troposphere_mole_content_of_nitrogendioxide',
                                                    'units':'molec/cm^2',
                                                    }    
                                           ),
        'tropospheric_NO2_column_number_density_temporal_std' : 
                              xr.DataArray(data = np.expand_dims(ds.std1.values,axis=0),
                                           dims = ['time','latitude','longitude'],
                                           attrs = {'description':'Uncertainty on the NO2 tropospheric vertical column'+
                                                                  ' number density associated with standard deviation of'+
                                                                  ' L2 input data (sigma_1) within the cell',
                                                    'long_name':'temporal standard deviation',
                                                    'units':'molec/cm^2',
                                                    }
                              ),
        'tropospheric_NO2_column_number_density_measurement_uncertainty_kernel' : 
                              xr.DataArray(data = np.expand_dims(ds.std2.values,axis=0),
                                           dims = ['time','latitude','longitude'],
                                           attrs = {'description':'Uncertainty on the NO2 tropospheric vertical column'+
                                                                  ' number density associated with area-averaged propagated'+
                                                                  ' uncertainty of L2 input data, without the profile'+
                                                                  ' uncertainty contribution (sigma_3)',
                                                    'long_name':'NO2 VCD uncertainty kernel',
                                                    'units':'molec/cm^2',
                                                    }
                                           ),
        'tropospheric_NO2_column_number_density_measurement_uncertainty' : 
                              xr.DataArray(data = np.expand_dims(ds.std3.values,axis=0),
                                           dims = ['time','latitude','longitude'],
                                           attrs = {'description':'Uncertainty on the NO2 tropospheric vertical column'+
                                                                  ' number density associated with area-averaged propagated'+
                                                                  ' uncertainty of L2 input data (sigma_2)',
                                                    'long_name':'NO2 VCD uncertainty',
                                                    'units':'molec/cm^2',
                                                    }
                                           ),
        'tropospheric_NO2_column_number_density_total_uncertainty_kernel' : 
                              xr.DataArray(data = np.expand_dims(ds.std2_total.values,axis=0),
                                           dims = ['time','latitude','longitude'],
                                           attrs = {'description':'Total uncertainty on the NO2 tropospheric vertical column'+
                                                                  ' number density associated with time-averaged propagated'+
                                                                  ' uncertainty of L2 input data and temporal representativity, '+
                                                                  'without the profile uncertainty contribution',
                                                    'long_name':'NO2 VCD total uncertainty kernel',
                                                    'units':'molec/cm^2',
                                                    }
                                           ),
        'tropospheric_NO2_column_number_density_total_uncertainty' : 
                              xr.DataArray(data = np.expand_dims(ds.std3_total.values,axis=0),
                                           dims = ['time','latitude','longitude'],
                                           attrs = {'description':'Total uncertainty on the NO2 tropospheric vertical column'+
                                                                  ' number density associated with time-averaged propagated'+
                                                                  ' uncertainty of L2 input data and temporal representativity',
                                                    'long_name':'NO2 VCD total uncertainty',
                                                    'units':'molec/cm^2',
                                                    }
                                           ),
        'NO2_slant_column_number_density_uncertainty' : 
                              xr.DataArray(data = np.expand_dims(ds.scd_uncer.values,axis=0),
                                           dims = ['time','latitude','longitude'],
                                           attrs = {'description':'NO2 slant column number density uncertainty',
                                                    'long_name':'NO2 SCDE',
                                                    'units':'molec/cm^2',
                                                    }
                                           ),    
        # 'tropospheric_NO2_column_number_density_uncertainty_random' : 
        #                       xr.DataArray(data = np.expand_dims(ds.random.values,axis=0),
        #                                    dims = ['time','latitude','longitude'],
        #                                    attrs = {'description':'Random uncertainty on the NO2 tropospheric '+
        #                                             'vertical column number density (slant column density)',
        #                                             'long_name':'NO2 VCD random uncertainty',
        #                                             'units':'molec/cm^2',
        #                                             }       
        #                                    ),
        # 'tropospheric_NO2_column_number_density_uncertainty_systematic' : 
        #                       xr.DataArray(data = np.expand_dims(ds.systematic.values,axis=0),
        #                                    dims = ['time','latitude','longitude'],
        #                                    attrs = {'description':'Systematic uncertainty on the NO2 tropospheric '+
        #                                             'vertical column number density (AMF and stratospheric column)',
        #                                             'long_name':'NO2 VCD systematic uncertainty',
        #                                             'units':'molec/cm^2',
        #                                             }
        #                       ),                             

        #
        'time' : xr.DataArray(data = np.array([np.datetime64(f"{attrs['files'][0][20:24]}-{attrs['files'][0][24:26]}-{attrs['files'][0][26:28]} 00:00:00.000000000")]),
                              dims = ['time'],
                              attrs = {'description':'start date of monthly mean',
                                       'standard_name':'time'
                                       }
                                ),
        'latitude' : xr.DataArray(data = ds.latitude.values,
                                  dims = ['latitude'],
                                  attrs = {'units':'degree_north',
                                           'standard_name':'latitude',
                                           'bounds':'latitude_bounds'
                                           }
                                  ),
        'longitude' : xr.DataArray(data = ds.longitude.values,
                                   dims = ['longitude'],
                                   attrs = {'units':'degree_east',
                                            'standard_name':'longitude',
                                            'bounds':'longitude_bounds'
                                            }
                                   ),
        'vertical' : xr.DataArray(data = np.arange(1,35,1).astype('int32'),
                                dims = ['vertical'],
                                attrs = {'units':'1',
                                         'long_name':'vertical dimension index'
                                         }
                                   ),
                                    },
        attrs = {'Conventions':'CF-1.8 HARP-1.0',  
                 'title':'NetCDF CF file providing L3 total nitrogendioxide satellite observations',
                 'institution':'KNMI',
                 'doi':'https://doi.org/10.21944/cci-no2-tropomi-l3',
                 'source':' ',
                 'project':'CCI+ ecv',
                 'summary':'TROPOMI L3 tropospheric nitrogendioxide columns for one individual sensor generated as'+
                           'part of the CCI + percursors',
                 'license':'',
                 'references':'',
                 'naming_authority':'KNMI',  
                 'keywords':'nitrogendioxide, total column, TROPOMI, level-3, satellite',
                 'keywords_vocabulary':'',
                 'cdm_data_type':'Grid',
                 'creator_name':'KNMI',
                 'creator_url':'',
                 'creator_email':'isolde.glissenaar@knmi.nl',
                 'geospatial_lat_min':'-90',
                 'geospatial_lat_max':'+90',
                 'geospatial_lat_units':'degrees_north',
                 'geospatial_lon_min':'-180',
                 'geospatial_lon_max':'+180',
                 'geospatial_lon_units':'degrees_east',
                 'geospatial_vertical_min':'',
                 'geospatial_vertical_max':'',
                 'time_coverage_duration':'P1M',
                 'time_coverage_resolution':'P1M',
                 'standard_name_vocabulary':'CF Standard Name Table v82',
                 'sensor_list':'TROPOMI/Sentinel 5 precursor',
                 'platform':'S5P',
                 'sensor':'TROPOMI',
                 'key_variables':'tropospheric_NO2_column_number_density',   
                 'gridding_software':'super-observations (Rijdsijk et al., 2024)',  
                 'gridding_selection_cloud_fraction_max':'0.3',
                 'gridding_selection_minimum_qa':0.75,
                 'gridding_selection_other_filters':'no descending node',
                 'temporal_uncertainty_correlation_scd':corr_coef_uncer['c_scd'],
                 'temporal_uncertainty_correlation_strat':corr_coef_uncer['c_strat'],
                 'temporal_uncertainty_correlation_amf':corr_coef_uncer['c_amf'],
                 'temporal_uncertainty_correlation_re':corr_coef_uncer['c_re'],
                 'comment':'',
                 'L1_version':attrs['L1_version'],
                 'L2_version':'2.3.1',
                 'tracking_id':'https://doi.org/10.21944/cci-no2-tropomi-l3',
                 'id':out_filename,
                 'product_version':attrs['L3_out_version'],
                 'geospatial_lat_resolution':attrs['geospatial_lat_resolution'],
                 'geospatial_lon_resolution':attrs['geospatial_lon_resolution'],
                 'List_of_L2_files':attrs['files'],
                 'time_coverage_start':attrs['files'][0][20:35],
                 'time_coverage_end':attrs['files'][-1][36:51],
                 'datetime_start':attrs['datetime_start'],
                 'datetime_stop':attrs['datetime_stop'],
                 'time_coverage_list':attrs['time_coverage_list'],
                 'date_created':attrs['date_created']
                 }
        )
        
    #Add extra variables from variables_2d & calc_vars: when get_mean is true, safe into xarray Dataset
    for lis in variables_2d:
        variables = variables_2d[lis]
        for var in variables:
            var_dict = variables[var]
            if var_dict['get_mean']:
                if var_dict['dimension']=='2d':
                    ds2[var_dict['out_name']] = xr.DataArray(data = np.expand_dims(ds[var_dict['out_name']].values,axis=0),
                                                            dims = ['time','latitude','longitude'],
                                                            attrs = var_dict['attrs']
                                                            )
                elif var_dict['dimension']=='3d':
                    ds2[var_dict['out_name']] = xr.DataArray(data = np.transpose(np.expand_dims(
                                                                                    ds[var_dict['out_name']].values,axis=0),
                                                                                 [0,2,3,1]),
                                                            dims = ['time','latitude','longitude','vertical'],
                                                            attrs = var_dict['attrs']
                                                            )
                    
    #Add variables from variables_1d 
    for var in variables_1d:
        var_dict = variables_1d[var]
        ds2 = add_nontime_vars(ds2,files,var,var_dict)
        
    #Add land water mask
    ds2 = add_land_water_mask(ds2, attrs)
    
    #Add lat_bnds and lon_bnds
    lat_bnds_1 = [-90]
    for i in range(len(ds2.latitude.values)-1):
        lat_bnds_1.append(np.round(ds2.latitude.values[i]*2-lat_bnds_1[i],2))
    lat_bnds_2 = lat_bnds_1[1:]
    lat_bnds_2.append(90)
    ds2['latitude_bounds'] = xr.DataArray(data = np.array([lat_bnds_1,lat_bnds_2]).transpose(),
                                          dims = ['latitude','independent_2'],
                                          attrs = {'long_name' : 'grid latitude bounds',
                                                   'units' : 'degree_north'}
                                          )
    # #Regular grid
    lon_bnds_1 = [-180]
    for i in range(len(ds2.longitude.values)-1):
        lon_bnds_1.append(np.round(ds2.longitude.values[i]*2-lon_bnds_1[i],2))
    lon_bnds_2 = lon_bnds_1[1:]
    lon_bnds_2.append(180)
    
    # Res-geoschem (2x2.5)
    # lon_bnds_1 = [178.75,-178.75]
    # for i in range(1,len(ds2.longitude.values)-1):
    #     lon_bnds_1.append(np.round(ds2.longitude.values[i]*2-lon_bnds_1[i],2))
    # lon_bnds_2 = lon_bnds_1[1:]
    # lon_bnds_2.append(178.75)
    
    
    ds2['longitude_bounds'] = xr.DataArray(data = np.array([lon_bnds_1,lon_bnds_2]).transpose(),
                                          dims = ['longitude','independent_2'],
                                          attrs = {'long_name' : 'grid longitude bounds',
                                                   'units' : 'degree_east'}
                                          )
    
    #Effective time
    ds2['eff_date'] = xr.DataArray(data = np.expand_dims(ds.eff_time.values,axis=0),
                                   dims = ['time','latitude','longitude'],
                                   attrs = {'description':'effective date of observation',
                                            'standard_name':'effective date'
                                           }
                                    )
    ds2['eff_frac_day'] = xr.DataArray(data = np.expand_dims(ds.local_time.values,axis=0),
                                       dims = ['time','latitude','longitude'],
                                       attrs = {'description':'effective fractional day in local solar time. '+
                                                                'UTC = local_solar_time - longitude/180',
                                                'standard_name':'effective fractional day'
                                                }
                                         )
        
    return ds2
    




    
    