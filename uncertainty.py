#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:02:41 2023

@author: Isolde Glissenaar

Functions to calculate the associated uncertainties 
of temporal means.
"""

import numpy as np
import xarray as xr
import os


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
        ds_SH['weighted_mean'] = ds.sel(latitude=slice(-90,0))['tropospheric_NO2_column_number_density']
        ds_SH['std1'], ds_SH['temporal_rep'] = standev1(ds_SH,weights[:,:450,:],files)
        ds_SH = ds_SH.drop_vars(["tropospheric_NO2_column_number_density","no_superobs","weighted_mean"])
        ds_SH['std2'], ds_SH['std3'], ds_SH['scd_uncer'] = standev2(ds_SH,ds.sel(latitude=slice(-90,0)),weights[:,:450,:],corr_coef_uncer)
        # ds_SH['random'], ds_SH['systematic'] = random_sys(ds_SH,ds.sel(latitude=slice(-90,0)),weights[:,:450,:],corr_coef_uncer)
        
        ds_NH = get_uncertainty_superobs(files,uncertainty_vars,region='NH')
        ds_NH['weighted_mean'] = ds.sel(latitude=slice(0,90))['tropospheric_NO2_column_number_density']
        ds_NH['std1'], ds_NH['temporal_rep'] = standev1(ds_NH,weights[:,450:,:],files)
        ds_NH = ds_NH.drop_vars(["tropospheric_NO2_column_number_density","no_superobs","weighted_mean"])
        ds_NH['std2'], ds_NH['std3'], ds_NH['scd_uncer'] = standev2(ds_NH,ds.sel(latitude=slice(0,90)),weights[:,450:,:],corr_coef_uncer)
        # ds_NH['random'], ds_NH['systematic'] = random_sys(ds_NH,ds.sel(latitude=slice(0,90)),weights[:,450:,:],corr_coef_uncer)
        
        ds_uncer = xr.concat([ds_SH,ds_NH], dim="latitude")
    elif split_hems==False:
        ds_uncer = get_uncertainty_superobs(files,uncertainty_vars,region='all')
        ds_uncer['weighted_mean'] = ds.tropospheric_NO2_column_number_density
        ds_uncer['std1'], ds_uncer['temporal_rep'] = standev1(ds_uncer,weights,files)
        ds_uncer = ds_uncer.drop_vars(["tropospheric_NO2_column_number_density","no_superobs","weighted_mean"])
        ds_uncer['std2'], ds_uncer['std3'], ds_uncer['scd_uncer'] = standev2(ds_uncer,ds,weights,corr_coef_uncer)
        # ds_uncer['random'], ds_uncer['systematic'] = random_sys(ds_uncer,ds,weights,corr_coef_uncer)
        
    ds['tropospheric_NO2_column_number_density_temporal_std'] = ds_uncer['std1']
    ds['tropospheric_NO2_column_number_density_measurement_uncertainty_kernel'] = ds_uncer['std2']
    ds['tropospheric_NO2_column_number_density_measurement_uncertainty'] = ds_uncer['std3']
    ds['NO2_slant_column_number_density_uncertainty'] = ds_uncer['scd_uncer']
    # ds['random'] = ds_uncer['random']
    # ds['systematic'] = ds_uncer['systematic'] 
    ds['tropospheric_NO2_column_number_density_total_uncertainty_kernel'] = np.sqrt( ds_uncer['std2']**2 + ds_uncer['temporal_rep']**2 )
    ds['tropospheric_NO2_column_number_density_total_uncertainty'] = np.sqrt( ds_uncer['std3']**2 + ds_uncer['temporal_rep']**2 )
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
    values = (~np.isnan(ds['tropospheric_NO2_column_number_density'].values))
    n = np.full((ds.sizes['latitude'],ds.sizes['longitude']),0)
    for i in range(values.shape[1]):
        for j in range(values.shape[2]):
            valid_days = days[values[:,i,j]]
            n[i,j] = len(np.unique(valid_days))
    
    #Fix obs-1 problem for only 1 observation
    no_superobs = ds['no_superobs'].values
    no_superobs[no_superobs==1] = 2
    std1 = np.sqrt(( np.nansum( weights*(ds['tropospheric_NO2_column_number_density'].values-ds['weighted_mean'].values)**2,axis=0 ) )/
                   ( (no_superobs-1) * np.nansum(weights,axis=0)) )
    
    temporal_rep = std1/np.sqrt(n) * np.sqrt( (N-n)/(N-1) )
    return (xr.DataArray(data = std1.astype('float32'), dims = ['latitude','longitude']), 
            xr.DataArray(data = temporal_rep.astype('float32'), dims = ['latitude','longitude']))


def standev2(ds,ds_in,weights,corr_coef_uncer):
    """
    Calculate propoagated measurement uncertainty.
    
    Parameters
    ----------
    ds : xr Dataset
        xarray Dataset with uncertainties.
    ds_in : xr Dataset
        Dataset with superobservation orbits.
    weights : array, float32
        weights used for averaging.
    corr_coef_uncer : list
        correlation factor of uncertainty components in temporal uncertainty 
        propagation.
    
    Returns
    -------
    std2 : array, float32
        measurement uncertainty with kernel.
    std3 : array, float32
        measurement uncertainty.
    sigma_sc_w : array, float32
        slant column density uncertainty.
    """
    sigma_amf_w = calc_corr_uncorr_uncer(weights, ds['sigma_amf'], corr_coef_uncer['c_amf'])
    sigma_sc_w = calc_corr_uncorr_uncer(weights, ds['sigma_sc'], corr_coef_uncer['c_scd'])
    sigma_strat_w = calc_corr_uncorr_uncer(weights, ds['sigma_strat'], corr_coef_uncer['c_strat'])
    sigma_re_w = calc_corr_uncorr_uncer(weights, ds['sigma_re'], corr_coef_uncer['c_re'])
    std2 = np.sqrt(sigma_amf_w**2+sigma_sc_w**2+sigma_strat_w**2+sigma_re_w**2)
    std3 = np.sqrt(sigma_amf_w**2+sigma_sc_w**2+sigma_strat_w**2+sigma_re_w**2+
                   (0.1*ds_in.tropospheric_NO2_column_number_density.values)**2)
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
    ds_in : xr Dataset
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
    return (xr.DataArray(data = random, dims = ['latitude','longitude']), 
            xr.DataArray(data = systematic, dims = ['latitude','longitude']))

