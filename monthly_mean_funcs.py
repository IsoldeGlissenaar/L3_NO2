#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:53:19 2024

@author: Isolde Glissenaar

Functions to retrieve temporal mean in L3 generation 
of NO2 columns.
"""

import calendar
import numpy as np
import xarray as xr
import os
from glob import glob
import datetime


def get_list_of_files_flat(date,dataset='res_geos_chem'):
    """
    Get list of filenames of superorbits.

    Parameters
    ----------
    date : str
        Date in yyyymm.
    dataset : str, optional
        Dataset to run. The default is 'new_qa'.

    Returns
    -------
    files : list
        List of filenames of superorbits to run.

    """
   
    #Get list of superobservation orbit files
    in_folder = f'/nobackup/users/glissena/data/TROPOMI/L2/superobs/{dataset}/{date[0:4]}_{date[4:6]}/'
    glob_pattern = os.path.join(in_folder, '*')
    files = sorted(glob(glob_pattern), key=os.path.getctime)
    return files


def get_list_of_files(date,dataset='new_qa'):
    """
    Get list of filenames of superorbits.

    Parameters
    ----------
    date : str
        Date in yyyymm.
    dataset : str, optional
        Dataset to run. The default is 'new_qa'.

    Returns
    -------
    files : list
        List of filenames of superorbits to run.

    """
    #Get array of available dates in month
    year = int(date[:4])
    month = int(date[4:6])
    last_day = str(calendar.monthrange(year, month)[1]+1)
    dates = np.arange(int(date+'01'),int(date+last_day),1).astype(str)
    
    #Get list of superobservation orbit files
    files = []
    for date in dates:
        in_folder = f'/nobackup/users/glissena/data/TROPOMI/L2/superobs/{dataset}/{date[0:4]}_{date[4:6]}/{date}'
        # files.extend(os.listdir(in_folder))
        glob_pattern = os.path.join(in_folder, '*')
        files.extend(sorted(glob(glob_pattern), key=os.path.getctime))
    return files



def get_mean_all_vars(variables_2d,files,dataset='new_qa',split_hems=True):
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
        if split_hems==True:
            ds_SH = get_superobs(files,var,var_dict,dataset=dataset,region='SH')
            lat_SH,lon_SH = ds_SH.latitude,ds_SH.longitude
            weights_SH, weighted_mean_SH = weighted_mean_func(ds_SH,var_dict)
            del ds_SH
            ds_NH = get_superobs(files,var,var_dict,dataset=dataset,region='NH')
            lat_NH,lon_NH = ds_NH.latitude,ds_NH.longitude
            weights_NH, weighted_mean_NH = weighted_mean_func(ds_NH,var_dict)
            del ds_NH
            ds_out['latitude'] = xr.concat([lat_SH,lat_NH], dim="latitude")
            ds_out['longitude'] = lon_SH
            ds_out[var_dict['out_name']] = xr.concat([weighted_mean_SH,weighted_mean_NH], dim="latitude")
            del lat_SH,lat_NH,lon_SH,lon_NH,weighted_mean_SH,weighted_mean_NH
        elif split_hems==False:
            ds = get_superobs(files,var,var_dict,dataset=dataset,region='all')
            weights, ds_out[var_dict['out_name']] = weighted_mean_func(ds,var_dict)
            ds_out['latitude'], ds_out['longitude'] = ds.latitude,ds.longitude
    if split_hems==True:
        weights = np.concatenate([weights_SH,weights_NH],axis=1)
    return(ds_out,weights)


    
def get_superobs(files,var,var_dict,dataset='new_qa',region='all'):
    """
    Load superobservations for separate orbits for given month and put in 
    shared xarray Dataset.
    
    Parameters
    ----------
    date : str
        Give year and month in yyyymm to retrieve superobservations from.

    Returns
    -------
    ds : xarray Dataset
        Dataset with all superobservations for given month.
    
    """

        
    ##Create empty Dataset
    #Get array size
    data = xr.open_dataset(files[-1])            

    #Create empty Dataset to fill
    ds = xr.Dataset(data_vars = {})
    if region=='all':
        lat_dim = int(data.sizes['latitude'])
    elif region in ['SH','NH']:
        lat_dim = int(data.sizes['latitude']/2)
    
    ds[var_dict['out_name']] = xr.DataArray(data = np.full((len(files),lat_dim,data.sizes['longitude']),
                                                           np.nan).astype('float32'),
                                            dims = ['time','latitude','longitude'])
    ds['re_rel'] = xr.DataArray(data = np.full((len(files),lat_dim,data.sizes['longitude']),
                                                            np.nan).astype('float32'),
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
            ds[var_dict['out_name']].values[c1,:,:][valid] = data[var].values[valid]*var_dict['conversion']
            ds['re_rel'].values[c1,:,:][valid] = data['no2_superobs_re_rel'].values[valid]
            c1=c1+1
    #Add longitude and latitude
    ds['latitude'] = xr.DataArray(data = data.latitude,
                                     dims = ['latitude']
                                     )
    ds['longitude'] = xr.DataArray(data = data.longitude,
                                     dims = ['longitude']
                                     )

    return ds

    
# def land_water(ds):    
#       TODO : if var=='snow_ice_flag' in get_mean_all_vars -> use this function to calc land_water_mask
#     land_water_mask = np.full((ds.sizes['time'],ds.sizes['latitude'],ds.sizes['longitude']),np.nan)
#     snow_ice = np.round(ds.land_water_mask.values,0)
#     land_water_mask[((snow_ice>=1)&(snow_ice<=100))|(snow_ice==255)] = 0 #water
#     land_water_mask[(snow_ice==0)|(snow_ice==101)|(snow_ice==103)] = 1 #land
#     land_water_mask[(snow_ice==252)] = 2 #land-water-transition
#     ds['land_water_mask'] = xr.DataArray(data = land_water_mask, dims = ['time','latitude','longitude'])
#     return ds


def weighted_mean_func(ds,var_dict):
    """
    Calculate weighted mean.
    
    Parameters
    ----------
    ds : xr Dataset
        xarray Dataset with superobservation
        orbits.

    Returns
    -------
    weights : float32
        weights used for calculating weighted
        mean.
    weighted_mean : float32
        weighted mean.
        
    """
    weights = (1-ds['re_rel'].values).astype('float32')
    weighted_mean = np.full((weights.shape[1],weights.shape[2]),np.nan)
    zero_weight = (np.nansum(weights,axis=0)==0)
    weighted_mean[~zero_weight] = np.nansum(weights*ds[var_dict['out_name']].values,axis=0)[~zero_weight]/np.nansum(weights,axis=0)[~zero_weight]
    
    weighted_mean = xr.DataArray(data = weighted_mean, dims = ['latitude','longitude'])
    
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
        ds_SH['std1'] = standev1(ds_SH,weights[:,:450,:])
        ds_SH = ds_SH.drop_vars(["no2","no_superobs","weighted_mean"])
        ds_SH['std2'] = standev2(ds_SH,weights[:,:450,:],corr_coef_uncer)
        ds_SH = ds_SH.drop_vars(["sigma_amf","sigma_strat","sigma_sc","sigma_re"])
        
        ds_NH = get_uncertainty_superobs(files,uncertainty_vars,region='NH')
        ds_NH['weighted_mean'] = ds.sel(latitude=slice(0,90))['no2']
        ds_NH['std1'] = standev1(ds_NH,weights[:,450:,:])
        ds_NH = ds_NH.drop_vars(["no2","no_superobs","weighted_mean"])
        ds_NH['std2'] = standev2(ds_NH,weights[:,450:,:],corr_coef_uncer)
        ds_NH = ds_NH.drop_vars(["sigma_amf","sigma_strat","sigma_sc","sigma_re"])
        
        ds_uncer = xr.concat([ds_SH,ds_NH], dim="latitude")
    elif split_hems==False:
        ds_uncer = get_uncertainty_superobs(files,uncertainty_vars,region='all')
        ds_uncer['weighted_mean'] = ds.no2
        ds_uncer['std1'] = standev1(ds_uncer,weights)
        ds_uncer = ds_uncer.drop_vars(["no2","no_superobs","weighted_mean"])
        ds_uncer['std2'] = standev2(ds_uncer,weights,corr_coef_uncer)
        ds_uncer = ds_uncer.drop_vars(["sigma_amf","sigma_strat","sigma_sc","sigma_re"])
        
    ds['std1'] = ds_uncer['std1']
    ds['std2'] = ds_uncer['std2']
    return ds




def get_uncertainty_superobs(files,uncertainty_vars,region='all',dataset='new_qa'):
    """
    Load superobservations for separate orbits for given month and put in 
    shared xarray Dataset.
    
    Parameters
    ----------
    date : str
        Give year and month in yyyymm to retrieve superobservations from.

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


def standev1(ds,weights):
    """
    Calculate temporal uncertainty.
    
    Parameters
    ----------
    ds : xr Dataset
        xarray Dataset with superobservation
        orbits.
    weights : float32
        weights used for averaging.
    
    Returns
    -------
    std1 : float32
        temporal uncertainty.
    """
    std1 = np.sqrt(( np.nansum( weights*(ds['no2'].values-ds['weighted_mean'].values)**2,axis=0 ) )/
                   ( (ds['no_superobs'].values-1)/ds['no_superobs'].values * 
                    np.nansum(weights,axis=0)) )/np.sqrt(ds['no_superobs'].values)
    return xr.DataArray(data = std1.astype('float32'), dims = ['latitude','longitude'])


def standev2(ds,weights,corr_coef_uncer):
    """
    Calculate propoagated measurement uncertainty.
    
    Parameters
    ----------
    ds : xr Dataset
        xarray Dataset with superobservation
        orbits.
    weights : array, float32
        weights used for averaging.
    c_amf : float
        correlation factor air mass factor in temporal uncertainty propagation,
        the default is 0.3.
    c_sc : float
        correlation factor slant column in temporal uncertainty propagation,
        the default is 0.
    c_amf : float
        correlation factor stratospheric in temporal uncertainty propagation,
        the default is 0.3.
    c_amf : float
        correlation factor representativity in temporal uncertainty propagation,
        the default is 0.
    
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
    return xr.DataArray(data = std2, dims = ['latitude','longitude'])



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
        print(var)
        var_dict = calc_vars[var]
        ds[var_dict['out_name']] = xr.DataArray(data=eval(var_dict['func']),
                                                dims = ['latitude','longitude'])
        
    # ds['lat_bounds'] = 1#..
    # ds['lon_bounds'] = 1#...
    # ds['effective_day'] = 1 #...
    # ds['effective_time_of_day'] = 1 #....
    return ds
    


def get_attrs(date,ds_out):
    """
    Get attributes needed for output dataset

    Parameters
    ----------
    date : str
        Date used to retrieve L3 data for.

    Returns
    -------
    attrs : dict
        Dictionary filled with information for the attributes for the L3 dataset.

    """
    #Get geospatial resolution
    geospatial_lat_res = ds_out.latitude.values[5]-ds_out.latitude.values[4]
    geospatial_lon_res = ds_out.longitude.values[5]-ds_out.longitude.values[4]

    #Get array of available dates in month
    year = int(date[:4])
    month = int(date[4:6])
    last_day = str(calendar.monthrange(year, month)[1]+1)
    dates = np.arange(int(date+'01'),int(date+last_day),1).astype(str)
    
    #Get list of superobservation orbit files
    files = []
    for date in dates:
        # in_folder = '/nobackup/users/glissena/data/TROPOMI/L2/orbits/'+date[0:4]+'_'+date[4:6]+'/'+date
        in_folder = f'/net/pc200252/nobackup_1/users/gomenrt/no2_tropomi/PAL_reduced/{date[:6]}/'
        files.extend(sorted(os.listdir(in_folder)))

    time_coverage_list = " ".join([file[20:35] for file in files])
    datetime_start = datetime.datetime.strptime(files[0][20:35],'%Y%m%dT%H%M%S') - datetime.datetime(1995,1,1)
    datetime_start = datetime_start.days + datetime_start.seconds/60/60/24
    datetime_stop = datetime.datetime.strptime(files[-1][36:51],'%Y%m%dT%H%M%S') - datetime.datetime(1995,1,1)
    datetime_stop = datetime_stop.days + datetime_stop.seconds/60/60/24
    date_created = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")+'Z'
    attrs = {'files': files,
             'datetime_start': datetime_start,
             'datetime_stop': datetime_stop,
             'time_coverage_list': time_coverage_list,
             'date_created': date_created,
             'geospatial_lat_resolution': geospatial_lat_res,
             'geospatial_lon_resolution': geospatial_lon_res}
    return attrs


def add_nontime_vars(ds,files,var,var_dict):
    f = os.path.join(files[-1])
    data = xr.open_dataset(f)       
    ds[var_dict['out_name']] = xr.DataArray(data = data[var].values*var_dict['conversion'],
                                            dims = ['layer','independent_2'],
                                            attrs = var_dict['attrs']
                                            )
    return ds


def output_dataset(ds,attrs,variables_2d,variables_1d,corr_coef_uncer,files):
    """
    Create output dataset.
    
    Parameters
    ----------
    ds : xr Dataset
        xarray Dataset with superobservation
        orbits.
    attrs : dictionary
    
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
                                           attrs = {'description':'temporal uncertainty',
                                                    'long_name':'STD1 - temporal uncertainty',
                                                    'units':'molec/cm^2',
                                                    }
                              ),
        'tropospheric_NO2_column_number_density_uncertainty_kernel' : 
                              xr.DataArray(data = np.expand_dims(ds.std2.values,axis=0),
                                           dims = ['time','latitude','longitude'],
                                           attrs = {'description':'superobs uncertainty',
                                                    'long_name':'STD2 - superobs uncertainty',
                                                    'units':'molec/cm^2',
                                                    }
                              ),                             

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
        'layer' : xr.DataArray(data = np.arange(1,35,1).astype(int),
                                dims = ['layer'],
                                attrs = {'units':'1',
                                         'long_name':'layer dimension index'
                                         }
                                   ),
                                    },
        attrs = {'Conventions':' ',  #<----- CF-1.8 HARP-1.0 in HCHO product
                 'title':'NetCDF CF file providing L3 total nitrogendioxide satellite observations',
                 'institution':'KNMI',
                 'source':' ',
                 'project':'CCI+ ecv',
                 'summary':'TROPOMI L3 tropospheric nitrogendioxide columns for one individual sensor generated as part of the CCI + percursors',
                 'license':'',
                 'references':'',
                 'naming_authority':'KNMI',  #<--- or BIRA-IASB?
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
                 'key_variables':'tropospheric_NO2_column_number_density',   #<---- HCHO has second key variable
                 'gridding_software':'super-observations (Rijdsijk et al., in prep)',  #<--- edit
                 'gridding_selection_cloud_fraction_min':'',
                 'gridding_selection_cloud_fraction_max':'',
                 'grdding_selection_surface_albedo_max':'',
                 'gridding_selection_sza_max':'',
                 'gridding_selection_minimum_qa':0.75,
                 'temporal_uncertainty_correlation_scd':corr_coef_uncer['c_scd'],
                 'temporal_uncertainty_correlation_strat':corr_coef_uncer['c_strat'],
                 'temporal_uncertainty_correlation_amf':corr_coef_uncer['c_amf'],
                 'temporal_uncertainty_correlation_re':corr_coef_uncer['c_re'],
                 'comment':'',
                 'L1_version':'1.0',
                 'L2_version':'2.3.1',
                 'tracking_id':'', #<----- ???
                 'id':'',
                 'product_version':'',
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
                ds2[var_dict['out_name']] = xr.DataArray(data = np.expand_dims(ds[var_dict['out_name']].values,axis=0),
                                                         dims = ['time','latitude','longitude'],
                                                         attrs = var_dict['attrs']
                                                         )
    #Add variables from variables_1d 
    for var in variables_1d:
        var_dict = variables_1d[var]
        ds2 = add_nontime_vars(ds2,files,var,var_dict)
    

    #Add lat_bnds and lon_bnds
    lat_bnds_1 = [-90]
    for i in range(len(ds2.latitude.values)-1):
        lat_bnds_1.append(ds2.latitude.values[i]*2-lat_bnds_1[i])
    lat_bnds_2 = lat_bnds_1[1:]
    lat_bnds_2.append(90)
    ds2['latitude_bounds'] = xr.DataArray(data = np.array([lat_bnds_1,lat_bnds_2]).transpose(),
                                          dims = ['latitude','independent_2'],
                                          attrs = {'long_name' : 'grid latitude bounds',
                                                   'units' : 'degree north'}
                                          )
    #TODO This is geos-chem specific!
    lon_bnds_1 = [178.75,-178.75]
    for i in range(1,len(ds2.longitude.values)-1):
        lon_bnds_1.append(ds2.longitude.values[i]*2-lon_bnds_1[i])
    lon_bnds_2 = lon_bnds_1[1:]
    lon_bnds_2.append(178.75)
    ds2['longitude_bounds'] = xr.DataArray(data = np.array([lon_bnds_1,lon_bnds_2]).transpose(),
                                          dims = ['longitude','independent_2'],
                                          attrs = {'long_name' : 'grid longitude bounds',
                                                   'units' : 'degree east'}
                                          )
        
    return ds2
    




    
    