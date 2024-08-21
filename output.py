#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:02:41 2023

@author: Isolde Glissenaar

Functions to create output dataset.
"""

import numpy as np
import xarray as xr
import os
import datetime
import var_funcs as vf



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
    
    #Get list of superobservation orbit files
    in_folder = f'{main_sets["path_L2"]}/{date}/'
    if main_sets['L2_version']=='2.4':
        in_folder = f'{main_sets["path_L2"]}/{date[:4]}/{date[4:]}/'
    files = [ f for f in os.listdir(in_folder) if os.path.isfile(os.path.join(in_folder,f)) ]
    def sort(f):
        return f[20:]
    files = sorted(files,key=sort)
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
             'L2_version': main_sets['L2_version'],
             'L3_out_version': main_sets['L3_out_version']
             }
    return attrs




def output_dataset(ds,attrs,variables_2d,variables_1d,corr_coef_uncer,files,out_filename,date):
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
    out_filesname: str
        Filename of output dataset.
    date: str
        Date used to retrieve L3 data for.
    
    Returns
    -------
    ds2 : float32
        output dataset for saving.
    """

    #Create dataset with dimensions and attributes
    ds2 = xr.Dataset(data_vars = {
                'time' : xr.DataArray(data = ds.time.values,
                                      dims = ['time'],
                                      attrs = {'description':'start date of monthly mean',
                                               'long_name':'number of days since 1995-01-01',
                                               'standard_name':'time',
                                               'units':'days since 1995-01-01 00:00:00 0:00'
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
                 'L2_version':attrs['L2_version'],
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
        
    #Add variables from variables_2d & calc_vars 
    for lis in variables_2d:
        variables = variables_2d[lis]
        for var in variables:
            var_dict = variables[var]
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
        ds2 = vf.add_nontime_vars(ds2,files,var,var_dict)
        
    #Add land water mask
    ds2 = vf.add_land_water_mask(ds2, attrs)
    
    #Add lat and lon bounds
    ds2 = vf.add_latlon_bnds(ds2)

        
    return ds2
    