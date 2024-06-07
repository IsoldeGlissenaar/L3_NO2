#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:02:41 2023

@author: Isolde Glissenaar

Determine monthly mean tropospheric NO2 column
from superobservations (from the spatial mean 
superobservation code by Pieter Rijsdijk) for
TROPOMI (2018-05-01 - 2021-12-31). 
"""

import sys
import warnings
from monthly_mean_funcs import (
    get_attrs,
    output_dataset,
    get_list_of_files,
    get_mean_all_vars, 
    get_uncertainty,
    add_vars,
    add_count,
    add_time
    )

warnings.filterwarnings("ignore")

def settings():
    '''
    Create lists of variables to read from 
    superobservation files.

    Returns
    -------
    date : str
        Date yyyymm to run.
    variables_2d : dict
        List of variables with dimensions lat,lon to read.
    variables_1d : dict
        List of variables with dimensions layer,vertices to read.
    uncertainty_vars : dict
        List of variables needed for uncertainty calculation.

    '''
    
    date = '201901' 

    main_sets = {'dataset':'02x02',
                 'split_hems':True,
                 'path_in':"/nobackup/users/glissena/data/TROPOMI/L2/superobs/",
                 'L3_out_version':'0120',
                 }
        
    #Correlation coefficients for uncertainty calculation
    corr_coef_uncer = {
        'c_scd' : 0,
        'c_strat' : 0.3,
        'c_amf' : 0.3,
        'c_re' : 0
                       }
    
    #List of uncertainty variables to read 
    uncertainty_vars = {
        'no2_superobs' :                  {'conversion' : 6.02214e19,
                                            'out_name' : 'no2'},
        'no2_superobs_sig_amf' :          {'conversion' : 6.02214e19,
                                            'out_name' : 'sigma_amf'},
        'no2_superobs_sig_slant_random' : {'conversion' : 6.02214e19,
                                            'out_name' : 'sigma_sc'},
        'no2_superobs_sig_stratosphere' : {'conversion' : 6.02214e19,
                                            'out_name' : 'sigma_strat'},
        'no2_superobs_sig_re' :           {'conversion' : 6.02214e19,
                                            'out_name' : 'sigma_re'},
                        }
                    
    #List of 2d variables to read
    variables_2d = {
        'no2_superobs' :                  {'conversion' : 6.02214e19,  #Mole/m2 to molecules/cm2
                                            'out_name' : 'no2',
                                            'get_mean' : False,
                                            'dimension' : '2d'},
        'surface_pressure' :              {'conversion' : 1e-2,   #Pa to hPa
                                            'out_name' : 'surface_pressure',
                                            'get_mean' : True,
                                            'dimension' : '2d',
                                            'attrs' : {'description':'surface pressure',
                                                        'long_name':'surface pressure',
                                                        'units':'hPa'}
                                            },
        'surface_albedo' :                {'conversion' : 1,
                                            'out_name' : 'surface_albedo',
                                            'get_mean' : True,
                                            'dimension' : '2d',
                                            'attrs' : {'description':'surface LER (440 nm)',
                                                        'long_name':'surface LER (440nm)',
                                                        'units':'1'}
                                            },
        'covered_area_fraction' :         {'conversion' : 1,
                                            'out_name' : 'covered_area_fraction',
                                            'get_mean' : False,
                                            'dimension' : '2d',},
        'scd' :                           {'conversion' : 6.02214e19,
                                            'out_name' : 'NO2_slant_column_number_density',
                                            'get_mean' : True,
                                            'dimension' : '2d',
                                            'attrs' : {'description' : 'NO2 slant column number density',
                                                        'long_name' : 'NO2 SCD',
                                                        'units' : 'molec/cm^2'}
                                            },
        'amf_trop_superobs' :             {'conversion' : 1,
                                            'out_name' : 'tropospheric_NO2_column_number_density_amf',
                                            'get_mean' : True,
                                            'dimension' : '2d',
                                            'attrs' : {'description' : 'tropospheric air mass factor',
                                                        'long_name' : 'NO2 tropospheric AMF (440nm)',
                                                        'units' : '1'}
                                            },
        'amf_total_superobs' :             {'conversion' : 1,
                                            'out_name' : 'total_NO2_column_number_density_amf',
                                            'get_mean' : True,
                                            'dimension' : '2d',
                                            'attrs' : {'description' : 'total air mass factor',
                                                        'long_name' : 'NO2 total AMF (440nm)',
                                                        'units' : '1'}
                                            },
        'strat_column' :                  {'conversion' : 6.02214e19,
                                            'out_name' : 'NO2_stratospheric_column_number_density',
                                            'get_mean' : True,
                                            'dimension' : '2d',
                                            'attrs' : {'description' : 'Stratospheric NO2 vertical column number density',
                                                        'long_name' : 'NO2 stratospheric VCD',
                                                        'units' : 'molec/cm^2'}
                                            },
        'cloud_radiance_fraction' :       {'conversion' : 1,
                                            'out_name' : 'cloud_fraction',
                                            'get_mean' : True,
                                            'dimension' : '2d',
                                            'attrs' : {'description' : 'effective cloud fraction',
                                                        'long_name' : 'cloud fraction',
                                                        'units' : '1'},
                                                },
        'cloud_pressure' :                {'conversion' : 1e-2,
                                            'out_name' : 'cloud_pressure',
                                            'get_mean' : True,
                                            'dimension' : '2d',
                                            'attrs' : {'description' : 'cloud pressure at optical centroid',
                                                        'long_name' : 'cloud_pressure',
                                                        'units' : 'hPa'}
                                                },
        'kernel_full' :                   {'conversion' : 1,
                                            'out_name' : 'NO2_averaging_kernel',
                                            'get_mean' : True,
                                            'dimension' : '3d',
                                            'attrs': {'description' : 'Column averaging kernel',
                                                      'long_name' : 'full averaging kernel',
                                                      'units' : '1'},
                                            },
                    }
    
    #List of none time-dependent variables to read
    variables_1d = {
        'tm5_constant_a' : {'conversion' : 1e-2, #Pa to hPa
                            'out_name':'tm5_sigma_a',
                            'attrs' : {'description' : "tm5 sigma-values a, pressure = tm5_sigma_a +"+
                                                        "surface_pressure * tm5_sigma_b",
                                        'long_name' : 'tm5 sigma-values a',
                                        'units' : 'hPa'},
                            },
        'tm5_constant_b' : {'conversion' : 1, 
                            'out_name':'tm5_sigma_b',
                            'attrs' : {'description' : "tm5 sigma-values b, pressure = tm5_sigma_a + "+
                                                        "surface_pressure * tm5_sigma_b",
                                        'long_name' : 'tm5 sigma-values b',
                                        'units' : '1'},
                            },
                    }
    
    
        
    #Variables to calculate 2D
    calc_vars = {
        'NO2_slant_column_number_density_troposphere' : {'func' : 'ds.no2.values*ds.tropospheric_NO2_column_number_density_amf.values',
                                                          'out_name' : 'NO2_slant_column_number_density_troposphere',
                                                          'get_mean' : True,
                                                          'do_func' : True,
                                                          'dimension' : '2d',
                                                          'attrs' : {'description' : 'Tropospheric NO2 slant column number density',
                                                                    'long_name' : 'NO2 trop SCD',
                                                                    'units' : 'molec/cm^2'}
                                                            },
        'tropospheric_NO2_column_number_density_count' : {'out_name' : 'tropospheric_NO2_column_number_density_count',
                                                          'get_mean' : True,
                                                          'do_func' : False,
                                                          'dimension' : '2d',
                                                          'attrs' : {'description' : 'Effective number of observations per cell/ fractional coverage (coverage divided by number of days)',
                                                                      'units' : '1'}
                                                        },
        'qa_L3' :                                      {'func' : '(ds.tropospheric_NO2_column_number_density_count>=0.1)&(~np.isnan(ds.no2.values))',
                                                        'out_name' : 'qa_L3',
                                                        'get_mean' : True,
                                                        'do_func' : True,
                                                        'dimension' : '2d',
                                                        'attrs' : {'description' : 'Gridded data quality assurance value (0: not valid, 1: valid)',
                                                                    'long_name' : 'data quality assurance value',
                                                                    'units' : '1'}
                                                        }, 
        'no_observations' :                              {'out_name' : 'no_observations',
                                                          'get_mean' : True,
                                                          'do_func' : False,
                                                          'dimension' : '2d',
                                                          'attrs' : {'description' : 'Total number of superobservations used to create temporal mean',
                                                                      'long_name' : 'number of superobservations',
                                                                      'units' : '1'}
                                                          }
                 }
    
    return date, main_sets, variables_2d, variables_1d, uncertainty_vars, calc_vars, corr_coef_uncer
    


    



def main():
    #Get settings
    date, main_sets, variables_2d, variables_1d, uncertainty_vars, calc_vars, corr_coef_uncer = settings()
    date = sys.argv[1]

    #Get monthly mean
    files = get_list_of_files(date,main_sets)
    ds_out,weights = get_mean_all_vars(variables_2d,files,dataset=main_sets['dataset'],split_hems=main_sets['split_hems'])
    ds_out = get_uncertainty(ds_out,weights,files,uncertainty_vars,corr_coef_uncer,split_hems=main_sets['split_hems'])
    ds_out = add_count(ds_out,files,date)
    ds_out = add_vars(ds_out,calc_vars)
    ds_out = add_time(ds_out,files,date,weights,split_hems=main_sets['split_hems'])        
    del weights
        
    #Save to file
    out_filename = f'ESACCI-PREC-L3-NO2_TC-TROPOMI_S5P-KNMI-1M-{date}{files[0][-20:-18]}_{date}{files[-1][-20:-18]}-fv{main_sets["L3_out_version"]}.nc'
    attrs = get_attrs(date,ds_out,main_sets)
    ds2 = output_dataset(ds_out,attrs,{'variables_2d':variables_2d,'calc_vars':calc_vars},variables_1d,
                         corr_coef_uncer,files,out_filename)
    ds2.to_netcdf(f'/nobackup/users/glissena/data/TROPOMI/out_L3/{main_sets["dataset"]}/{out_filename}')
    del ds_out,ds2

if __name__ == "__main__": 
    main()
    
    
    
    