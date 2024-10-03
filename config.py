#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:02:41 2023

@author: Isolde Glissenaar

Configuration settings for L3 temporal mean
code. 
"""

from dataclasses import dataclass

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
    
    date = '202105' 

    main_sets = {'dataset':'1x1',
                 'split_hems':False,
                 'L2_version':'2.3.1',
                 'L3_out_version':'0122',
                 'path_in':"/nobackup/users/glissena/data/TROPOMI/L2/superobs/",
                 'path_L2':"/net/pc200252/nobackup_1/users/gomenrt/no2_tropomi/PAL_reduced/",  ##v2.3.1
                 #'path_L2':"/net/pc230013/nobackup_1/users/gomenrt/no2v2_reduced",  ##v2.4+
                 'path_out':"/nobackup/users/glissena/data/TROPOMI/out_L3/",
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
                                            'out_name' : 'tropospheric_NO2_column_number_density'},
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
                                            'out_name' : 'tropospheric_NO2_column_number_density',
                                            'dimension' : '2d',
                                            'attrs' : {'description':'NO2 tropospheric vertical column number density',
                                                       'long_name':'NO2 VCD',
                                                       'standard_name':'troposphere_mole_content_of_nitrogendioxide',
                                                       'units':'molec/cm^2',
                                                     }    
                                            },
        # 'surface_pressure' :              {'conversion' : 1e-2,   #Pa to hPa
        #                                     'out_name' : 'surface_pressure',
        #                                     'dimension' : '2d',
        #                                     'attrs' : {'description':'surface pressure',
        #                                                 'long_name':'surface pressure',
        #                                                 'units':'hPa'}
        #                                     },
        # 'surface_albedo' :                {'conversion' : 1,
        #                                     'out_name' : 'surface_albedo',
        #                                     'dimension' : '2d',
        #                                     'attrs' : {'description':'surface LER (440 nm)',
        #                                                 'long_name':'surface LER (440nm)',
        #                                                 'units':'1'}
        #                                     },
        # 'scd' :                           {'conversion' : 6.02214e19,
        #                                     'out_name' : 'NO2_slant_column_number_density',
        #                                     'dimension' : '2d',
        #                                     'attrs' : {'description' : 'NO2 slant column number density',
        #                                                 'long_name' : 'NO2 SCD',
        #                                                 'units' : 'molec/cm^2'}
        #                                     },
        # 'amf_trop_superobs' :             {'conversion' : 1,
        #                                     'out_name' : 'tropospheric_NO2_column_number_density_amf',
        #                                     'dimension' : '2d',
        #                                     'attrs' : {'description' : 'tropospheric air mass factor',
        #                                                 'long_name' : 'NO2 tropospheric AMF (440nm)',
        #                                                 'units' : '1'}
        #                                     },
        # 'amf_total_superobs' :             {'conversion' : 1,
        #                                     'out_name' : 'total_NO2_column_number_density_amf',
        #                                     'dimension' : '2d',
        #                                     'attrs' : {'description' : 'total air mass factor',
        #                                                 'long_name' : 'NO2 total AMF (440nm)',
        #                                                 'units' : '1'}
        #                                     },
        # 'strat_column' :                  {'conversion' : 6.02214e19,
        #                                     'out_name' : 'stratospheric_NO2_column_number_density',
        #                                     'dimension' : '2d',
        #                                     'attrs' : {'description' : 'Stratospheric NO2 vertical column number density',
        #                                                 'long_name' : 'NO2 stratospheric VCD',
        #                                                 'units' : 'molec/cm^2'}
        #                                     },
        'cloud_radiance_fraction' :       {'conversion' : 1,
                                            'out_name' : 'cloud_fraction',
                                            'dimension' : '2d',
                                            'attrs' : {'description' : 'effective cloud fraction',
                                                        'long_name' : 'cloud fraction',
                                                        'units' : '1'},
                                                },
        # 'cloud_pressure' :                {'conversion' : 1e-2,
        #                                    'out_name' : 'cloud_pressure',
        #                                    'dimension' : '2d',
        #                                    'attrs' : {'description' : 'cloud pressure at optical centroid',
        #                                               'long_name' : 'cloud_pressure',
        #                                               'units' : 'hPa'}
        #                                    },
        # 'kernel_full' :                   {'conversion' : 1,
        #                                     'out_name' : 'NO2_averaging_kernel',
        #                                     'dimension' : '3d',
        #                                     'attrs': {'description' : 'Column averaging kernel',
        #                                               'long_name' : 'full averaging kernel',
        #                                               'units' : '1'},
        #                                     },
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
    
    
        
    #Additional variables that will be calculated
    calc_vars = {
        'NO2_slant_column_number_density_troposphere' : {'func' : 'ds.tropospheric_NO2_column_number_density.values*ds.tropospheric_NO2_column_number_density_amf.values',
                                                         'out_name' : 'NO2_slant_column_number_density_troposphere',
                                                         'do_func' : True,
                                                         'dimension' : '2d',
                                                         'attrs' : {'description' : 'Tropospheric NO2 slant column number density',
                                                                    'long_name' : 'NO2 trop SCD',
                                                                    'units' : 'molec/cm^2'}
                                                            },
        'tropospheric_NO2_column_number_density_count' : {'out_name' : 'tropospheric_NO2_column_number_density_count',
                                                          'do_func' : False,
                                                          'dimension' : '2d',
                                                          'attrs' : {'description' : 'Effective number of observations per cell/ fractional coverage (coverage divided by number of days)',
                                                                     'units' : '1'}
                                                        },
        'qa_L3' :                                      {'func' : '(ds.tropospheric_NO2_column_number_density_count>=0.1)&(~np.isnan(ds.tropospheric_NO2_column_number_density.values))',
                                                        'out_name' : 'qa_L3',
                                                        'do_func' : True,
                                                        'dimension' : '2d',
                                                        'attrs' : {'description' : 'Gridded data quality assurance value (0: not valid, 1: valid)',
                                                                   'long_name' : 'data quality assurance value',
                                                                   'units' : '1'}
                                                        }, 
        'no_observations' :                              {'out_name' : 'no_observations',
                                                          'do_func' : False,
                                                          'dimension' : '2d',
                                                          'attrs' : {'description' : 'Total number of superobservations used to create temporal mean',
                                                                     'long_name' : 'number of superobservations',
                                                                     'units' : '1'}
                                                          },
        'tropospheric_NO2_column_number_density_temporal_std' : {'out_name' : 'tropospheric_NO2_column_number_density_temporal_std',
                                                                 'do_func' : False,
                                                                 'dimension' : '2d',
                                                                 'attrs' : {'description':'Temporal standard deviation in the NO2 tropospheric vertical column',
                                                                            'long_name':'temporal standard deviation',
                                                                            'units':'molec/cm^2',
                                                                            }
                                                                 },
        # 'tropospheric_NO2_column_number_density_measurement_uncertainty_kernel' : {
        #                                                          'out_name' : 'tropospheric_NO2_column_number_density_measurement_uncertainty_kernel',
        #                                                          'do_func' : False,
        #                                                          'dimension' : '2d',
        #                                                          'attrs' : {'description':'Uncertainty on the NO2 tropospheric vertical column'+
        #                                                                                   ' number density associated with area-averaged propagated'+
        #                                                                                   ' uncertainty of L2 input data, without the profile'+
        #                                                                                   ' uncertainty contribution (sigma_3)',
        #                                                                     'long_name':'NO2 VCD uncertainty kernel',
        #                                                                     'units':'molec/cm^2',
        #                                                                     }
        #                                                          },
        # 'tropospheric_NO2_column_number_density_measurement_uncertainty' : {
        #                                                          'out_name' : 'tropospheric_NO2_column_number_density_measurement_uncertainty',
        #                                                          'do_func' : False,
        #                                                          'dimension' : '2d',
        #                                                          'attrs' : {'description':'Uncertainty on the NO2 tropospheric vertical column'+
        #                                                                                   ' number density associated with area-averaged propagated'+
        #                                                                                   ' uncertainty of L2 input data (sigma_2)',
        #                                                                     'long_name':'NO2 VCD uncertainty',
        #                                                                     'units':'molec/cm^2',
        #                                                                     }
        #                                                          },
        'tropospheric_NO2_column_number_density_total_uncertainty_kernel' : {
                                                                 'out_name' : 'tropospheric_NO2_column_number_density_total_uncertainty_kernel',
                                                                 'do_func' : False,
                                                                 'dimension' : '2d',
                                                                 'attrs' : {'description':'Total uncertainty on the NO2 tropospheric vertical column'+
                                                                                        ' number density associated with time-averaged propagated'+
                                                                                        ' uncertainty of L2 input data and temporal representativity, '+
                                                                                        'without the profile uncertainty contribution',
                                                                            'long_name':'NO2 VCD total uncertainty kernel',
                                                                            'units':'molec/cm^2',
                                                                            }
                                                                 },
        'tropospheric_NO2_column_number_density_total_uncertainty' : {
                                                                 'out_name' : 'tropospheric_NO2_column_number_density_total_uncertainty',
                                                                 'do_func' : False,
                                                                 'dimension' : '2d',
                                                                 'attrs' : {'description':'Total uncertainty on the NO2 tropospheric vertical column'+
                                                                                          ' number density associated with time-averaged propagated'+
                                                                                          ' uncertainty of L2 input data and temporal representativity',
                                                                            'long_name':'NO2 VCD total uncertainty',
                                                                            'units':'molec/cm^2',
                                                                            }
                                                                 },
        'NO2_slant_column_number_density_uncertainty' : {'out_name' : 'NO2_slant_column_number_density_uncertainty',
                                                          'do_func' : False,
                                                          'dimension' : '2d',
                                                          'attrs' : {'description':'NO2 slant column number density uncertainty',
                                                                    'long_name':'NO2 SCDE',
                                                                    'units':'molec/cm^2',
                                                                    }
                                                          },
        'eff_date' :     {'out_name' : 'eff_date',
                          'do_func' : False,
                          'dimension' : '2d',
                          'attrs' : {'long_name':'effective date',
                                     'description':'effective date of observation',
                                     'standard_name':'effective date',
                                     'units':f'days since {date[0:4]}-{date[4:6]}-01 00:00:00',
                                     }
                          },
        'eff_frac_day' : {'out_name' : 'eff_frac_day',
                          'do_func' : False,
                          'dimension' : '2d',
                          'attrs' : {'description':'effective fractional day in local solar time. '+
                                                   'UTC = local_solar_time - longitude/180',
                                     'standard_name':'effective fractional day',
                                     'units':'1'
                                     }
                          },
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
                }
                                

    
    @dataclass 
    class Settings:
        date: str
        main_sets: dict
        variables_2d: dict
        variables_1d: dict
        uncertainty_vars: dict
        calc_vars: dict
        corr_coef_uncer: dict

    
    return Settings(date = date, 
                    main_sets = main_sets, 
                    variables_2d = variables_2d, 
                    variables_1d = variables_1d, 
                    uncertainty_vars = uncertainty_vars, 
                    calc_vars = calc_vars, 
                    corr_coef_uncer = corr_coef_uncer
                    )
    




