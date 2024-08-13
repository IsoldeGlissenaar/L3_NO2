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

import config
import temporal_mean as tm
import uncertainty as uncer
import var_funcs as vf
import output

warnings.filterwarnings("ignore")


def main():
    #Get settings
    settings = config.settings()
    settings.date = sys.argv[1]

    #Get monthly mean
    files = tm.get_list_of_files(settings.date, settings.main_sets)
    ds_out,weights = tm.get_mean_all_vars(settings.variables_2d, files, dataset=settings.main_sets['dataset'],
                                          split_hems=settings.main_sets['split_hems'])
    ds_out = uncer.get_uncertainty(ds_out, weights, files, settings.uncertainty_vars, settings.corr_coef_uncer,
                                   split_hems=settings.main_sets['split_hems'])
    ds_out = vf.add_count(ds_out, files, settings.date)
    ds_out = vf.add_vars(ds_out, settings.calc_vars)
    ds_out = vf.add_time(ds_out, files, settings.date, weights, split_hems=settings.main_sets['split_hems'])        
    del weights
        
    #Save to file
    out_filename = f"ESACCI-PREC-L3-NO2_TC-TROPOMI_S5P-KNMI-1M-{settings.date}{files[0][-20:-18]}_{settings.date}{files[-1][-20:-18]}-fv{settings.main_sets['L3_out_version']}.nc"
    attrs = output.get_attrs(settings.date, ds_out, settings.main_sets)
    ds2 = output.output_dataset(ds_out, attrs, {'variables_2d':settings.variables_2d,'calc_vars':settings.calc_vars},
                                settings.variables_1d, settings.corr_coef_uncer, files, out_filename, settings.date)
    ds2.to_netcdf(f"/nobackup/users/glissena/data/TROPOMI/out_L3/{settings.main_sets['dataset']}/{out_filename}")
    del ds_out,ds2


if __name__ == "__main__": 
    main()
    
    