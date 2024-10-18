#!/bin/bash

for file in /nobackup/users/glissena/data/TROPOMI/out_L3/02x02/v1_22/*
do
  ncatted -a units,longitude_bounds,a,c,"degree_east" "$file"
  ncatted -a units,latitude_bounds,a,c,"degree_north" "$file"
done
