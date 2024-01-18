<!-- title -->
# L3 Temporal mean software 

<!-- subtitle -->
A software package that takes superobservations of satellite measurements and outputs the temporal mean
L3 observations. 

<!-- description -->

This is the documentation for the NO2 L3-software developed by Isolde Glissenaar (KNMI). 

### FOREWORD

Satellite observations of tropospheric trace gases and aerosols are evolving rapidly. Recently launched instruments provide
increasingly higher spatial resolutions with footprint diameters in the range of 2-8 km, with daily global coverage for polar
satellites or hourly observations from geostationary orbit. Using such vast amounts of data in modern (global) data assimila-
tion systems in an optimal way is challenging. Often the modelling system has a lower spatial resolution than the satellites
used, with a model grid size in the range of 10-100 km. When the resolution mismatch is not properly accounted for, the final
analysis based on the satellite data may be degraded. Superobservations are averages of individual observations matching the
resolution of the model and are functional to reduce the data load on the assimilation system. 


This software package implements a superobservation algoritm based upon the research paper *Constructing superobservations from satellite NO2 for assimilation
and model evaluation* (2023, Rijsdijk et al.) The software was designed to work with NO2 measurements from TROPOMI but can be generalised to other instruments and trace gasses. The distinguishing feature of this superobservation software is the realistic uncertainty estimate.


<!-- TOC -->
<!--lint disable awesome-toc-->
## Contents
<!--lint enable awesome-toc-->

- [Installation](#installation)
- [How to Use](#HowToUse)
- [Command Line Interface](#command-line-interface)
- [Advanced Configuration Options](#advanced-configuration-options)


<!-- CONTENT -->

# Installation

<p>
The package uses (micro)mamba instead of (ana/micro)conda. Install micromamba with the command:

~~~ 
curl -L micro.mamba.pm/install.sh | bash
~~~ 

and allow the installer to modify your bashrc.

The Installer will put micromamba in your Home directory by default.
If an alternative location is needed, then run the following commands and follow the instructions.

~~~ 
curl -L micro.mamba.pm/install.sh > install.sh
bash install.sh --prefix {installlocation)
~~~ 

</p>
 
<br />

<p>
After a successful installation, create an environment using the environment.yml file provided.

~~~ 
micromamba env create -f environment.yml
~~~ 

(In case additional packages are installed, it's always good to update the environment file:

~~~ 
micromamba env export -n L3_NO2 > environment.yml)
~~~ 

Using the command line, enable the new environment with the command:

~~~ 
micromamba activate L3_NO2
~~~ 
</p>

# How to use

There are two ways to use the superobservation software. Either you configure the `config.py` file or use the command line interface.

## Configuration Options in `config.py`

The `config.py` file is used to configure various options for the superobservation software. Here's an overview of the available configuration options:

- **`data_folder`**: This option specifies the path to the data input folder or data file. It determines the source of the input data for superobservation processing. Users can provide the path to the folder containing multiple data files or a single data file.

- **`search`**: The `search` option is a regular expression that determines which files in the `data_folder` should be considered during processing. It allows users to filter and select specific data files based on a pattern. An example is

    **Example:**  '`*.nc`' or '`*TROPOMI*_03*.nc`'

- **`out_folder`**: This option specifies the path to the data output folder. It determines where the output superobservation files will be saved after processing.

- **`search_out`**: The `search_out` option is a regular expression-based search criteria used within the data source folder. It defines the pattern to identify existing superobservation files in the output folder. These existing files will be skipped during the processing to avoid overwriting.

## Running the software using `config.py`

Once the `config.py`-file is set-up, the software is executed by

```shell
python superobs.py
```

## Command Line Interface

The software package has a built-in command line interface bypassing the need to set-up the config-file. You can run the program from the command line with

```shell
python run_sobs_cmd.py [options]
```

### Command Line Options

The following command-line options are available:

- `-i`, `--input` : Input Folder or File
  - **Description:** Specifies the input data source. It can be a folder containing multiple data files (recursively searched) or a single input file.
  - **Example:** `-i input_data/` or `-i input_file.nc`

- `-o`, `--output` : Output Folder
  - **Description:** Specifies the folder where the output superobservation files will be saved.
  - **Example:** `-o output_folder/`

- `-s`, `--settings` : Settings File
  - **Description:** Specifies the settings file to use for superobservation processing. The default is 'settings_sob'.
  - **Example:** `-s my_settings`

- `-sr`, `--search` : Search Pattern
  - **Description:** Specifies the regex search pattern to match data files in the input folder when searching for input data. The default is '*.nc'.
  - **Example:** `-sr '*.txt'`

- `-or`, `--overwrite` : Overwrite Existing Files
  - **Description:** Determines whether to overwrite existing output files with the same name. If set to 'yes' or 'y', existing files will be overwritten.
  - **Example:** `-or yes`

### Example Usage

Here's an example of how to use superobservation-software from the command line with using specified options:

```shell
python superobservation.py -i 'input_data/' -o 'output_folder/' -s 'my_settings' -sr '*TROPOMI*_03_*.nc' -or 'yes'
```

In this example, the script is run with specific input data, output folder, custom settings file, a different search pattern, and permission to overwrite existing files.


## Advanced Configuration Options

By default, the software is set-up to run with TROPOMI NO2 data. It can however be set up to work with different 
instruments or trace gasses. This chapter of the documentation discusses the necessary advanced configuration options. 

### The configuration file `settings_sob.py`

The advanced configuration file `settings_sob.py` creates a dictionary containing all the configurations for the superobservation calculation. The settings can be subdivised in six categories, with an additional 'general' category for the items that do not fit neatly into any of the other categories.

- [General Settings](#general-settings)
- [Superobservation Grid Definition](#superobservation-grid-definition)
- [Instrument Data Structure](#instrument-data-structure)
- [Instruments Variables to Carry Over and Output](#instruments-variables-to-carry-over-and-output)
- [Instrument Grid](#instrument-grid)
- [Stratospheric Error](#stratospheric-error)
- [Superobservation Uncertainty Calculation](#superobservation-uncertainty-calculation)


#### General Settings

- **`use_orig_name`**:
  - Type: Boolean (True or False)
  - Description: If set to `True`, the application will use the name from the input file as the name for the output file. If set to `False`, it will use the names from the dictionaries.

- **`n_points`**:
  - Type: Integer
  - Description: Specifies the number of points to calculate overlap. 

- **`cover_min`**:
  - Type: Float
  - Description: Specifies a minimum coverage value (0.3 in this case).

- **`fill_value`**:
  - Type: Float
  - Description: Specifies a fill value. We use the default value 9.96921E36.

- **`min_footprint`**:
  - Type: Tuple of two Float values (x, y)
  - Description: Specifies the minimum footprint of the satellite, with `x` and `y` representing the dimensions.

- **`mask_dict`**:
  - Type: Dictionary
  - Description: Defines a quality assurance mask using the following parameters:
    - `mask_vars`: A list of variables to be masked (e.g., ['qa']).
    - `mask_expr`: The mask expression that determines which values to mask (e.g., '(qa > 0.75)').

- **`unit_dict`**:
  - Type: Dictionary
  - Description: Allows unit conversion, where results are multiplied by a factor and optionally given a new unit. Units can be converted by specifying the variable name as the key and providing conversion details as a nested dictionary.

- **`corr_len`**:
  - Type: Float
  - Description: Specifies the AMF (Air Mass Factor) correlation length in kilometers.

- **`power`**:
  - Type: Float
  - Description: Specifies the power for the AMF correlation function.



#### Superobservation Grid Definition

The superobservation-grid (output) is defined

- **`y_vec_centers`**:
  - Type: Array of floats
  - Description: 

- **`grid_dict`**:
  - Type: Dictionary
  - Description:
    - `x_type`:
    - `y_type`:
    - `x_params`:
    - `y_vec`:

#### Instrument Data Structure

- **`time_path`**:
  - Description: Specifies the path to the time variable in the netCDF file.

- **`lon_corn_path`**:
  - Description: Specifies the path to the longitude bounds variable in the netCDF file. These bounds represent the corners of the latitude and longitude grid cells.

- **`lat_corn_path`**:
  - Description: Specifies the path to the latitude bounds variable in the netCDF file. These bounds represent the corners of the latitude and longitude grid cells.

- **`corn_inds`**:
  - Description: Provides a dictionary of indices for the corners of satellite observations. The dictionary includes the following keys:
    - `lb`: Index for the lower-left corner.
    - `rb`: Index for the lower-right corner.
    - `rt`: Index for the upper-right corner.
    - `lt`: Index for the upper-left corner.

#### Instruments Variables to Carry Over and Output

- **`dim_list`**:
  - Description: Specifies additional dimensions to carry over from the input file. Standard dimensions are time, latitude, and longitude.

- **`atr_list`**:
  - Description: Lists netCDF attributes to carry over from the input file, such as 'time_coverage_start' and 'time_coverage_end'.

- **`in_dict_no_change`**:
  - Description: Defines input variables that are to be carried over from the input file but remain unused for superobservations. These include 'tm5_constant_a' and 'tm5_constant_b'.

- **`in_dict_xy`**:
  - Description: Specifies input variables with x and y components that will be area-weight averaged. These variables include 'qa', 'no2_superobs', 'trop_col_precis', 'amf_trop_superobs', 'scd', 'scd_precis', 'cloud_radiance_fraction', 'surface_pressure', 'sze', and 'vze'.

- **`in_dict_xyz`**:
  - Description: Lists input variables with x, y, and z components. For example, 'kernel_full' refers to 'PRODUCT/averaging_kernel'.

- **`calc_dict`**:
  - Description: Defines variables that need to be calculated based on other input variables. Each variable in the dictionary includes:
    - `calc`: The calculation formula.
    - `dimension`: The dimensions involved in the calculation.
    - `units`: The units of the variable.
    - `name_long`: A descriptive name for the variable. Examples include 'tropopsheric slant column precision' and 'NO2 tropospheric averaging kernel'.

- **`calc_var_dict`**:
  - Description: Lists variables used only in calculations but will not be written to the output file. Variables such as 'sze', 'vze', 'trop_layer', and 'amf_full' are included.


#### Instrument Grid

TODO: User input more complicated.

#### Stratospheric Error

TODO: User input more complicated.

#### Superobservation Uncertainty Calculation

TODO: User input more complicated.

## Questions and Notes for Pieter:

* note: refactored the cmd-interface so it is just a single script wether you input a folder or a single file.

* note: reshuffeld the settings_sob.py to make it more user-friendly. (I categorised the different user-settings)




<!-- END CONTENT -->