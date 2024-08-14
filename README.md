<!-- title -->
# L3 Temporal mean software 

<!-- subtitle -->
A software package that takes superobservations of satellite measurements and outputs the temporal mean
L3 observations. 

<!-- description -->

This is the documentation for the NO2 L3-software developed by Isolde Glissenaar (KNMI). 

### FOREWORD

This software package creates the temporal mean of multiple satellite orbits of observations of atmospheric gases. The code was designed to work with NO2 measurements and to create monthly means for the instruments TROPOMI and OMI, but can be adapted to other gases, time periods, or instruments. This software package takes as input superobservations from the (not yet publicly available) algorithm based upon the research paper **Quantifying uncertainties of satellite NO2 superobservations for data assimilation and model evaluation** (Rijsdijk et al., in review). The end result is a regularly gridded monthly mean Level-3 dataset. The distinguishing feature of this L3 software is the realistic uncertainty estimate. This software is developed as part of the ESA CCI+ Precursors for aerosols and ozone project, with the main goal being to develop long-term climate data records of the GCOS Precursors for Aerosol and Ozone Essential Climate Variable. 


<!-- TOC -->
<!--lint disable awesome-toc-->
## Contents
<!--lint enable awesome-toc-->

- [Installation](#installation)
- [How to Use](#HowToUse)


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

Change the settings in config.py to the correct dataset. In order to run the temporal mean script 
for one month, type in your command prompt/terminal:

~~~
python main.py 'yyyymm'
~~~

where yyyymm is for example 202006 for June 2020.



<!-- END CONTENT -->
