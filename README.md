<!-- title -->
# L3 Temporal mean software 

<!-- subtitle -->
A software package that takes superobservations of satellite measurements and outputs the temporal mean
L3 observations. 

<!-- description -->

This is the documentation for the NO2 L3-software developed by Isolde Glissenaar (KNMI). 

### FOREWORD

...


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

...



<!-- END CONTENT -->