#!/bin/bash


# Add path of python folder to PYTHONPATH -- adjust to your path
export PYTHONPATH=

# The following commands needs to be run from the python directory.

# PRECIPITATION



## Using pre-trained models

### Dynamical adjustment

#### re-train linear model and produce plots, using provided model

python3.8 climate_ae/models/ae/main_linear.py --lin_config='' 


### Weather generator

#### re-train linear model and produce plots, using provided model
# python3.8 climate_ae/models/ae/main_generator.py --checkpoint_id='nKGagmsKDb_4249785' --precip=1

