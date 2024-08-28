#!/bin/bash


# Add path of python folder to PYTHONPATH -- adjust to your path
export PYTHONPATH=

# The following commands needs to be run from the python directory.

# PRECIPITATION

## From scratch

### Dynamical adjustment

#### train autoencoder from scratch
#python3.8 climate_ae/models/ae/main_ae.py --config '/home/jkuettel/latent-linear-adjustment-autoencoders/python/climate_ae/models/ae/configs/config_dyn_adj_precip_main_ae.json' 

#### re-train linear model and produce plots (pass CHECKPOINT_ID from previous step)
# python3.8 climate_ae/models/ae/main_linear.py --checkpoint_id='CHECKPOINT_ID' --precip=1

### Weather generator

#### re-train linear model and produce plots
# python3.8 climate_ae/models/ae/main_generator.py --checkpoint_id='CHECKPOINT_ID' --precip=1


## Using pre-trained models

### Dynamical adjustment

#### re-train linear model and produce plots, using provided model

python3.8 climate_ae/models/ae/main_linear.py --lin_config='/home/jkuettel/latent-linear-adjustment-autoencoders/python/climate_ae/models/ae/configs/co_train_individual_detrend/test_config_dyn_adj_precip_linear_eobs.json' --precip 1 
python3.8 climate_ae/models/ae/main_linear.py --lin_config='' --precip 1 


### Weather generator

#### re-train linear model and produce plots, using provided model
# python3.8 climate_ae/models/ae/main_generator.py --checkpoint_id='nKGagmsKDb_4249785' --precip=1

