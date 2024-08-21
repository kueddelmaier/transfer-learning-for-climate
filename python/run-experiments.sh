#!/bin/bash


# Add path of python folder to PYTHONPATH -- adjust to your path
export PYTHONPATH=home/jkuettel/latent-linear-adjustment-autoencoders/python:$PYTHONPATH

# The following commands needs to be run from the python directory.

# PRECIPITATION

## From scratch

### Dynamical adjustment

#### train autoencoder from scratch
#python3.8 climate_ae/models/ae/main_ae.py --config '/home/jkuettel/latent-linear-adjustment-autoencoders/python/climate_ae/models/ae/configs/mixed_train_masked_to_eobs_same_detrend_no_end/config_dyn_adj_precip_main_ae.json' --num_epochs 10
#python3.8 climate_ae/models/ae/main_ae.py --config '/home/jkuettel/latent-linear-adjustment-autoencoders/python/climate_ae/models/ae/configs/co_train_individual_detrend/config_dyn_adj_precip_main_ae.json' 
#python3.8 climate_ae/models/ae/main_ae.py --config '/home/jkuettel/latent-linear-adjustment-autoencoders/python/climate_ae/models/ae/configs/co_train_individual_detrend/test_config_dyn_adj_precip_main_ae_1.json'
python3.8 climate_ae/models/ae/main_ae.py --config '/home/jkuettel/latent-linear-adjustment-autoencoders/python/climate_ae/models/ae/configs/co_train_individual_detrend/test_config_dyn_adj_precip_main_ae_eobs.json'

#python3.8 climate_ae/models/ae/main_ae.py --data_from_exp 1 --exp_id '8SqDYKCnJj'


#### re-train linear model and produce plots (pass CHECKPOINT_ID from previous step)
# python3.8 climate_ae/models/ae/main_linear.py --checkpoint_id='CHECKPOINT_ID' --precip=1

### Weather generator

#### re-train linear model and produce plots
# python3.8 climate_ae/models/ae/main_generator.py --checkpoint_id='CHECKPOINT_ID' --precip=1


## Using pre-trained models

### Dynamical adjustment

#### re-train linear model and produce plots, using provided model

#python3.8 climate_ae/models/ae/main_linear.py #--checkpoint_id 'nKGagmsKDb_4249785' --precip 1 # --save_nc_files=1 #--load_json=1


### Weather generator

#### re-train linear model and produce plots, using provided model
# python3.8 climate_ae/models/ae/main_generator.py --checkpoint_id='nKGagmsKDb_4249785' --precip=1
