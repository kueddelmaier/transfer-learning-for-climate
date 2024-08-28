# Latent Linear Adjustment autoencoders: A novel method for estimating and emulating dynamic precipitation at high resolution

This repository contains the code for the Latent Linear Autoencoder (LLAAE) model developed in the study
"Transfer learning for estimating circulation-induced precipitation variability: from climate models to observations"

In this study, we have shown how LLAAE's unique architecture enables effective transfer learning from models to observations in climate sciences.

This README is meant to be used in conjunction with the accompanying paper, rather than as a standalone guide. Below, we provide an overview of the model, along with detailed instructions on how to reproduce the results presented in the paper.


## Abstract

While deep learning and neural networks are gaining increased popularity in
climate science, the lack of observational training data presents a significant challenge. In this study,
we show that we can overcome this issue by capitalizing on the abundance of climate model data
available from regional climate models such as CORDEX. We test this transfer learning approach
in the context of dynamical adjustment on daily, high-resolution precipitation data over Europe
using Latent Linear Auto Encoders. These novel statistical models offer a powerful tool
for transfer learning applications in climate sciences. Firstly, they permit to adapt the model to the
new domain of interest by retraining the model’s linear components with only a few data samples.
Secondly, by encoding the data into a lower dimensional probabilistic space, transfer learning can
be achieved by finding meaningful structures, so-called latent features, between different domains
that facilitate knowledge transfer. We first validate this transfer learning approach by successfully
estimating dynamic precipitation and recovering trend estimates across structurally different climate
models, improving spatial trend estimates by a factor of three compared to raw trends. Finally, we
apply the Latent Linear Auto Encoder to reanalysis and observational data, showing that the predicted precipitation
patterns closely match the observations, demonstrating the method’s transferability between climate
models and real-world observations.


## Latent Linear Adjustment Autoencoder (LLAAE)

The Latent Linear Adjustment Autoencoder (LLAAE) builds upon the standard Variational Autoencoder (VAE) architecture with an additional linear component that enhances its robustness, making it particularly interesting for transfer learning applications. The model is designed to capture the relationship between large-scale atmospheric circulation and high-resolution precipitation fields, allowing for spatially coherent predictions in complex climate systems.


### Model Architecture ###

<img src="https://github.com/kueddelmaier/transfer-learning-for-climate/blob/main/documentation/LLAAE_illustration.png" width="800">

The LLAAE consists of two main components:

* **Nonlinear Variational Autoencoder (VAE)**: The spatial precipitation fields re encoded into a lower-dimensional probabilistic latent space by the encoder. The decoder then reconstructs the original precipitation fields from this latent representation.

* **Linear Model** The coarse-resolution sea level pressure field is used to predict the circulation-induced precipitation through the linear model. 


### How to Use the LLAAE ###

* **Training the model:** Encoder, decoder and the linear model are initially trained in an alternating manner on large climate model datasets.

* **Transfer to Observations:** When the LLAAE is applied to observational data, the model can be fine-tuned by retraining the linear model with only a few samples of real observations. This retraining allows the model to adapt to new domains, such as observational datasets that differ from the climate model used in the initial training phase.

* **Applications:** The LLAAE can be used for various climate science applications, such as estimating precipitation trends, performing statistical downscaling, and detecting circulation-induced precipitation variability. 



## Installing dependencies

You need Python 3.8. The dependencies are managed with [``poetry``](https://python-poetry.org/). To create a virtual environment and install them using poetry, run:

```
python -m virtualenv env
source env/bin/activate
pip install poetry
poetry install
```

## Data

We provide a sample data set which is available on Zenodo: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3949748.svg)](URL)



To correctly load the data, you need to copy the file [``settings.py``](https://github.com/kueddelmaier/transfer-learning-for-climate/blob/master/python/settings.py) and rename it to ``local_settings.py``. In ``local_settings.py``, specify (a) where the data is located in ``DATA_PATH``, and (b) where the output should be saved in ``OUT_PATH``. 


## Running experiments

The commands to run the experiments are detailed in ``python/run-experiments.sh``. Note that you need to add the path of the [``python``](https://github.com/kueddelmaier/transfer-learning-for-climate/blob/master/python) directory to your ``PYTHONPATH`` (see ``python/run-experiments.sh``). 

The first step consists of training the Latent Linear Adjustment autoencoder model. From the ``python`` directory run:

```
python3.7 climate_ae/models/ae/main_ae.py
```

By default, the hyperparameters from the file ``python/climate_ae/models/ae/configs/config_dyn_adj_precip.json`` will be used which correspond to the settings needed to reproduce  the precipitation results reported in the manuscript. 

Each trained model is associated with a so-called ``CHECKPOINT_ID`` which is needed to load a trained model. The ``CHECKPOINT_ID`` is returned as the last logging statement when training the autoencoder and it is also saved in the model outputs that are written to ``OUT_PATH``.

After training the autoencoder, the linear model can be refitted non-iteratively (keeping the autoencoder parameter fixed) and a number of evaluation plots are produced with the following command. The ``CHECKPOINT_ID`` from the trained autoencoder needs to be passed here, such that the correct model is loaded.

```
python3.7 climate_ae/models/ae/main_linear.py --checkpoint_id='CHECKPOINT_ID' --precip=1
```

Finally, the weather generator can be trained using the following command, again passing the ``CHECKPOINT_ID`` from the trained autoencoder:

```
python3.7 climate_ae/models/ae/main_generator.py --checkpoint_id='CHECKPOINT_ID' --precip=1
```

### Command-line arguments and further hyperparameters

#### ``main_ae.py``

The following experimental settings are controlled via command line arguments:
* ``config``: Specifies the path to the config file which contains further hyperparameter settings.
* ``penalty_weight``: Weight of the penalty in the loss function that enforces the linearity between circulation and the latent space of the autoencoder. 
* ``local_json_dir_name``: Directory name where metrics and configs are saved.
* ``dim_latent``: Dimension of the latent space of the autoencoder. 
* ``num_fc_layers``: Number of fully connected layers; only relevant when ``architecture`` is set to ``fc``.
* ``num_conv_layers``: Number of convolutional layers; only relevant when ``architecture`` is set to ``convolutional``.
* ``num_residual_layers``: Number of residual layers; only relevant when ``architecture`` is set to ``convolutional``.
* ``learning_rate``: Learning rate for training the autoencoder.
* ``learning_rate_lm``: Learning rate for training the linear model.
* ``batch_size``: Batch size for training. 
* ``dropout_rate``: Dropout rate.
* ``ae_l2_penalty_weight``: Weight of L2 penalty for autoencoder parameters.
* ``ae_type``: Autoencoder type; can be ``variational`` or ``deterministic``.
* ``architecture``: Autoencoder architecture; can be ``convolutional`` or ``fc`` (fully-connected).
* ``anno_indices``: Number of annotations to use. Here, number of SLP (EOF-derived) time series to use as input _X_ to the linear model _h_. 
* ``lm_l2_penalty_weight``: L2 penalty weight for linear model.
* ``num_epochs``: Number of epochs used for training the model.
* ``data_from_exp``: Load hparams from previous experiment.
* ``exp_id``: Experiment id where to load hparams.


Further hyperparameters such as filter and kernel size, as well as the path for the training and evaluation files can be set in the ``config`` file.

#### ``main_linear.py``
The following experimental settings are controlled via command line arguments:
* ``checkpoint_id``: Specifies the checkpoint ID of the autoencoder model that should be loaded. 
* ``precip``: Flag whether the loaded model was trained for precipitation (otherwise temperature).
* ``save_nc_files``: Flag whether to save nc files with predictions.
* ``lin_config``: Path to config file for linear model.

#### ``main_generator.py``
The following experimental settings are controlled via command line arguments:
* ``checkpoint_id``: Specifies the checkpoint ID of the autoencoder model that should be loaded. 
* ``precip``: Flag whether the loaded model was trained for precipitation (otherwise temperature).
* ``save_nc_files``: Flag whether to save nc files with (original as well as emulated) predictions.
* ``var_order``: If set to ``0``, a simple block bootstrap is used. Otherwise, a parameteric bootstrap based on a VAR model with the given order. 
* ``block_size``: Block size for block-bootstrap.
* ``n_bts_samples``: Number of bootstrap samples to generate.
* ``n_steps``: Number of steps to forecast if parametric bootstrap is used. If set to ``0``, forecast will be made for the entire dataset size. 

## Pre-trained models

We provide the checkpoints of the models used to produce the results in the manuscript on Zenodo: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3950045.svg)](https://doi.org/10.5281/zenodo.3950045)

The file ``checkpoints.zip`` needs to be extracted into the directory ``OUT_PATH``. 

For precipitation, the ``CHECKPOINT_ID`` is ``nKGagmsKDb_4249785``. For temperature, it is ``LDifH9DdVh_4383207``. Hence, to e.g. refit the linear model non-iteratively and to produce the evaluation plots as above, run the following command: 

```
python3.7 climate_ae/models/ae/main_linear.py --checkpoint_id='nKGagmsKDb_4249785' --precip=1
```

## ETH-internal: Running on Leonhard

### Installing dependencies

Run the following commands from the root directory of the repository to install the requirements: 
```
bsub -Is -R "rusage[mem=9000, ngpus_excl_p=1]" -R "select[gpu_model1==GeForceGTX1080Ti]" bash
module load python_gpu/3.7.4
module load eth_proxy
python -m venv env
source env/bin/activate
pip install -r requirements.txt
exit
```

### Data
Follow the above steps as described under "Data".


### Running experiments
From the login node, run the following commands to launch the precipitation and temperature experiments, respectively:
```
source env/bin/activate
cd launch_scripts
sh submit-precip.sh
sh submit-temp.sh
```

## References

