import argparse
import os
import numpy as np
import datetime
import pickle
import pdb
from absl import flags, app, logging

logging.set_verbosity(logging.INFO)
import json

import local_settings
import climate_ae.models.utils as utils
from climate_ae.models.ae.train_linear_model import train_linear_model
from climate_ae.experiments_utils.experiment_repo import gen_short_uuid


parser = argparse.ArgumentParser(description='Train linear model.')

parser.add_argument('--checkpoint_id', type = str,
    help='checkpoint directory')
parser.add_argument('--load_json', type = int,
    help='Flag whether to save metrics to json file.')
parser.add_argument('--results_path', type = str,
    help='where to save json')
parser.add_argument('--precip', type = int, 
    help='Flag whether handling precipitation (otherwise temperature).')
parser.add_argument('--offset', type = int, 
    help='')
parser.add_argument('--save_nc_files', type = int,
    help='Flag whether to save nc files.')


parser.add_argument('--lin_config', type=str,
    default="climate_ae/models/ae/configs/config_dyn_adj_precip_linear_era5.json", 
    help='Path to config file for linear model.')#

parser.add_argument('--Debug', type=int, help='only evaluate portion of data to debug')



def main():
    # parse args and get configs

    DEBUG = False

    args = parser.parse_args()
    if args.Debug is not None:
        DEBUG = args.Debug
    

    if DEBUG:
        print('----------------')
        print('### ON DEBUG ###')
        print('----------------')

    #get linear configs
    lin_config = utils.get_config(args.lin_config)
    
    #update linear configs
    lin_config = utils.update_config(lin_config, args)

    #generate unique exp id
    lin_eval_id = gen_short_uuid(10)
    lin_config["lin_eval_id"] = lin_eval_id

    print('----------------')
    print('### Lin Eval Id', lin_eval_id,  '###')
    print('----------------')



    dataset_configs_path = os.path.join(lin_config["data_folder_name"], 'config.json')


    # get results and checkpoint paths

    checkpoint_path = os.path.join(local_settings.OUT_PATH, 'checkpoints')
    checkpoint_folders = os.listdir(checkpoint_path)
    checkpoint_folder = [f for f in checkpoint_folders if lin_config['checkpoint_id'] in f]
    
    if len(checkpoint_folder) == 0:
        raise Exception("No matching folder found.")
    elif len(checkpoint_folder) > 1:
        logging.info(checkpoint_folder)
        raise Exception("More than one matching folder found.")
    else:
        checkpoint_folder = checkpoint_folder[0]
        logging.info("Restoring from {}".format(checkpoint_folder))
    checkpoint_dir = os.path.join(checkpoint_path, checkpoint_folder)
    print(checkpoint_dir)
    

    # get configs from ae_model
    with open(os.path.join(checkpoint_dir, "hparams.pkl"), 'rb') as f:
        ae_configs = pickle.load(f)

    
    #append linear configs
    ae_configs.update(lin_config)


    
    config = utils.config_to_namedtuple(ae_configs)



    # Set up model directory
    current_time = datetime.datetime.now().strftime(r"%y%m%d_%H%M")

    if config.linear_train_length == 'full':
        reduced_ds_str = "false"

    elif isinstance(config.linear_train_length, int):
        reduced_ds_str = str(config.linear_train_length)

    elif isinstance(config.linear_train_length, list):
        reduced_ds_str = '_'.join(['range', str(min(config.linear_train_length)), str(max(config.linear_train_length))])

    else:
        raise ValueError('Check linear_train_length')
    
    if DEBUG:
        reduced_ds_str += '_DEBUG'


    out_folder = '_'.join(["eval_{}_{}".format(current_time, lin_eval_id), 'Scaler', config.scaler, 'reduced_training', reduced_ds_str ])
    out_dir = os.path.join(
        checkpoint_dir, out_folder)

    
    os.makedirs(out_dir, exist_ok=True)
    print('start training...')

#to do
#modfiy trainl linear so only argument is config

    if isinstance(config.linear_train_length, list):
        #datapoints are processed in batches of 64
        # use floor division so that the length of the dataset is a multiple of 64
        datapoint_list = [x // 64 for x in config.linear_train_length]
        lin_config['linear_train_length'] = [x*64 for x in datapoint_list]

        for n_datapoints in datapoint_list:
            print('training for ' + str(n_datapoints * 64) + ' in process...')
            train_linear_model(config, checkpoint_dir, out_dir, reduced_dataset = True, reduced_dataset_len = n_datapoints, DEBUG = DEBUG)

    elif isinstance(config.linear_train_length, int):
        datapoints = config.linear_train_length // 64
        lin_config['linear_train_length'] = datapoints * 64
        train_linear_model(config, checkpoint_dir, out_dir, reduced_dataset = True, reduced_dataset_len = datapoints, DEBUG = DEBUG)
            
    else:
        assert config.linear_train_length == 'full', 'linear_train_length must be either a list of integers, a integer or "full"'
        train_linear_model(config, checkpoint_dir, out_dir, DEBUG = DEBUG)

    #save hyperparams
    
    with open(os.path.join(out_dir, "lin_params.json"), 'w') as f:
        json.dump(lin_config, f, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "lin_params.pkl"), 'wb') as f:
        pickle.dump(lin_config, f)


if __name__ == "__main__":
    main()
