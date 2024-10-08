import datetime
import json
from multiprocessing.dummy import Value
import numpy as np
import os
import copy

import pickle
import shutil
import pandas as pd
from numpy.core.numeric import False_



import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from absl import logging
logging.set_verbosity(logging.INFO)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

import local_settings
from climate_ae.models import utils

from climate_ae.data_generator.datahandler import input_fn
import climate_ae.models.ae.eval_utils as eval_utils
import climate_ae.models.ae.climate_utils as climate_utils

import climate_ae.models.ae.train as train

import xarray as xr
import xesmf as xe




def custom_square(num):
    if num >=0:
        return np.square(num)
    else:
        return np.square(num) * (-1)
    
custom_square_vec = np.vectorize(custom_square)

def custom_root(num):
    if num >=0:
        return np.sqrt(num)
    else:
        return np.sqrt(np.abs(num)) * (-1)
custom_root_vec = np.vectorize(custom_root)

def Simple_Ridge(X_train, X_test, y_train, y_test, alpha = 0.1):
    clf = Ridge(alpha=alpha)
    clf.fit(X_train, y_train)
    return explained_variance_score(y_test, clf.predict(X_test)), clf.score(X_test,y_test)



def load_data(inputs, model, subset=False, reduced_dataset = False, reduced_dataset_len = 0, debug=False):
    # get training data for linear latent space model
    # concat the data into arrays
    for b, features in enumerate(inputs):
        if debug and b % 10 == 0 and b > 0:
            break
        if b % 100 == 0:                    
            print(b)

        if reduced_dataset and b % reduced_dataset_len == 0 and b > 0:
            break
   
        input_ = features["inputs"]
        recon_ = model.autoencode(input_, training=False)["output"]
        anno_ = train.get_annotations(features)
        year_ = features["year"]
        month_ = features["month"]
        day_ = features["day"]
        encodings_ = model.mean_encode(input_, training=False)['z'].numpy()
        # encodings_z = encodings['z'].numpy()
        
        if b == 0:
            inputs = input_
            recons = recon_
            latents = encodings_
            annos = anno_
            years = year_
            months = month_
            days = day_
        else:
            latents = np.r_[latents, encodings_] #np.r_ -> concat array
            annos = np.r_[annos, anno_]
            if subset and b <= 10:
                # just keep a subset in memory
                inputs = np.r_[inputs, input_]       
                recons = np.r_[recons, recon_]
                years = np.r_[years, year_]
                months = np.r_[months, month_]
                days = np.r_[days, day_]
            else:
                inputs = np.r_[inputs, input_]       
                recons = np.r_[recons, recon_]
                years = np.r_[years, year_]
                months = np.r_[months, month_]
                days = np.r_[days, day_]
    return inputs, recons, latents, annos, years, months, days


def load_data_predictions_only(inputs, subset=False, reduced_dataset = False, reduced_dataset_len = 0, debug=False):
    # for dataset with different precipitation shape
   
    for b, features in enumerate(inputs):
        if debug and b % 10 == 0 and b > 0:
            break
        if b % 100 == 0:
            print(b)

        if reduced_dataset and b % reduced_dataset_len == 0 and b > 0:
            break
   
        input_ = features["inputs"]

        anno_ = train.get_annotations(features)
        year_ = features["year"]
        month_ = features["month"]
        day_ = features["day"]

        
        if b == 0:
            inputs = input_
            annos = anno_
            years = year_
            months = month_
            days = day_
        else:
            annos = np.r_[annos, anno_]
            if subset and b <= 10:
                # just keep a subset in memory
                inputs = np.r_[inputs, input_]       
                years = np.r_[years, year_]
                months = np.r_[months, month_]
                days = np.r_[days, day_]
            else:
                inputs = np.r_[inputs, input_]       
                years = np.r_[years, year_]
                months = np.r_[months, month_]
                days = np.r_[days, day_]
    return inputs, annos, years, months, days


def predict_latents_and_decode(model, reg_model, annos, out_shape, square_annos):
    #print('out_shape ', out_shape)
     # predict latents

    latentshat = reg_model.predict(annos)
    if square_annos:
        latentshat = custom_root_vec(latentshat)


    # decode predicted latents
    xhatexp = np.zeros(out_shape)
    for i in range(xhatexp.shape[0]):
        xhatexp[i, ...] = model.decode(np.expand_dims(latentshat[i, ...], 
            axis=0), training=False)["output"]
    
    return xhatexp



def process_holdout_train_grid(label, holdout_inputs,  model, config, reg_model, sqrt_out_dir, orig_out_dir, DEBUG = False):


    if config.eval_years != 'all' and DEBUG == False:
        ho_inputs, ho_recons, _, ho_annos, ho_years, ho_months, ho_days = \
            load_data(holdout_inputs, model, reduced_dataset = True, reduced_dataset_len = config.eval_years, subset=True,debug=DEBUG)

    else:
        ho_inputs, ho_recons, _, ho_annos, ho_years, ho_months, ho_days = \
            load_data(holdout_inputs, model, debug=DEBUG)

    # predict latents for holdout set and decode
    ho_xhatexp = predict_latents_and_decode(model, reg_model, ho_annos, 
        np.shape(ho_inputs), config.square_annos)


    # save
    if config.save_nc_files:
        climate_utils.save_ncdf_file_high_res_prec(ho_inputs, ho_years, ho_months, 
            ho_days, "ho_{}_input.nc".format(label),sqrt_out_dir)
        climate_utils.save_ncdf_file_high_res_prec(ho_xhatexp, ho_years, ho_months, 
            ho_days, "ho_{}_pred.nc".format(label), sqrt_out_dir)


    r2_maps_ho = eval_utils.plot_r2_map(ho_inputs, ho_recons, 
        ho_xhatexp, sqrt_out_dir, "holdout_{}".format(label)) 

    mse_map_ho = eval_utils.plot_mse_map(ho_inputs, ho_recons, ho_xhatexp, 
        sqrt_out_dir, "holdout_{}".format(label)) 
    mean_mse_x_xhat = np.mean(mse_map_ho[0])
    mean_mse_x_xhatexp = np.mean(mse_map_ho[1])
    mean_r2_x_xhat = np.mean(r2_maps_ho[0])
    mean_r2_x_xhatexp = np.mean(r2_maps_ho[1])


    # R2 map
    r2_maps_ho = eval_utils.plot_r2_map(ho_inputs, ho_recons, ho_xhatexp, 
        sqrt_out_dir, "holdout_{}".format(label), hist = False) 
    np.save(os.path.join(sqrt_out_dir, "r2map_xxhat_holdout_{}.npy".format(label)), r2_maps_ho[0])
    np.save(os.path.join(sqrt_out_dir, "r2map_xxhatexp_holdout_{}.npy".format(label)), r2_maps_ho[1])

    # MSE Map
    mse_map_ho = eval_utils.plot_mse_map(ho_inputs, ho_recons, ho_xhatexp, 
        sqrt_out_dir, "holdout_{}".format(label)) 
    np.save(os.path.join(sqrt_out_dir, "mse_map_xxhat_holdout_{}.npy".format(label)), mse_map_ho[0])
    np.save(os.path.join(sqrt_out_dir, "mse_map_xxhatexp_holdout_{}.npy".format(label)), mse_map_ho[1])

    # visualize reconstructions and interventions -- random
    # imgs_ho = eval_utils.visualize(ho_inputs, ho_annos, model, reg, sqrt_out_dir, 
    #     "holdout")
    # np.save(os.path.join(sqrt_out_dir, "ho_x_holdout_{}.npy".format(label)), imgs_ho[0])
    # np.save(os.path.join(sqrt_out_dir, "ho_xhat_holdout_{}.npy".format(label)), imgs_ho[1])
    # np.save(os.path.join(sqrt_out_dir, "ho_xhatexp_holdout_{}.npy".format(label)), imgs_ho[2])

    eval_utils.visualize(ho_inputs, ho_annos, model, reg_model, sqrt_out_dir, 
        "holdout_{}".format(label)) 


    print("\n#### Holdout ensemble: {}".format(label))
    print("Mean MSE(x, xhat): {}".format(mean_mse_x_xhat))
    print("Mean MSE(x, xhatexp): {}".format(mean_mse_x_xhatexp))
    print("Mean R2(x, xhat): {}".format(mean_r2_x_xhat))
    print("Mean R2(x, xhatexp): {}".format(mean_r2_x_xhatexp))
    
    # save metrics again in checkpoint dir
    save_path = os.path.join(sqrt_out_dir, "metrics_{}.json".format(label))
    metrics = {'mean_mse_x_xhat': mean_mse_x_xhat, 
        'mean_mse_x_xhatexp': mean_mse_x_xhatexp,
        'mean_r2_x_xhat': mean_r2_x_xhat,
        'mean_r2_x_xhatexp': mean_r2_x_xhatexp}

    with open(save_path, 'w') as result_file:
        json.dump(metrics, result_file, sort_keys=True, indent=4)

    if config.precip: 
        ho_inputs_2 = ho_inputs ** 2
        ho_recons_2 = ho_recons ** 2
        ho_xhatexp_2 = ho_xhatexp ** 2
        if config.offset:
            ho_inputs_2 = ho_inputs_2 - 25
            ho_recons_2 = ho_recons_2 - 25
            ho_xhatexp_2 = ho_xhatexp_2 - 25 


        r2_maps_ho_orig = eval_utils.plot_r2_map(ho_inputs_2, ho_recons_2, 
            ho_xhatexp_2, orig_out_dir, "holdout_orig_{}".format(label)) 
        mse_map_ho_orig = eval_utils.plot_mse_map(ho_inputs_2, ho_recons_2, 
            ho_xhatexp_2, orig_out_dir, "holdout_orig_{}".format(label)) 
        mean_mse_x_xhat = np.mean(mse_map_ho_orig[0])
        mean_mse_x_xhatexp = np.mean(mse_map_ho_orig[1])
        mean_r2_x_xhat = np.mean(r2_maps_ho_orig[0])
        mean_r2_x_xhatexp = np.mean(r2_maps_ho_orig[1])
        eval_utils.visualize(ho_inputs, ho_annos, model, reg_model, orig_out_dir, 
            "holdout_orig_{}".format(label), transform_back=True, offset=config.offset) 


        
        # R2 map
        r2_maps_ho = eval_utils.plot_r2_map(ho_inputs_2, ho_recons_2, 
                ho_xhatexp_2, orig_out_dir, "holdout_{}".format(label), hist = False) 
        np.save(os.path.join(orig_out_dir, "r2map_xxhat_holdout_{}.npy".format(label)), r2_maps_ho[0])
        np.save(os.path.join(orig_out_dir, "r2map_xxhatexp_holdout_{}.npy".format(label)), r2_maps_ho[1])

        # MSE Map
        mse_map_ho = eval_utils.plot_mse_map(ho_inputs_2, ho_recons_2, 
                ho_xhatexp_2, orig_out_dir, "holdout_{}".format(label)) 
        np.save(os.path.join(orig_out_dir, "mse_map_xxhat_holdout_{}.npy".format(label)), mse_map_ho[0])
        np.save(os.path.join(orig_out_dir, "mse_map_xxhatexp_holdout_{}.npy".format(label)), mse_map_ho[1])


        print("\n# Orig: {}".format(label))
        print("Mean MSE(x, xhat): {}".format(mean_mse_x_xhat))
        print("Mean MSE(x, xhatexp): {}".format(mean_mse_x_xhatexp))
        print("Mean R2(x, xhat): {}".format(mean_r2_x_xhat))
        print("Mean R2(x, xhatexp): {}".format(mean_r2_x_xhatexp))

        # save metrics again in checkpoint dir
        save_path = os.path.join(orig_out_dir, "metrics_orig_{}.json".format(label))
        metrics = {'mean_mse_x_xhat': mean_mse_x_xhat, 
            'mean_mse_x_xhatexp': mean_mse_x_xhatexp,
            'mean_r2_x_xhat': mean_r2_x_xhat,
            'mean_r2_x_xhatexp': mean_r2_x_xhatexp}

        with open(save_path, 'w') as result_file:
            json.dump(metrics, result_file, sort_keys=True, indent=4)




def train_linear_model(config, checkpoint_path, out_dir, reduced_dataset = False, reduced_dataset_len = 0, DEBUG = False):
    model, _ = train.get_models(config)

 
        ### Remove after ####
    grid_configs = 'climate_ae/models/ae/configs/grid_configs'

    #get training configs
    train_grid = config.train_grid
    train_file_grid_conf_path = os.path.join(local_settings.CODE_PATH, grid_configs, train_grid + '.json')
    train_file_grid_conf_dict = utils.get_config(train_file_grid_conf_path)
    train_file_grid_conf = utils.config_to_namedtuple(train_file_grid_conf_dict)


    #load model info
    df_model_info = pd.read_csv(os.path.join(local_settings.DATA_PATH,config.file_info_models))
    df_observations_info = pd.read_csv(os.path.join(local_settings.DATA_PATH,config.file_info_observations))

    #get folder for grid congis
    #grid_configs = config.grid_configs


    #load the mask
    if config.land_values_only:
        mask = np.load(os.path.join(local_settings.DATA_PATH, 'grids/masks/{}.npy'.format(train_grid)))
    else:
        #if no masking, just load a mask with only ones
        mask = np.ones((train_file_grid_conf.grid_shape[0], train_file_grid_conf.grid_shape[1]))

    


    # input function
    def input_anno(file, grid, params, repeat, n_repeat=None):
     
        dataset = input_fn(file = file, grid = grid, params=params, repeat=repeat, 
            n_repeat=n_repeat, shuffle=False)

        dataset = dataset.map(lambda x:
            {"inputs": x["inputs"],
            "anno": tf.gather(x["anno"], params.anno_indices, axis=1),
            "year": x["year"],
            "month": x["month"], 
            "day": x["day"]
            })
            
        return dataset

    global_step = tf.Variable(initial_value=0, dtype=tf.int64, trainable=False, 
        name="global_step")


    train_inputs = input_anno(os.path.join(config.train_subdir, config.train_file), grid = config.train_grid, params = config, repeat=False) 

    if not config.split_tr_file_in_tr_and_te:
        test_inputs = input_anno(os.path.join(config.test_subdir, config.test_file) , grid = config.train_grid, params = config, repeat=False)
        
    #ds_test2 = input_anno(config.holdout_file, grid = config.holdout_grid, params = config, repeat=False)

    # dummy run - otherwise, the model wouldn't be fully build
    show_inputs = iter(train_inputs)
    _ = model(next(show_inputs)["inputs"])
    
    # restore model from checkpoint
    checkpoint = tf.train.Checkpoint(model=model, global_step=global_step)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)
    status = checkpoint.restore(manager.latest_checkpoint)
    status.assert_consumed()


    #np.save('/home/jkuettel/latent-linear-adjustment-autoencoders/plotting_notebooks/final_plots_paper/meeting_07_06_nicolai/tr_annos.npy', tr_annos)
    #np.save('/home/jkuettel/latent-linear-adjustment-autoencoders/plotting_notebooks/final_plots_paper/meeting_07_06_nicolai/tr_latents.npy', tr_latents)
    
    for scaler in ['no_scaler']:

    
        print(f" \n #### Evaluating {scaler} ##### \n ")


        out_dir_scaler = os.path.join(out_dir, scaler)
        if scaler == 'no_scaler':
            if config.square_annos:
                
                print('### Scaling the Annos ###')
                
                reg = LinearRegression().fit(tr_annos, custom_square_vec(tr_latents))
            else:
                print('### Do not Scaling the Annos ###')
                print('number of nans annos :', np.sum(np.isnan(tr_annos))) 
                print('number of nans latents :', np.sum(np.isnan(tr_latents)))
                

                reg = LinearRegression().fit(tr_annos, tr_latents)

        elif scaler == 'Ridge':
            print('\n******** ')
            
            kfold = KFold(n_splits=10, shuffle=False)
            alphas =  config.alphas_ridge
            r2_score_arr = np.zeros(len(alphas))
            r2_score_arr_te = np.zeros(len(alphas))


            for i, alpha in enumerate(alphas):
                print('alpha: ', alpha)
                
                r2_score = []
                r2_score_te = []

                for tr, te in kfold.split(tr_annos, tr_latents):

                    ridge_model = Ridge(alpha = alpha, normalize = True)
                    reg = ridge_model.fit(tr_annos[tr], tr_latents[tr])

                    # predict latents for test set and decode
                    xhatexp_alpha_tr = predict_latents_and_decode(model, reg, tr_annos[te], 
                        np.shape(tr_inputs[te]), config.square_annos)

                    
                    r2_score.append(np.nanmean(eval_utils.compute_r2_map_3d(np.flip(tr_inputs[te] ** 2 , axis = 1), np.flip(xhatexp_alpha_tr ** 2 , axis = 1))* mask ))
                                    

                                    
                xhatexp_alpha_te = predict_latents_and_decode(model, reg, te_annos, 
                        np.shape(te_inputs), config.square_annos)
                r2_score_te = np.nanmean(eval_utils.compute_r2_map_3d(np.flip(te_inputs ** 2 , axis = 1), np.flip(xhatexp_alpha_te ** 2 , axis = 1))* mask)


                r2_score_arr[i] = np.mean(r2_score)
                r2_score_arr_te[i] = r2_score_te
                print('r2 score: ', np.mean(r2_score))
                print('r2 score test: ', r2_score_te)
                print('')
            
            print('best scorevtr: ', np.max(r2_score_arr))
            best_alpha = alphas[np.argmax(r2_score_arr) - 1 ] if np.argmax(r2_score_arr) != 0 else alphas[np.argmax(r2_score_arr)]
            print('best alpha tr: ', best_alpha)


            print('')


            print('best score test: ', np.max(r2_score_arr_te))
            best_alpha_te = alphas[np.argmax(r2_score_arr_te)]
            print('best alpha test: ', best_alpha_te)


            print('-------')
            reg = Ridge(alpha = best_alpha, normalize = True).fit(tr_annos, tr_latents)



#############
            if config.square_annos:
                reg = reg.fit(tr_annos, custom_square_vec(tr_latents))
            else:
                reg = reg.fit(tr_annos, tr_latents)

#######
            # # summarize result
            # print('Best Score: %s' % reg.best_score_)
            # print('Best Hyperparameters: %s' % reg.best_params_)
#######           
            
        else:
            raise ValueError('scaler not found')


        if isinstance(config.linear_train_length, list):
            out_dir_scaler_sqrt = os.path.join(out_dir_scaler, 'sqrt_' + str(reduced_dataset_len * 64))
            out_dir_scaler_orig = os.path.join(out_dir_scaler, 'orig_' + str(reduced_dataset_len * 64))
        
        else:
            out_dir_scaler_sqrt = os.path.join(out_dir_scaler, 'sqrt_data')
            out_dir_scaler_orig = os.path.join(out_dir_scaler, 'orig_data')



        ## set up folder to save results ##
        if isinstance(config.linear_train_length, list):
            orig_out_dir = os.path.join(out_dir_scaler, 'orig_' + str(reduced_dataset_len * 64))
        else:
            orig_out_dir = os.path.join(out_dir_scaler, 'orig_data')
        #create dirs
        os.makedirs(orig_out_dir, exist_ok=True)
    
        # predict latents for test set and decode
        tr_xhatexp = predict_latents_and_decode(model, reg, tr_annos, 
            np.shape(tr_inputs), config.square_annos)

        te_xhatexp = predict_latents_and_decode(model, reg, te_annos, 
            np.shape(te_inputs), config.square_annos)

        #save
        if config.save_nc_files:
            climate_utils.save_ncdf_file_high_res_prec(tr_inputs, tr_years, 
                tr_months, tr_days, "tr_input.nc", out_dir_scaler)
            climate_utils.save_ncdf_file_high_res_prec(te_inputs, te_years, 
                te_months, te_days, "te_input.nc", out_dir_scaler)

            climate_utils.save_ncdf_file_high_res_prec(tr_xhatexp, tr_years, 
                tr_months, tr_days, "tr_pred.nc", out_dir_scaler)
            climate_utils.save_ncdf_file_high_res_prec(te_xhatexp, te_years, 
                te_months, te_days, "te_pred.nc", out_dir_scaler)
        
        sqrt_test_metrics = {}

 
        if 'E-OBS' in config.test_file: #more than 200 nans, full analysis not possible
            #r2_map_xhatexp_sqrt = eval_utils.plot_r2_map_xhatexp_only(te_inputs, te_xhatexp, out_dir_scaler_sqrt, 'r2_map_sqrt_te', pdf = False)

            if config.scale_eobs:

                te_xhatexp = te_xhatexp / np.sqrt(config.scale_factor)

            r2_map_xhatexp_sqrt = eval_utils.plot_r2_map_xhatexp_only(np.flip(te_inputs , axis = 1), np.flip(te_xhatexp , axis = 1), out_dir_scaler_sqrt, 'r2_map_sqrt_te', pdf = False) * mask
            sqrt_test_metrics.update({"mean_mse_x_xhatexp": np.nanmean(r2_map_xhatexp_sqrt)})
            print(f"test score r2_score_sqrt: {np.nanmean(r2_map_xhatexp_sqrt)}")


        else:

            # R2 map
            r2_maps_test = eval_utils.plot_r2_map(te_inputs, te_recons, te_xhatexp, 
                out_dir_scaler_sqrt, "test") 
            np.save(os.path.join(out_dir_scaler_sqrt, "r2map_test_xxhat.npy"), r2_maps_test[0])
            np.save(os.path.join(out_dir_scaler_sqrt, "r2map_test_xxhatexp.npy"), r2_maps_test[1])

            # MSE map 
            mse_map_test = eval_utils.plot_mse_map(te_inputs, te_recons, te_xhatexp, 
                out_dir_scaler_sqrt, "test") 
            np.save(os.path.join(out_dir_scaler_sqrt, "mse_map_test_xxhat.npy"), mse_map_test[0])
            np.save(os.path.join(out_dir_scaler_sqrt, "mse_map_test_xxhatexp.npy"), mse_map_test[1])

            # mean MSE over entire field
            
            sqrt_test_metrics.update({"mean_mse_x_xhat": np.mean(mse_map_test[0])})
            sqrt_test_metrics.update({"mean_mse_x_xhatexp": np.mean(mse_map_test[1])})
            # mean R2 over entire field
            sqrt_test_metrics.update({"mean_r2_x_xhat": np.mean(r2_maps_test[0])})
            sqrt_test_metrics.update({"mean_r2_x_xhatexp": np.mean(r2_maps_test[1])})

            print(" \n #### Test Sqrt Metrics: ")
            print("Mean MSE(x, xhat): {}".format(sqrt_test_metrics["mean_mse_x_xhat"]))
            print("Mean MSE(x, xhatexp): {}".format(sqrt_test_metrics["mean_mse_x_xhatexp"]))
            print("Mean R2(x, xhat): {}".format(sqrt_test_metrics["mean_r2_x_xhat"]))
            print("Mean R2(x, xhatexp): {}".format(sqrt_test_metrics["mean_r2_x_xhatexp"]))





        # Transform Back to original scale to evaluate Original Data

        te_inputs_2 = te_inputs ** 2
        te_recons_2 = te_recons ** 2
        te_xhatexp_2 = te_xhatexp ** 2

        if config.offset:
            te_inputs_2 = te_inputs_2 - 25
            te_recons_2 = te_recons_2 - 25
            te_xhatexp_2 = te_xhatexp_2 - 25

        orig_test_metrics = {}
    

        if 'E-OBS' in config.test_file:

            if config.scale_eobs:

                    te_xhatexp = te_xhatexp / np.sqrt(config.scale_factor)

            r2_map_xhatexp_orig = eval_utils.plot_r2_map_xhatexp_only(np.flip(te_inputs_2, axis = 1), np.flip(te_xhatexp_2 , axis = 1), out_dir_scaler_sqrt, 'r2_map_orig_te', pdf = False) * mask
            orig_test_metrics.update({"mean_mse_x_xhatexp": np.nanmean(r2_map_xhatexp_orig)})
            print(f"test score r2_score_orig: {np.nanmean(r2_map_xhatexp_orig)}")

        else:

            r2_maps_test_orig = eval_utils.plot_r2_map(te_inputs_2, te_recons_2, 
                te_xhatexp_2, out_dir_scaler_orig, "test_orig") 
            np.save(os.path.join(out_dir_scaler_orig, "r2map_test_orig_xxhat.npy"), 
                r2_maps_test_orig[0])
            np.save(os.path.join(out_dir_scaler_orig, "r2map_test_orig_xxhatexp.npy"), 
                r2_maps_test_orig[1])
        
            mse_maps_test_orig = eval_utils.plot_mse_map(te_inputs_2, te_recons_2, 
                te_xhatexp_2, out_dir_scaler_orig, "test_orig") 
            np.save(os.path.join(out_dir_scaler_orig, "mse_map_test_orig_xxhat.npy"), 
                mse_maps_test_orig[0])
            np.save(os.path.join(out_dir_scaler_orig, "mse_map_test_orig_xxhatexp.npy"), 
                mse_maps_test_orig[1])
            
            # mean MSE over entire field

            orig_test_metrics.update({"mean_mse_x_xhat": np.mean(mse_maps_test_orig[0])})
            orig_test_metrics.update({"mean_mse_x_xhatexp": np.mean(mse_maps_test_orig[1])})
            # mean R2 over entire field
            orig_test_metrics.update({"mean_r2_x_xhat": np.mean(r2_maps_test_orig[0])})
            orig_test_metrics.update({"mean_r2_x_xhatexp": np.mean(r2_maps_test_orig[1])})


            print(" \n #### Test Orig Metrics: ")
            print("Mean MSE(x, xhat): {}".format(orig_test_metrics["mean_mse_x_xhat"]))
            print("Mean MSE(x, xhatexp): {}".format(orig_test_metrics["mean_mse_x_xhatexp"]))
            print("Mean R2(x, xhat): {}".format(orig_test_metrics["mean_r2_x_xhat"]))
            print("Mean R2(x, xhatexp): {}".format(orig_test_metrics["mean_r2_x_xhatexp"]))


        #################################
        #### Evaluate holdout files #####
        #################################




        if config.lin_holdout_data_from_folder:
            folder_files = os.listdir(os.path.join(local_settings.DATA_PATH, config.data_folder_name, 'tfrecords_data')) 
            holdout_file_names = [ho for ho in folder_files if 'holdout' in ho] #list with the filenames
            holdout_identifiers = [ho.split('_holdout_')[0] for ho in holdout_file_names] #list with the identifiers
            print(holdout_file_names)
            holdout_dict = dict(zip(holdout_identifiers, holdout_file_names))
        else:
            raise NotImplementedError


        ### Now loop over all the holdouts
        

        for eval_file in holdout_dict:
            #if('E-OBS_025' in eval_file or 'E-OBS' in eval_file or 'ERA5' in eval_file or 'gmp-imerg' in eval_file): ## still causes problems, see screenshot e-obs_problem
            #if ("E-OBS_CORDEX11" not in eval_file):
            #    continue

            print(f"\n Evaluating holout {eval_file} \n")


            sqrt_out_dir_eval_file = os.path.join(out_dir_scaler_sqrt, eval_file)
            orig_out_dir_eval_file = os.path.join(out_dir_scaler_orig, eval_file)

            os.makedirs(sqrt_out_dir_eval_file, exist_ok=True)
            os.makedirs(orig_out_dir_eval_file, exist_ok=True)

            eval_file_path = os.path.join(config.data_folder_name, 'tfrecords_data', holdout_dict[eval_file])

            if max(df_model_info["Identifier"].str.contains(eval_file)):
                df_file_info = df_model_info
                #state = "model"
                state = "obs"
               
            elif max(df_observations_info["Identifier"].str.contains(eval_file)):
                df_file_info = df_observations_info
                state = "obs"
            else:
                raise ValueError(F"Eval file {eval_file}could not be found in the Dataframes")

            info_col_eval_file = df_file_info[df_file_info["Identifier"].str.contains(eval_file)]
            
            if info_col_eval_file.shape[0] != 1:
                raise ValueError(f"More than one column found for file {eval_file}")
                    
            
            eval_file_grid = info_col_eval_file["grid"].item()

            holdout_inputs = input_anno(file = eval_file_path, grid = eval_file_grid, params=config, repeat=False)

            #if eval_file_grid == config.train_grid:
            if state == "model":
                if config.reanalysis_only == True:
                    continue
                # the pr file to evaluate has the same grid as the training data
                # no regridding needed, the files an be processed as usual
                # the precip values can be compared directly
                # process and save predictions for additional holdout datasets
                else:
                    results = process_holdout_train_grid(eval_file, holdout_inputs, model, config, reg, sqrt_out_dir_eval_file, orig_out_dir_eval_file, DEBUG = DEBUG)

            else:
                

            # the pr file to evaluate has not the same grid as the training data hence not the same dimensions
            # to compare predicitons and true values, regridding is needed

                #get eval grid configs


                eval_file_grid_conf_path = os.path.join(local_settings.CODE_PATH, grid_configs, eval_file_grid + '.json')
                eval_file_grid_conf_dict = utils.get_config(eval_file_grid_conf_path)
                eval_file_grid_conf = utils.config_to_namedtuple(eval_file_grid_conf_dict)


                #pdb.set_trace()
                # get holdout data
                eval_inputs, eval_annos, eval_years, eval_months, eval_days = \
                    load_data_predictions_only(holdout_inputs, model, debug=DEBUG)
                #construct the shape of era5 predictions

                #get shape for eval_xhatexp predictions (has to be the same as the training grid) 
                eval_predictions_shape  = copy.deepcopy(train_file_grid_conf.grid_shape)

                #insert the time dimension
                eval_predictions_shape.insert(0, eval_inputs.shape[0])

                # predict latents for holdout set and decode
                eval_xhatexp = predict_latents_and_decode(model, reg, eval_annos, 
                    tuple(eval_predictions_shape), config.square_annos)
                    
                
                eval_xhatexp_2 = np.square(eval_xhatexp)

                if config.save_nc_files:
                    climate_utils.save_ncdf_file_high_res_prec(eval_inputs, eval_years, 
                            eval_months, eval_days, eval_file + "_input.nc", sqrt_out_dir_eval_file)
        
                    climate_utils.save_ncdf_file_high_res_prec(eval_xhatexp, eval_years, 
                            eval_months, eval_days, eval_file + "_xhatexp_sqrt.nc", sqrt_out_dir_eval_file)

                    climate_utils.save_ncdf_file_high_res_prec(eval_xhatexp_2, eval_years, 
                            eval_months, eval_days, eval_file + "_xhatexp_orig.nc", orig_out_dir_eval_file)

                #flip predicitons since all predictions are upside down
                eval_xhatexp = np.flip(eval_xhatexp, axis = 1)


                ########### REMOVE AFTER #################
                if config.scale_eobs:
 
                    eval_xhatexp = eval_xhatexp / np.sqrt(config.scale_factor)

                ############################################


                #reshape so that we have same dimension as xarray
                eval_xhatexp_shape = eval_xhatexp.shape
                eval_xhatexp = eval_xhatexp.reshape(eval_xhatexp_shape[0], eval_xhatexp_shape[1], eval_xhatexp_shape[2])

                if eval_file_grid != config.train_grid:
                    train_grid_path =  os.path.join(local_settings.DATA_PATH, train_file_grid_conf.grid_path) 
                    eval_grid_path = os.path.join(local_settings.DATA_PATH, eval_file_grid_conf.grid_path)

                    #load gridsdataset
                    train_grid = np.load(train_grid_path,allow_pickle='TRUE').item()
                    eval_grid = np.load(eval_grid_path, allow_pickle='TRUE').item()

                    #regrid data
                    #reshape so that regridder can read it

                    regridder = xe.Regridder(train_grid, eval_grid, 'bilinear')
                    reshaped_eval_xhatexp = regridder(eval_xhatexp)

                    eval_mask = np.load(os.path.join(local_settings.DATA_PATH, eval_file_grid_conf.mask_path))
                    reshaped_eval_xhatexp_masked = eval_mask * reshaped_eval_xhatexp
                    eval_xhatexp = reshaped_eval_xhatexp_masked


                #load raw pr data
                raw_pr_eval_ds = xr.open_dataset(os.path.join(local_settings.DATA_PATH, info_col_eval_file["pr_path"].item()))
                raw_pr_eval = raw_pr_eval_ds[info_col_eval_file["pr_variable_name"].item()].values 


                if info_col_eval_file["pr_format"].item() == "m/day":
                    raw_pr_eval = raw_pr_eval * 1000
                
                elif info_col_eval_file["pr_format"].item() == "mm/s":
                    raw_pr_eval = raw_pr_eval * 3600 * 24

                elif info_col_eval_file["pr_format"].item() == "mm/day":
                        raw_pr_eval = raw_pr_eval 

                else:
                    raise ValueError("raw pr_format not known")

                

                if config.eval_years != 'all' and DEBUG == False:
                    
                    datapoints = config.eval_years * 90 
                    assert datapoints <= eval_xhatexp.shape[0], "eval years are more than the available years in the dataset"
    
                    eval_xhatexp = eval_xhatexp[:datapoints, :, :]
                    raw_pr_eval = raw_pr_eval[:datapoints, :, :]
                    print(f"Cutting holdout files, the eval_xhatexp shape: {eval_xhatexp.shape} raw_pr_eval shape: {raw_pr_eval.shape}")

                else:
                    raw_pr_eval = raw_pr_eval[:eval_xhatexp.shape[0], :, :] #since values are read in batches of 64
                
                if raw_pr_eval.shape != eval_xhatexp.shape:
                    raise ValueError("true era5 and masked era5 dont have same values, true shape: ", raw_pr_eval.shape, " masked shape: ", reshaped_eval_xhatexp_masked.shape)
                



                #raw_pr_eval_masked = raw_pr_eval * eval_mask
                
                plot_dir_sqrt = os.path.join(local_settings.CODE_PATH, "plots/test_plots_obs/sqrt")
                plot_dir_orig = os.path.join(local_settings.CODE_PATH, "plots/test_plots_obs/orig")

                eval_xhatexp_2 = np.square(eval_xhatexp)
                raw_pr_eval_sqrt = np.sqrt(raw_pr_eval)

                #save regridded predictions
                np.save(os.path.join(sqrt_out_dir_eval_file, '{}_xhatexp_sqrt.npy'.format(eval_file)), eval_xhatexp)
                np.save(os.path.join(orig_out_dir_eval_file, '{}_xhatexp_orig.npy'.format(eval_file)), eval_xhatexp_2)

                
                r2_map_xhatexp_sqrt = eval_utils.plot_r2_map_xhatexp_only(raw_pr_eval_sqrt, eval_xhatexp, plot_dir_sqrt, 'r2_map_{}_sqrt'.format(eval_file), pdf = False)
                r2_map_xhatexp_orig = eval_utils.plot_r2_map_xhatexp_only(raw_pr_eval, eval_xhatexp_2, plot_dir_orig, 'r2_map_{}_orig'.format(eval_file), pdf = False)

    
                
                r2_map_xhatexp_sqrt = r2_map_xhatexp_sqrt * mask
                r2_map_xhatexp_orig = r2_map_xhatexp_orig * mask

                np.save(os.path.join(sqrt_out_dir_eval_file, '{}_r2map.npy'.format(eval_file)), r2_map_xhatexp_sqrt)
                print(f"ho member: {eval_file} r2_score_sqrt: {np.nanmean(r2_map_xhatexp_sqrt)}")

                np.save(os.path.join(orig_out_dir_eval_file, '{}_r2map.npy'.format(eval_file)), r2_map_xhatexp_orig)
                print(f"ho member: {eval_file} r2_score_orig: {np.nanmean(r2_map_xhatexp_orig)}")

                #### METRICS ####

                # save metrics again in checkpoint dir
                metrics_sqrt_path = os.path.join(sqrt_out_dir_eval_file, "metrics_{}.json".format(eval_file))
                metrics_orig_path = os.path.join(orig_out_dir_eval_file, "metrics_{}.json".format(eval_file))
                
                sqrt_metrics_eval_file = {} 
                orig_metrics_eval_file = {} 

                sqrt_metrics_eval_file["mean_r2_x_xhatexp_not_adjusted"] =  np.nanmean(r2_map_xhatexp_sqrt)
                orig_metrics_eval_file["mean_r2_x_xhatexp_not_adjusted"] =  np.nanmean(r2_map_xhatexp_orig)


                if config.eval_adjusted_mean:
                    #sqrt values
                    raw_pr_eval_sqrt = raw_pr_eval_sqrt - raw_pr_eval_sqrt.mean(axis = 0)
                    eval_xhatexp = eval_xhatexp - eval_xhatexp.mean(axis = 0)   

                    #orig values
                    raw_pr_eval = raw_pr_eval - raw_pr_eval.mean(axis = 0)
                    eval_xhatexp_2 = eval_xhatexp_2 - eval_xhatexp_2.mean(axis = 0)

                    #save regridded predictions
                    np.save(os.path.join(sqrt_out_dir_eval_file, '{}_xhatexp_sqrt_mean_adjusted.npy'.format(eval_file)), eval_xhatexp)
                    np.save(os.path.join(orig_out_dir_eval_file, '{}_xhatexp_orig_mean_adjusted.npy'.format(eval_file)), eval_xhatexp_2)

                    r2_map_xhatexp_sqrt_adjusted = eval_utils.plot_r2_map_xhatexp_only(raw_pr_eval_sqrt, eval_xhatexp, plot_dir_sqrt, 'r2_map_mean_adjusted{}_sqrt'.format(eval_file), pdf = False)
                    r2_map_xhatexp_orig_adjusted = eval_utils.plot_r2_map_xhatexp_only(raw_pr_eval, eval_xhatexp_2, plot_dir_orig, 'r2_map_mean_adjusted{}_orig'.format(eval_file), pdf = False)

                    r2_map_xhatexp_sqrt_adjusted = r2_map_xhatexp_sqrt_adjusted * mask
                    r2_map_xhatexp_orig_adjusted = r2_map_xhatexp_orig_adjusted * mask     

                    np.save(os.path.join(sqrt_out_dir_eval_file, '{}_r2map_mean_adjusted.npy'.format(eval_file)), r2_map_xhatexp_sqrt_adjusted)
                    print(f"ho member: {eval_file} r2_score_mean_adjusted_sqrt: {np.nanmean(r2_map_xhatexp_sqrt_adjusted)}")

                    np.save(os.path.join(orig_out_dir_eval_file, '{}_r2map_mean_adjusted.npy'.format(eval_file)), r2_map_xhatexp_orig_adjusted)
                    print(f"ho member: {eval_file} r2_score_mean_adjusted_orig: {np.nanmean(r2_map_xhatexp_orig_adjusted)}")

                    sqrt_metrics_eval_file["mean_r2_x_xhatexp_not_adjusted"] =  np.nanmean(r2_map_xhatexp_sqrt)
                    orig_metrics_eval_file["mean_r2_x_xhatexp_not_adjusted"] =  np.nanmean(r2_map_xhatexp_orig)

                    #note that since the inputs have a different shape, y_hat can not be calculated

                with open(metrics_sqrt_path, 'w') as result_file:
                    json.dump(sqrt_metrics_eval_file, result_file, sort_keys=True, indent=4)

                with open(metrics_orig_path, 'w') as result_file:
                    json.dump(orig_metrics_eval_file, result_file, sort_keys=True, indent=4)
        
        