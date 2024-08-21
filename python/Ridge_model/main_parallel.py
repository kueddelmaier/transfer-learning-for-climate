
import xarray as xr
import os
from climate_utils import *
from models import fit_ridge
from multiprocessing import Process, Queue


#parameters of input data
WIDTH = 128
HEIGHT = 128
TOP_PC = 750

DEBUG = True

save_path = '/home/jkuettel/latent-linear-adjustment-autoencoders/python/Ridge_model/data/eval_dez_2022'

basedir = "/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/preprocessed_data/MIX/prec_psl_train_mixed11_holdout_all_reanalysis_from_1955_date_2022_06_09__19_22_id_FNDBN7132U_detrend_TRUE_DEBUG_FALSE/rda_data"


tr_file = "train_FNDBN7132U.rda"
test_file = "test_FNDBN7132U.rda"

#basedir = "/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/preprocessed_data/CORDEX/can_train_39_cordex_holdout_year_all_months_1_2_12_npc_psl1000_mean_ens_temp_psl_detrendTRUE_scale_TRUE_test_split_0.8_temp_disjoint"
#tr_file = "train_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs.rda"
#test_file = "test_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs.rda"

''' ho_names = [
    "KNMI_IPSL-IPSL-CM5A-MR_KNMI-RACMO22E_r1i1p1",
    "KNMI_MPI-M-MPI-ESM-LR_KNMI-RACMO22E_r1i1p1",
    "CLMcom_MPI-M-MPI-ESM-LR_CLMcom-CCLM4-8-17_r1i1p1",
    "DMI_ICHEC-EC-EARTH_DMI-HIRHAM5_r1i1p1",
    "IPSL_ICHEC-EC-EARTH_IPSL-WRF381P_r12i1p1"
] '''


ho_names = ['ERA5_1955_2020_holdout_FNDBN7132U.rda', 'ERA5_modern_1979_2020_holdout_FNDBN7132U.rda']

alphas = [1e-6, 1e-5,1e-4,1e-3,1e-2]

#load train / test data
Y_tr, X_tr = prec_and_psl_to_python(os.path.join(basedir, tr_file),'tr')
Y_te, X_te = prec_and_psl_to_python(os.path.join(basedir, test_file), 'te')

#load holdout data
dir_ls = os.listdir(basedir)

ho_list = []

import pdb
for ho_name in ho_names:

    ho_paths = [p for p in dir_ls if ho_name in p]
    if len(ho_paths) != 1:
        print('more than 1 holdout found')
        print('ho :', ho_name, ' list: ', ho_paths)
        raise ValueError('bo many holdouts')

    ho_path = os.path.join(basedir, ho_paths[0])
    pr, psl = prec_and_psl_to_python(ho_path, 'ho')
    


    ho_list.append(holdout(ho_name, pr, psl, pr.shape[0], HEIGHT, WIDTH))


print("finished loading data")

if DEBUG:
    loc_touples =  [(i,j) for i in range(3) for j in range(3)]
else:
    loc_touples =  [(i,j) for i in range(WIDTH) for j in range(HEIGHT)]




def get_results(loc):
    global ho_list

    target_tr = Y_tr[:,loc[0],loc[1]]
    target_te = Y_te[:,loc[0],loc[1]]

    clf = fit_ridge(X_tr, X_te, target_tr, target_te, alphas = alphas)

    for ho in ho_list:

        x_hat_exp = clf.predict(ho.psl)
        x_hat_exp_orig = clf.predict(ho.psl) **2

        print('psl shape: ', ho.psl.shape)
        print('x_hat_exp shape: ', x_hat_exp.shape)
        print('x_hat_exp: ', x_hat_exp)

        ho.xhat_exp[:,loc[0], loc[1]] = x_hat_exp
        ho.xhat_exp_orig[:,loc[0], loc[1]] = x_hat_exp_orig

        print('ho.x_hat_exp: ', ho.xhat_exp_orig[:,loc[0], loc[1]])
        
        if ho.pr.shape[1] == WIDTH and ho.pr.shape[2] == HEIGHT:
            
            ho.sqrt_score[loc[0], loc[1]] = compute_r2(ho.pr[:,loc[0],loc[1]], clf.predict(ho.psl))
            ho.orig_score[loc[0], loc[1]] = compute_r2((ho.pr[:,loc[0],loc[1]])**2 , (clf.predict(ho.psl)) **2)


def main():
    

    queue = Queue()

    processes = [Process(target=get_results, args=(loc,  ))
                for loc in loc_touples]

    for i,p in enumerate(processes):
        p.start()
        if i%100 == 0:
            print(i)

    for p in processes:
        p.join()
  
    pdb.set_trace()

    for ho in ho_list:
        save_path_ho = os.path.join(save_path, ho.name[:-4])
        if not os.path.exists(save_path_ho):
            os.makedirs(save_path_ho)

        np.save(os.path.join(save_path_ho, 'xhat_exp_{}_sqrt.npy'.format(ho.name[:-4])), ho.xhat_exp)
        np.save(os.path.join(save_path_ho, 'xhat_exp_{}_orig.npy'.format(ho.name[:-4])), ho.xhat_exp_orig)
        np.save(os.path.join(save_path_ho, 'r2_{}_sqrt.npy'.format(ho.name[:-4])), ho.sqrt_score)
        np.save(os.path.join(save_path_ho, 'r2_{}_orig.npy'.format(ho.name[:-4])), ho.orig_score)
   

if __name__ == '__main__':
    main()
