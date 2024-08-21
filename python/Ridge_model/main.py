
import xarray as xr
import os
from climate_utils import *
from models import fit_ridge

from multiprocessing import Process, Queue


WIDTH = 128
HEIGHT = 128

basedir = "/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/preprocessed_data/CORDEX/can_train_39_cordex_holdout_year_all_months_1_2_12_npc_psl1000_mean_ens_temp_psl_detrendTRUE_scale_TRUE_test_split_0.8_temp_disjoint"
tr_file = "train_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs.rda"
test_file = "test_prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs.rda"
ho_file = "prec_psl_ens_kba_kbc_kbe_kbg_kbi_kbk_kbm_kbq_kbs_holdout_KNMI_IPSL-IPSL-CM5A-MR_KNMI-RACMO22E_r1i1p1.rda"

alphas = [1e-6, 1e-5,1e-4,1e-3,1e-2]
score = np.zeros((HEIGHT, WIDTH))


X_tr = psl_to_python(os.path.join(basedir, tr_file),'tr')
X_te = psl_to_python(os.path.join(basedir, test_file), 'te')
Y_tr = prec_to_python(os.path.join(basedir, tr_file), 'tr')
Y_te = prec_to_python(os.path.join(basedir, test_file), 'te')

Y_ho = prec_to_python(os.path.join(basedir, ho_file), 'ho')
X_ho = psl_to_python(os.path.join(basedir, ho_file), 'ho')

print("finished loading data")

def get_results(loc):

    target_tr = Y_tr[:,loc[0],loc[1]]
    target_te = Y_te[:,loc[0],loc[1]]

    clf = fit_ridge(X_tr, X_te, target_tr, target_te, alphas = alphas)

    score[loc[0], loc[1]] = compute_r2(Y_ho[:,loc[0],loc[1]], clf.predict(X_ho))
        


def main():

  
    loc_touples =  [(i,j) for i in range(WIDTH) for j in range(HEIGHT)]

    queue = Queue()

    processes = [Process(target=get_results, args=(loc, ))
                for loc in loc_touples]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
 
    np.save('/home/jkuettel/latent-linear-adjustment-autoencoders/Ridge_model/test.npy', score)

if __name__ == '__main__':
    main()