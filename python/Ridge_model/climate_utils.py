import xarray as xr
import rpy2.robjects as robjects
import numpy as np
import matplotlib.pyplot as plt

class holdout():

    def __init__(self, name, pr,psl, time, height, width):
        self.name = name
        self.pr = pr
        self.psl = psl
        self.xhat_exp = np.zeros((time, height, width))
        self.xhat_exp_orig = np.zeros((time, height, width))
        self.sqrt_score = np.zeros((height, width))
        self.orig_score = np.zeros((height, width))



    def append_score(self, score):
        self.score_list.append(score)
    
    def calculate_mean(self):
        return np.mean(self.score_array)


def prec_to_python(path, sub):
    robjects.r['load'](path)
    images = np.array(robjects.r['prec_mat_sqrt_'+sub])
    
    pr_array = np.zeros((images.shape[2],images.shape[0],images.shape[1]))

    for i in range(images.shape[2]):
        pr_array[i,:,:] = images[:,:,i]
    
    return np.flip(pr_array, axis = 1)


def psl_to_python(path, sub):
    robjects.r['load'](path)
    psl_array = np.array(robjects.r['psl_Z_'+sub])

    return psl_array

def compute_r2(target, pred):
    ''' computes R2 given target time series and predicted time series
         in other words, computes R2 at the grid point level'''
    residual_ss = np.sum((target-pred)**2)
    colmeans_target = np.mean(target, axis=0)
    total_ss = np.sum((target-colmeans_target)**2)
    r2 = 1-residual_ss/total_ss  
    return r2


def prec_and_psl_to_python(path, sub):

    robjects.r['load'](path)
    images = np.array(robjects.r['prec_mat_sqrt_'+sub])
    pr_array = np.zeros((images.shape[2],images.shape[0],images.shape[1]))

    for i in range(images.shape[2]):
        pr_array[i,:,:] = images[:,:,i]

    psl_array = np.array(robjects.r['psl_Z_'+sub])

    return np.flip(pr_array, axis = 1), psl_array








