#!/bin/bash
## PROCESS HOURLY RCM DATA into daily files:
# -------------------------------------

# Joel Küttel
# 23.03.2021



#####################################################################################


module load cdo
module load nco
module load ncl



base_dir=/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/E-OBS-24.oe/0.25_deg/

target_dir=/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/regridded_data/E-OBS/0.25_deg

testgrid=/net/h2o/climphys1/sippels/_DATASET/CanESM2_LE_reg/orig_grid.txt

cristina_dir_psl=/net/h2o/climphys1/heinzec/sds/psl_nc_data
christina_dir_pr=/net/h2o/climphys1/heinzec/sds/pr_nc_data
cd ${target_dir}


## Define grid for definition and regridding:
cdo -griddes ${cristina_dir_psl}/psl_EUR-11_CCCma-CanESM2_historical_OURANOS-CRCM5_kba_1d_ALLYEARS_1d00_1_2_12.nc > psl_grid.txt
cdo -griddes ${christina_dir_pr}/pr_EUR-11_CCCma-CanESM2_historical_OURANOS-CRCM5_kba_1d_ALLYEARS_ALPS_1_2_12.nc > pr_grid.txt




mkdir ${target_dir}/pr
mkdir ${target_dir}/psl 

cdo -sellonlatbox,-5,20,40.5,57 ${base_dir}/rr/rr_ens_mean_0.25deg_reg_v24.0e.nc ${target_dir}/pr/ALPS_eobs.nc

#select only winter moths
cdo -selmon,12,1,2 ${target_dir}/pr/ALPS_eobs.nc ${target_dir}/pr/ALPS_eobs_winter.nc

#remove the leap days
cdo delete,month=2,day=29 ${target_dir}/pr/ALPS_eobs_winter.nc  ${target_dir}/pr/pr_EUR-25_E-OBS_1950_2021_day_ALPS.nc 

rm ${target_dir}/pr/ALPS_eobs.nc ${target_dir}/pr/ALPS_eobs_winter.nc 

echo cdo -ntime ${target_dir}/pr/pr_EUR-25_E-OBS_1950_2021_day_ALPS.nc 



