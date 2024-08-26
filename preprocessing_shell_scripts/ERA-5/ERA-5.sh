#!/bin/bash
## PROCESS HOURLY RCM DATA into daily files:
# -------------------------------------

# Joel Küttel
# 23.03.2021



#####################################################################################


module load cdo
module load nco
module load ncl



base_dir=/net/xenon/climphys/lukbrunn/data/ERA5/

target_dir=/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/regridded_data/ERA5

testgrid=/net/h2o/climphys1/sippels/_DATASET/CanESM2_LE_reg/orig_grid.txt

source_dir_psl=/net/h2o/climphys1/heinzec/sds/psl_nc_data
source_dir_pr=/net/h2o/climphys1/heinzec/sds/pr_nc_data
cd ${target_dir}


## Define grid for definition and regridding:
cdo -griddes ${source_dir_psl}/psl_EUR-11_CCCma-CanESM2_historical_OURANOS-CRCM5_kba_1d_ALLYEARS_1d00_1_2_12.nc > psl_grid.txt
cdo -griddes ${source_dir_pr}/pr_EUR-11_CCCma-CanESM2_historical_OURANOS-CRCM5_kba_1d_ALLYEARS_ALPS_1_2_12.nc > pr_grid.txt


mkdir ${target_dir}/boxcuts
mkdir ${target_dir}/boxcuts/pr
mkdir ${target_dir}/boxcuts/psl

mkdir ${target_dir}/pr
mkdir ${target_dir}/psl 







for d in ${base_dir}/total_precipitation/day/native/*; do
    cdo -sellonlatbox,-5,20,40.5,57 ${d} ${target_dir}/boxcuts/pr/ALPS_${d##*/}
    done

#merge all the boxcuts in time    
cdo -mergetime ${target_dir}/boxcuts/pr/*  ${target_dir}/pr/time_merged.nc

#select only winter moths
cdo -selmon,12,1,2 ${target_dir}/pr/time_merged.nc ${target_dir}/pr/1955_2100_winter.nc 

#remove the leap days
cdo delete,month=2,day=29 ${target_dir}/pr/1955_2100_winter.nc  ${target_dir}/pr/pr_EUR-11_ERA-5_1997_2020_day_ALPS_historic_and_rcp85.nc 

rm ${target_dir}/pr/time_merged.nc ${target_dir}/pr/1955_2100_winter.nc 

echo cdo -ntime ${target_dir}/pr/pr_EUR-11_ERA-5_1997_2020_day_ALPS_historic_and_rcp85.nc 


#now psl


for d in ${base_dir}/mean_sea_level_pressure/day/native/*; do
    cdo -remapbil,psl_grid.txt ${d} ${target_dir}/boxcuts/psl/EUROPE_${d##*/}
    done

#merge all the boxcuts in time    
cdo -b F64 mergetime ${target_dir}/boxcuts/psl/*  ${target_dir}/psl/time_merged.nc

#select only winter moths
cdo -selmon,12,1,2 ${target_dir}/psl/time_merged.nc ${target_dir}/psl/1955_2100_winter.nc 

#remove the leap days
cdo delete,month=2,day=29 ${target_dir}/psl/1955_2100_winter.nc  ${target_dir}/psl/psl_EUR-11_ERA-5_1997_2020_day_ALPS_historic_and_rcp85.nc 

rm ${target_dir}/psl/time_merged.nc ${target_dir}/psl/1955_2100_winter.nc 

echo cdo -ntime ${target_dir}/psl/psl_EUR-11_ERA-5_1997_2020_day_ALPS_historic_and_rcp85.nc 
