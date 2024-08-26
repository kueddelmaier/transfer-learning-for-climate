#!/bin/bash
## PROCESS HOURLY RCM DATA into daily files:
# -------------------------------------

# Joel Küttel
# 23.03.2021



#####################################################################################


module load cdo
module load nco
module load ncl


#flehner_dir = /net/meso/climphys/flehner/ClimEx/CanESM2_driven_50_members/
base_dir=/net/atmos/data/cordex/EUR-11
#target_dir=/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/1deg/knmi
target_dir=/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/regridded_cordex_data

testgrid=/net/h2o/climphys1/sippels/_DATASET/CanESM2_LE_reg/orig_grid.txt

cristina_dir_psl=/net/h2o/climphys1/heinzec/sds/psl_nc_data
christina_dir_pr=/net/h2o/climphys1/heinzec/sds/pr_nc_data
cd ${target_dir}

cordex_member="DMI"
ensembles=("CNRM-CERFACS-CNRM-CM5" "ICHEC-EC-EARTH" "IPSL-IPSL-CM5A-MR" "MOHC-HadGEM2-ES" "MPI-M-MPI-ESM-LR" "NCC-NorESM1-M")
regional_models=("DMI-HIRHAM5")
runs=("r1i1p1" "r2i1p1" "r3i1p1" "r12i1p1")
epochs=("historical" "rcp85")
## Define grid for definition and regridding:
cdo -griddes ${cristina_dir_psl}/psl_EUR-11_CCCma-CanESM2_historical_OURANOS-CRCM5_kba_1d_ALLYEARS_1d00_1_2_12.nc > psl_grid.txt
cdo -griddes ${christina_dir_pr}/pr_EUR-11_CCCma-CanESM2_historical_OURANOS-CRCM5_kba_1d_ALLYEARS_ALPS_1_2_12.nc > pr_grid.txt

mkdir ${cordex_member}
mkdir ${cordex_member}/boxcuts
mkdir ${cordex_member}/boxcuts/pr
mkdir ${cordex_member}/boxcuts/psl
mkdir ${cordex_member}/pr
mkdir ${cordex_member}/psl 






for ens in "${ensembles[@]}"
    do
    mkdir ${cordex_member}/boxcuts/pr/${ens}
    mkdir ${cordex_member}/boxcuts/psl/${ens}
    mkdir ${cordex_member}/pr/${ens}
    mkdir ${cordex_member}/psl/${ens}

    echo $ens
    for regional_model in "${regional_models[@]}"
        do
        mkdir ${cordex_member}/boxcuts/pr/${ens}/${regional_model}
        mkdir ${cordex_member}/boxcuts/psl/${ens}/${regional_model}
        mkdir ${cordex_member}/pr/${ens}/${regional_model}
        mkdir ${cordex_member}/psl/${ens}/${regional_model}

        for run in "${runs[@]}"
            do
            echo ${base_dir}/rcp85/day/pr/${cordex_member}/${ens}/${regional_model}/${run}
            if [ -d ${base_dir}/rcp85/day/pr/${cordex_member}/${ens}/${regional_model}/${run} ] && [ -d ${base_dir}/rcp85/day/psl/${cordex_member}/${ens}/${regional_model}/${run} ] && \
            [ -d ${base_dir}/historical/day/pr/${cordex_member}/${ens}/${regional_model}/${run} ] && [ -d ${base_dir}/historical/day/psl/${cordex_member}/${ens}/${regional_model}/${run} ]; then
                echo ${run} "of regional model" ${regional_model} "of ensemble" ${ens}
                mkdir ${cordex_member}/boxcuts/pr/${ens}/${regional_model}/${run}
                mkdir ${cordex_member}/boxcuts/psl/${ens}/${regional_model}/${run}
                mkdir ${cordex_member}/pr/${ens}/${regional_model}/${run}
                mkdir ${cordex_member}/psl/${ens}/${regional_model}/${run}

                #start with pr
                for epoch in "${epochs[@]}"
                    do
                    for d in ${base_dir}/${epoch}/day/pr/${cordex_member}/${ens}/${regional_model}/${run}/*_v1_*; do
                        cdo -sellonlatbox,0,18.9,42,54.8 ${d} ${cordex_member}/boxcuts/pr/${ens}/${regional_model}/${run}/ALPS_${d##*/}
                        done
                    done
                #merge all the boxcuts in time    
                cdo -mergetime ${cordex_member}/boxcuts/pr/${ens}/${regional_model}/${run}/*  ${cordex_member}/pr/${ens}/${regional_model}/${run}/time_merged.nc
                
                #select years from 1955 to 2100
                cdo -seldate,1955-01-01T00:30:00,2099-12-31T23:30:00 ${cordex_member}/pr/${ens}/${regional_model}/${run}/time_merged.nc ${cordex_member}/pr/${ens}/${regional_model}/${run}/1955_2100.nc

                #select only winter moths
                cdo -selmon,12,1,2 ${cordex_member}/pr/${ens}/${regional_model}/${run}/1955_2100.nc ${cordex_member}/pr/${ens}/${regional_model}/${run}/1955_2100_winter.nc 

                #remove the leap days
                cdo delete,month=2,day=29 ${cordex_member}/pr/${ens}/${regional_model}/${run}/1955_2100_winter.nc ${cordex_member}/pr/pr_EUR-11_${cordex_member}_${ens}_${regional_model}_${run}_day_ALPS_historic_and_rcp85.nc 
                
                rm ${cordex_member}/pr/${ens}/${regional_model}/${run}/time_merged.nc ${cordex_member}/pr/${ens}/${regional_model}/${run}/1955_2100.nc ${cordex_member}/pr/${ens}/${regional_model}/${run}/1955_2100_winter.nc 
                echo ${cordex_member}/pr/pr_EUR-11_${cordex_member}_${ens}_${regional_model}_${run}_day_ALPS_historic_and_rcp85.nc   
                echo cdo -ntime ${cordex_member}/pr/pr_EUR-11_${cordex_member}_${ens}_${regional_model}_${run}_day_ALPS_historic_and_rcp85.nc 


                #now psl

                for epoch in "${epochs[@]}"
                    do
                    for d in ${base_dir}/${epoch}/day/psl/${cordex_member}/${ens}/${regional_model}/${run}/*; do
                        cdo -remapbil,psl_grid.txt ${d} ${cordex_member}/boxcuts/psl/${ens}/${regional_model}/${run}/EUROPE_${d##*/}
                        done
                    done
                #merge all the boxcuts in time    
                cdo -mergetime ${cordex_member}/boxcuts/psl/${ens}/${regional_model}/${run}/*  ${cordex_member}/psl/${ens}/${regional_model}/${run}/time_merged.nc
                
                #select years from 1955 to 2100
                cdo -seldate,1955-01-01T00:30:00,2099-12-31T23:30:00 ${cordex_member}/psl/${ens}/${regional_model}/${run}/time_merged.nc ${cordex_member}/psl/${ens}/${regional_model}/${run}/1955_2100.nc

                #select only winter moths
                cdo -selmon,12,1,2 ${cordex_member}/psl/${ens}/${regional_model}/${run}/1955_2100.nc ${cordex_member}/psl/${ens}/${regional_model}/${run}/1955_2100_winter.nc 

                #remove the leap days
                cdo delete,month=2,day=29 ${cordex_member}/psl/${ens}/${regional_model}/${run}/1955_2100_winter.nc ${cordex_member}/psl/psl_EUR-11_${cordex_member}_${ens}_${regional_model}_${run}_day_EUROPE_historic_and_rcp85.nc  
                
                rm ${cordex_member}/psl/${ens}/${regional_model}/${run}/time_merged.nc ${cordex_member}/psl/${ens}/${regional_model}/${run}/1955_2100.nc  ${cordex_member}/psl/${ens}/${regional_model}/${run}/1955_2100_winter.nc 
                fi
            done
        done
    done