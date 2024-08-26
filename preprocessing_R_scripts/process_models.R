# Extends diff_ens_mean by not looping over hodlouts and training, but first the training and then the holdouts, making the code much more understandeable
#So we don't need a holdoutindex, a training index and a i

rm(list = ls())
options(warn=1)
library(raster)
library(ncdf4)
library(ncdf4.helpers)
library(glmnet)
library(bigmemory)
library(jsonlite)
library(glue)

require(abind)
require(foreach)
require(doParallel)
registerDoParallel(cores = 42)

DEBUG <- FALSE
#specifiy name to identify the directory
directory_name = "train_mixed11_holdout_all_reanalysis_from_1955"

if (DEBUG){
  cat("\n########################")
  cat("\n####### ON DEBUG #######")
  cat("\n########################\n")
}

source("/home/jkuettel/latent-linear-adjustment-autoencoders/preprocessing_R_scripts/better_diff_ens_mean/helpers_sds_multiple_lenght_diff_ens_mean.R")

training_ens <- c(
    "CanESM2_kba",
    "GERICS_MPI-M-MPI-ESM-LR_GERICS-REMO2015_r3i1p1",
    "CLMcom_ICHEC-EC-EARTH_CLMcom-CCLM4-8-17_r12i1p1",
    "CLMcom-ETH_NCC-NorESM1-M_CLMcom-ETH-COSMO-crCLIM-v1-1_r1i1p1",
    "DMI_IPSL-IPSL-CM5A-MR_DMI-HIRHAM5_r1i1p1",
    "GERICS_CNRM-CERFACS-CNRM-CM5_GERICS-REMO2015_r1i1p1",
    "IPSL_ICHEC-EC-EARTH_IPSL-WRF381P_r12i1p1",
    "KNMI_IPSL-IPSL-CM5A-MR_KNMI-RACMO22E_r1i1p1",
    "MOHC_NCC-NorESM1-M_MOHC-HadREM3-GA7-05_r1i1p1",
    "MPI-CSC_MPI-M-MPI-ESM-LR_MPI-CSC-REMO2009_r1i1p1",
    "SMHI_CNRM-CERFACS-CNRM-CM5_SMHI-RCA4_r1i1p1",
    "CanESM2_kbs"

)


holdout_ens  <- c("E-OBS_CORDEX11_ERA5_1950_2020_15y_test_psl_nans_replaced_train",
                  "E-OBS_CORDEX11_ERA5_1950_2020_15y_test_psl_nans_replaced_test",
                  "E-OBS_CORDEX11_ERA5_1950_2020_20y_test_psl_nans_replaced_train",
                  "E-OBS_CORDEX11_ERA5_1950_2020_20y_test_psl_nans_replaced_test")

# holdout_ens  <- 
#   c("CanESM2_kbj",
#     "CanESM2_kbk",
#     "CanESM2_kco",
#     "CanESM2_kcs",
#     "CanESM2_kbp",
#     "CanESM2_kbb",
#     "CanESM2_kbd",
#     "CanESM2_kbn",
#     "CanESM2_kcm",
#     "CanESM2_kcr",
#     "CanESM2_kcp",
#     "CanESM2_kbx",
#     "CanESM2_kbc",
#     "CanESM2_kcc",
#     "CanESM2_kba",
#     "CanESM2_kch",
#     "CanESM2_kcg",
#     "CanESM2_kcn",
#     "CanESM2_kbi",
#     "CanESM2_kbq",
#     "CanESM2_kbv",
#     "CanESM2_kbo",
#     "CanESM2_kbh",
#     "CanESM2_kbm",
#     "CanESM2_kbt",
#     "CanESM2_kcl",
#     "CanESM2_kcd",
#     "CanESM2_kbw",
#     "CanESM2_kce",
#     "CanESM2_kck",
#     "CanESM2_kbf",
#     "CanESM2_kbr",
#     "CanESM2_kcu",
#     "CanESM2_kcb",
#     "CanESM2_kcx",
#     "CanESM2_kcj",
#     "CanESM2_kci",
#     "CanESM2_kbl",
#     "CanESM2_kbz",
#     "CanESM2_kct",
#     "CanESM2_kbg",
#     "CanESM2_kca",
#     "CanESM2_kbu",
#     "CanESM2_kby",
#     "CanESM2_kcq",
#     "CanESM2_kcf",
#     "CanESM2_kcw",
#     "CanESM2_kbe",
#     "CanESM2_kbs",
#     "CanESM2_kcv")




file_info_models <- read.csv(file = '/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/regridded_data/net_cdf_info/file_info_models.csv', header = TRUE)
df_file_info_models <- data.frame(file_info_models)

file_info_reanalysis <- read.csv(file = '/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/regridded_data/net_cdf_info/file_info_reanalysis.csv', header = TRUE)
df_file_info_reanalysis <- data.frame(file_info_reanalysis)

detrend_id <- create_id(1)

cat("\n Started processing, Detrending id: ", detrend_id, "\n")
#for each of the specified ensembles, get the path

year <- "all"
months <- "1_2_12"

# parameters
n_CORDEX <- 11500
n_SMHI <- 10500

cut_holdout <- FALSE
n_HOLDOUT <- 11000

scale_train <- FALSE
scale_holdout <- FALSE
scale_factor <- 100000

## number of principal component directions to extract
n_pc_psl <- 1000
## fraction of data to use for training
n_train_fraction <- .8 
## number of data points to use for training
n_train_obs <- 3956
## flag whether to split by n_train_fraction or n_train_obs
by_frac <- TRUE
## flag whether to scale data
do_scale <- TRUE
## flag whether to detrend
detrend <- FALSE

data_dir <- "/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data"

## flag if only use land data and mask for e-obs use
mask_to_eobs <- TRUE

#flag if slp should be masked to the size of canesm slp (smaller)
mask_slp_to_canesm <- FALSE


### Spline Parameters ###

spline_dof <- 3
spline_lambda <- 0.4

#reanalysis_detrend_file <- 


#get psl index for non_zero values:

psl_index_path <- "/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/grids/slp_indices/spatial_slp_indices_canesm.txt"
psl_index <- read.table(file = psl_index_path, header = FALSE, sep = ",")
psl_spatial_index_values <- as.integer(as.numeric(unlist(psl_index)))


## specify path and name to save preprocessed data
save_directory <- "/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/preprocessed_data/MIX"


save_name <- paste("prec_psl_", 
                   directory_name, 
                   "_date_", format(Sys.time(), "%Y_%m_%d__%H_%M") ,
                   "_id_", detrend_id, 
                   "_detrend_", detrend,
                   "_DEBUG_", DEBUG,
                   sep = "")

cat("### Directory: ", save_name, " ###")

save_path <- file.path(save_directory,save_name)
dir.create(save_path)

rda_data_path <- file.path(save_path, 'rda_data')
dir.create(rda_data_path)

# pressure




#need time series in the same format as the data (netcdf package), so we load it again
time_stamps_path <- "/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/ens_means/multi_model_ens_mean/psl_ensmean_with_smhi_1955_2083/psl_ensmean_with_smhi_1955_2083.nc"

time_stamps_nc_file <- nc_open(time_stamps_path)
time_stamps_daily_corresp_to_ens_mean <- nc.get.time.series(time_stamps_nc_file, 
                                                            time.dim.name = "time")


#### Load all Ens Means ###

ens_mean_names <- c(
  "CanESM2",
  "GERICS",
  "CLMcom",
  "CLMcom-ETH",
  "DMI",
  "IPSL",
  "KNMI",
  "MOHC",
  "MPI-CSC",
  "SMHI",
  "Multi_Model"
)


ens_mean_list <- vector("list", length(ens_mean_names))

# Load all the ens means to detrend in the ens_mean_list
for (i in seq_along(ens_mean_names)){
  cat("\n loading ens mean of :", ens_mean_names[i])
  
  if (ens_mean_names[i] == "CanESM2"){
    ## for detrending of EOF time series: load SLP ensemble mean
    psl_ensmean_path <- "/net/h2o/climphys1/sippels/_DATASET/CanESM2_LE_reg/psl/psl_EUR-11_CCCma-CanESM2_historical_rcp85_OURANOS-CRCM5_1d_ENSMEAN_1d00.nc"
    psl = brick(psl_ensmean_path) + 0
    
    ## select only januaries and non-NA data points
    psl.jan = subset(psl, seq(1, 1740, 12))  
    
  }
  
  else if (ens_mean_names[i] == "Multi_Model"){
    ## for detrending of EOF time series: load SLP ensemble meann_train <- ceiling(n_train_fraction*n)
    #psl_ensmean_path <- "/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/ens_means/multi_model_ens_mean/psl_ensmean_with_smhi_1955_2083/psl_ensmean_with_smhi_1955_2083_mon_mean.nc"
    psl_ensmean_path <- "/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/ens_means/multi_model_ens_mean/ensmean_psl_13050_mon_mean.nc"
    psl = brick(psl_ensmean_path) + 0
    
    ## select only januaries and non-NA data points
    psl.jan = subset(psl, seq(1, dim(psl)[3], 3))  
    
  }
  else {
    
    base_path <- "/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/regridded_data/CORDEX"
    psl_ensmean_path <- file.path(base_path, ens_mean_names[i], "psl", glue("mon_mean_psl_ensmean_{ens_mean_names[i]}.nc"))
    psl = brick(psl_ensmean_path) + 0

    psl.jan = subset(psl, seq(1, dim(psl)[3], 3))
  }
  
  psl.jan.values = (values(psl.jan))
  na.ix = unique(which(is.na(psl.jan.values), arr.ind=T)[,1])
  psl.jan.values = psl.jan.values[-na.ix,]
  
  raster.area = values(raster::area(psl.jan))[-na.ix]
  psl.jan.values.scale = (psl.jan.values - rowMeans(psl.jan.values)) * raster.area / sum(raster.area)
  psl.jan.values.scale.SVD = svd(psl.jan.values.scale)
  psl_v1 <- psl.jan.values.scale.SVD$v[,1] * (-1)
  
  inp <- 1:length(psl_v1)
  # fit_psl_v1 <- lm(psl_v1~inp)
  fit_psl_v1 <- smooth.spline(x=inp, y=psl_v1, df = spline_dof, lambda = spline_lambda) 
  fitted_psl_v1 <- fitted(fit_psl_v1)
  if(do_scale){
    fitted_psl_v1 <- scale(fitted_psl_v1)
  }
  ## bring to daily resolution 
  fitted_psl_v1_add_to_daily <- rep(fitted_psl_v1, each=90)
  
  ens_mean_list[[i]] <- fitted_psl_v1_add_to_daily
  
}
#create an named list, so can acces first pc of ens mean by name of ens
ens_mean_list_named <- setNames(ens_mean_list, ens_mean_names)


## init list for training, test and holdout data
psl_mat_list_tr <- vector("list", length(training_ens))
psl_mat_list_te <- vector("list", length(training_ens))
psl_mat_list_ho <- vector("list", length(holdout_ens))

train_slp_detrend_list <- vector("list", length(training_ens))
test_slp_detrend_list <- vector("list", length(training_ens))
holdout_slp_detrend_list <- vector("list", length(holdout_ens))



## load data
for(i in seq_along(training_ens)){
  
  cat('\n processing: ', training_ens[i])
  
  
  
  if(training_ens[i] %in% df_file_info_models$Identifier){
    cat('\n got df file info')
    ens_name <- df_file_info_models[which(df_file_info_models$Identifier == training_ens[i]), "Ensemble"]
    psl_nc_path <- df_file_info_models[which(df_file_info_models$Identifier == training_ens[i]), "psl_path"]
    var_psl <- "psl"}
  
  else if(training_ens[i] %in% df_file_info_reanalysis$Identifier){
    cat('\n got df reanalysis')
    ens_name <- "Multi_Model"
    psl_nc_path <- df_file_info_reanalysis[which(df_file_info_reanalysis$Identifier == training_ens[i]), "psl_path"]
    var_psl <- df_file_info_reanalysis[which(df_file_info_reanalysis$Identifier == training_ens[i]), "psl_variable_name"]
    
  }
  
  
  else {
    stop("\n Filepath of: ", training_ens[i], " not found")
  }
  
  
  #psl <- brick(psl_nc_path) + 0
  #psl_mat <- as.array(psl)
  cat('\n psl path: ',  psl_nc_path, " var: ", var_psl)
  
  psl_nc <- nc_open(file.path(data_dir,psl_nc_path))
  psl_arr_r <- ncvar_get(psl_nc, var_psl)
  
  psl_mat <- array(0, dim=c(dim(psl_arr_r)[2],dim(psl_arr_r)[1],dim(psl_arr_r)[3]))
  
  for (k in 1:dim(psl_arr_r)[3]){
    psl_mat[,,k] = apply(t(psl_arr_r[,,k]),2,rev)
    
  }
  
  #mask to cansem
  
  if (mask_slp_to_canesm){
    canesm_mask_path <- load("/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/grids/masks/R_masks/canesm_slp_mask.gzip")
    
    for (t in 1:dim(psl_mat)[3]){
      psl_mat[,,t] = canesm_slp_mask * psl_mat[,,t] 
      
    }
  }
  
  
  if(ens_name == "SMHI"){
    n <- n_SMHI
  }
  else {
    n <- n_CORDEX
  }
  
  
  n_train <- ceiling(n_train_fraction*n)
  cat("\n n_train psl: ", n_train)
  
  
  # process training ensemble members
  
  ## split into training and test 
  
  
  if(by_frac){
    n_train <- ceiling(n_train_fraction*n)
  }else{
    n_train <- n_train_obs
  }
  psl_mat_list_tr[[i]] <- psl_mat[,,1:n_train]
  psl_mat_list_te[[i]] <- psl_mat[,,(n_train+1):n]
  
  cat("\n dim psl train: ", dim(psl_mat[,,1:n_train]))
  cat("\n dim psl test: ",  dim(psl_mat[,,(n_train+1):n]))
  train_slp_detrend_list[[i]] <- ens_mean_list_named[[ens_name]][1:n_train]
  test_slp_detrend_list[[i]] <- ens_mean_list_named[[ens_name]][(n_train+1):n]
  
  cat("\n")
}



for(i in seq_along(holdout_ens)){
  cat('\n processing: ', holdout_ens[i])
  
  if(holdout_ens[i] %in% df_file_info_models$Identifier){
    cat('\n got df file info')
    ens_name <- df_file_info_models[which(df_file_info_models$Identifier == holdout_ens[i]), "Ensemble"]
    psl_nc_path <- df_file_info_models[which(df_file_info_models$Identifier == holdout_ens[i]), "psl_path"]
    var_psl <- "psl"}
  
  else if(holdout_ens[i] %in% df_file_info_reanalysis$Identifier){
    cat('\n got df reanalysis')
    ens_name <- "Multi_Model"
    psl_nc_path <- df_file_info_reanalysis[which(df_file_info_reanalysis$Identifier == holdout_ens[i]), "psl_path"]
    var_psl <- df_file_info_reanalysis[which(df_file_info_reanalysis$Identifier == holdout_ens[i]), "psl_variable_name"]
    
  }
  
  
  else {
    stop("\n Filepath of: ", holdout_ens[i], " not found")
  }
  
  #psl <- brick(psl_nc_path) + 0
  #psl_mat <- as.array(psl)
  cat('\n psl path: ',  file.path(data_dir,psl_nc_path), " var: ", var_psl)
  
  psl_nc <- nc_open(file.path(data_dir,psl_nc_path))
  psl_arr_r <- ncvar_get(psl_nc, var_psl)
  
  psl_mat <- array(0, dim=c(dim(psl_arr_r)[2],dim(psl_arr_r)[1],dim(psl_arr_r)[3]))
  
  for (k in 1:dim(psl_arr_r)[3]){
    psl_mat[,,k] = apply(t(psl_arr_r[,,k]),2,rev)
  }
  
  #mask to cansem
  
  if (mask_slp_to_canesm){
    canesm_mask_path <- load("/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/grids/masks/R_masks/canesm_slp_mask.gzip")
    
    for (t in 1:dim(psl_mat)[3]){
      psl_mat[,,t] = canesm_slp_mask * psl_mat[,,t] 
      
    }
  }
  
  
  # process hold out ensemble members
  # process training ensemble members
  
  if (cut_holdout){
    psl_mat_list_ho[[i]] <- psl_mat[,,1:n_HOLDOUT]
  }
  
  else
  {
    psl_mat_list_ho[[i]] <- psl_mat
  }
  
  
  cat("dim psl holdout: ", dim(psl_mat_list_ho[[i]]))
  
  
  if (ens_name == "Multi_Model"){
    time_stamps_ho <- nc.get.time.series(psl_nc, 
                                         time.dim.name = "time")
    
    if (cut_holdout){
      time_stamps_ho <- time_stamps_ho[1:n_HOLDOUT]
    }
    
    else {
      time_stamps_ho <- time_stamps_ho
    }
    
    
    holdout_slp_detrend_list[[i]] <- ens_mean_list_named[[ens_name]][1:dim(psl_mat_list_ho[[i]])[3]]
    
    
    
    
  }
  
  else {
    
    if (cut_holdout){
      holdout_slp_detrend_list[[i]] <- ens_mean_list_named[[ens_name]][1:n_HOLDOUT]
    }
    else {
      holdout_slp_detrend_list[[i]] <- ens_mean_list_named[[ens_name]]
    }
    cat("\n dim detrend values: ", length(holdout_slp_detrend_list[[i]]))
    
    
  }
  cat("\n")
}

psl_mat_tr <- do.call(abind, psl_mat_list_tr)
psl_mat_te <- do.call(abind, psl_mat_list_te)

train_slp_detrend_values <- do.call(abind, train_slp_detrend_list)
test_slp_detrend_values <- do.call(abind, test_slp_detrend_list) 

if(dim(psl_mat_tr)[3] != dim(train_slp_detrend_values)){
  cat('dim of train: ', dim(psl_mat_tr), 'dim of detrend: ', dim(train_slp_detrend_values))
  stop("psl_mat_tr and train_slp_detrend_values have not same shape ")
}

if(dim(psl_mat_te)[3] != dim(test_slp_detrend_values)){
  cat('dim of test: ', dim(psl_mat_te), 'dim of detrend: ', dim(test_slp_detrend_values))
  stop("psl_mat_te and test_slp_detrend_values have not same shape ")
}

cat("\nDim total psl train:", dim(psl_mat_tr))
cat("\nDim total psl test:", dim(psl_mat_te))
cat("\nDim total psl detrend test:", dim(test_slp_detrend_values))


psl_pcs <- compute_pcs_ho_list(psl_mat_tr, psl_mat_te, psl_mat_list_ho, n_pc_psl, psl_spatial_index_values, train_slp_detrend_values, test_slp_detrend_values, holdout_slp_detrend_list,
                               n_train, length(training_ens), fitted_psl_v1_add_to_daily, detrend=detrend)



##############################################################################################################################
######################################################## NOW PRECIP #########################################################
##############################################################################################################################



## init list for training, test and holdout data
prec_mat_list_tr <- vector("list", length(training_ens))
prec_mat_list_te <- vector("list", length(training_ens))
prec_mat_list_ho <- vector("list", length(holdout_ens))


## load data
for(i in seq_along(training_ens)){
  
  # check whether file belongs to training ensemble members
  
  
  if(training_ens[i] %in% df_file_info_models$Identifier){
    prec_nc_path <- df_file_info_models[which(df_file_info_models$Identifier == training_ens[i]), "pr_path"]
    ens_name <- df_file_info_models[which(df_file_info_models$Identifier == training_ens[i]), "Ensemble"]
    var_pr <- 'pr'
    pr_format <- "mm/s"
  }
  
  else if(training_ens[i] %in% df_file_info_reanalysis$Identifier){
    prec_nc_path <- df_file_info_reanalysis[which(df_file_info_reanalysis$Identifier == training_ens[i]), "pr_path"]
    var_pr <- df_file_info_reanalysis[which(df_file_info_reanalysis$Identifier == training_ens[i]), "pr_variable_name"]
    pr_format <- df_file_info_reanalysis[which(df_file_info_reanalysis$Identifier == training_ens[i]), "pr_format"]
    
  }
  
  
  else {
    stop("\n Filepath of: ", training[i], " not found")
  }
  
  
  cat('\n pr path: ',  prec_nc_path, " var: ", var_pr)
  
  prec_nc <- nc_open(file.path(data_dir,prec_nc_path))
  
  prec_arr_r <- ncvar_get(prec_nc, var_pr)
  
  prec_arr <- array(0, dim=c(dim(prec_arr_r)[2],dim(prec_arr_r)[1],dim(prec_arr_r)[3]))
  
  
  for (k in 1:dim(prec_arr_r)[3]){
    prec_arr[,,k] = apply(t(prec_arr_r[,,k]),2,rev)
  }
  
  if (mask_to_eobs){
    eobs_mask_path <- load("/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/grids/masks/R_masks/eobs_mask_land.gzip")
    
    for (t in 1:dim(prec_arr)[3]){
      prec_arr[,,t] = eobs_mask * prec_arr[,,t] 
      
    }
  }
  
  if (pr_format == "mm/s"){
    prec_arr <- prec_arr * 3600 * 24 # convert from mm/s into mm/d
  }
  
  
  else if (pr_format == "m/day"){
    prec_arr <- prec_arr * 1000 # convert from m/d into mm/d
  }
  
  else if (pr_format == "mm/day"){
    prec_arr <- prec_arr  # convert from mm/d into mm/d
  }

  
  else{
    stop("raw pr_format", pr_format ," of ens ", training_ens[i], "is not known")
  }
  
  
  if (scale_train) {
    prec_arr <- prec_arr * scale_factor # convert from mm/s into mm/d
  }
  
  #prec_arr <- prec_arr * 3600 * 24 # convert from mm/s into mm/d
  ### apply sqrt transform
  prec_mat_sqrt <- sqrt(prec_arr)
  
  
  ## all negative pr values are now nans, replace them with zeros
  prec_mat_sqrt[is.na( prec_mat_sqrt)] <- 0
  
  
  if(ens_name == "SMHI"){
    n <- n_SMHI
  }
  else {
    n <- n_CORDEX
  }
  
  
  n_train <- ceiling(n_train_fraction*n)
  cat("\n n_train pr: ", n_train)
  
  
  
  
  # process training ensemble members
  cat("\nProcessing training training_ens, file", training_ens[i])
  ## split into training and test 
  
  if(by_frac){
    n_train <- ceiling(n_train_fraction*n)
  }else{
    n_train <- n_train_obs
  }
  prec_mat_list_tr[[i]] <- prec_mat_sqrt[,,1:n_train]
  prec_mat_list_te[[i]] <- prec_mat_sqrt[,,(n_train+1):n]
  
  cat("\nDim prec train:", dim(prec_mat_sqrt[,,1:n_train])) #dim returns dimension of object, cat prints out
  cat("\nDim prec test:", dim(prec_mat_sqrt[,,(n_train+1):n])) #dim returns dimension of object, cat prints out
  
}


## load data
for(i in seq_along(holdout_ens)){
  cat('\n processing holdout : ', holdout_ens[i])
  # check whether file belongs to training ensemble members
  
  
  if(holdout_ens[i] %in% df_file_info_models$Identifier){
    prec_nc_path <- df_file_info_models[which(df_file_info_models$Identifier == holdout_ens[i]), "pr_path"]
    var_pr <- 'pr'
  }
  
  else if(holdout_ens[i] %in% df_file_info_reanalysis$Identifier){
    prec_nc_path <- df_file_info_reanalysis[which(df_file_info_reanalysis$Identifier == holdout_ens[i]), "pr_path"]
    var_pr <- df_file_info_reanalysis[which(df_file_info_reanalysis$Identifier == holdout_ens[i]), "pr_variable_name"]
    pr_format <- df_file_info_reanalysis[which(df_file_info_reanalysis$Identifier == holdout_ens[i]), "pr_format"]
    cat("\n reanalysis pr format", pr_format)
  }
  
  
  else {
    stop("\n Filepath of: ", holdout_ens[i], " not found")
  }
  
  
  cat('\n pr path: ',  prec_nc_path, " var: ", var_pr)
  
  prec_nc <- nc_open(file.path(data_dir,prec_nc_path))
  prec_arr_r <- ncvar_get(prec_nc, var_pr)
  
  prec_arr <- array(0, dim=c(dim(prec_arr_r)[2],dim(prec_arr_r)[1],dim(prec_arr_r)[3]))
  
  
  for (k in 1:dim(prec_arr_r)[3]){
    prec_arr[,,k] = apply(t(prec_arr_r[,,k]),2,rev)
    
  }
  if (mask_to_eobs){
    eobs_mask_path <- load("/net/h2o/climphys1/jkuettel/lin_lat_autoencoder_data/grids/masks/R_masks/eobs_mask_land.gzip")
    
    for (t in 1:dim(prec_arr)[3]){
      prec_arr[,,t] = eobs_mask * prec_arr[,,t] 
      
    }
  }
  
  cat("\n pr_format: ", pr_format)
  
  if (pr_format == "mm/s"){
    cat("\n got mm/s")
    prec_arr <- prec_arr * 3600 * 24 # convert from mm/s into mm/d
    
  }  else if (pr_format == "m/day"){
    cat("\n got m/day")
    prec_arr <- prec_arr * 1000 # convert from m/d into mm/d
    
  }  else if (pr_format == "mm/day"){
    cat("\n got mm/day")
    prec_arr <- prec_arr  
    
  }  else{
    stop("raw pr_format", pr_format ," of ens ", training_ens[i], "is not known")
  }
  
  if (scale_holdout) {
    prec_arr <- prec_arr * scale_factor # convert from mm/s into mm/d
  }
  
  if (cut_holdout) {
    prec_arr <- prec_arr[,,1:n_HOLDOUT]
  }
  
  else {
    prec_arr <- prec_arr
  }
  
  
  ### apply sqrt transform
  prec_mat_sqrt <- sqrt(prec_arr)
  
  cat('\n dim prec array: ', dim(prec_mat_sqrt))
  
  ## all negative pr values are now nans, replace them with zeros
  prec_mat_sqrt[is.na( prec_mat_sqrt)] <- 0
  
  cat("\nDim prec:", dim(prec_mat_sqrt)) #dim returns dimension of object, cat prints out
  
  
  prec_mat_list_ho[[i]] <- prec_mat_sqrt
}





prec_mat_sqrt_tr <- do.call(abind, prec_mat_list_tr)
prec_mat_sqrt_te <- do.call(abind, prec_mat_list_te)

cat("\nDim total prec train:", dim(prec_mat_sqrt_tr))
cat("\nDim total prec test:", dim(prec_mat_sqrt_te))


# temperature
test = brick("/net/h2o/climphys1/sippels/_DATASET/CanESM2_LE_reg/tas/tas_EUR-11_CCCma-CanESM2_historical_rcp85_OURANOS-CRCM5_1d_ENSMEAN_1d00.nc") + 0
test.mon = subset(test, seq(1, 1740, 12))  # select only januaries
test.mon.values2 = (values(test.mon))
na.ix = unique(which(is.na(test.mon.values2), arr.ind=T)[,1])
test.mon.values2 = test.mon.values2[-na.ix,]
raster.area = values(raster::area(test.mon))[-na.ix]
test.mon.values2.scale = (
  test.mon.values2 - rowMeans(test.mon.values2)) * raster.area / sum(raster.area)
test.mon.values2.scale.SVD = svd(test.mon.values2.scale)

warming_trend <- test.mon.values2.scale.SVD$v[,1] * (-1)
if(do_scale){
  warming_trend <- scale(warming_trend)
}
warming_trend_add_to_daily <- rep(warming_trend, each=90)
warming_trend_add_to_daily_tr <- warming_trend_add_to_daily[1:n_train]
warming_trend_add_to_daily_te <- warming_trend_add_to_daily[(n_train+1):n]

temp_ens_mean_eof_ho <- warming_trend_add_to_daily
temp_ens_mean_eof_tr <- rep(warming_trend_add_to_daily_tr, 
                            times = length(training_ens))
temp_ens_mean_eof_te <- rep(warming_trend_add_to_daily_te, 
                            times = length(training_ens))

years <- unique(as.numeric(sapply(names(psl), function(i) substr(i, 2, 5))))
df_temp <- paste(paste(years, ": ", warming_trend, sep = ""), collapse=",")
sel <- c(1955, 1975, 1995, 2015, 2035, 2055, 2075, 2095)
df_temp_sel <- paste(paste(years[is.element(years, sel)], ": ", 
                           warming_trend[is.element(years, sel)], sep = ""), collapse=",")

df_psl <- paste(paste(years, ": ", fitted_psl_v1, sep = ""), collapse=",")
df_psl_sel <- paste(paste(years[is.element(years, sel)], ": ", 
                          fitted_psl_v1[is.element(years, sel)], sep = ""), collapse=",")

## dates
date_indices = brick('/net/h2o/climphys1/heinzec/sds/psl_nc_data/psl_EUR-11_CCCma-CanESM2_historical_OURANOS-CRCM5_kba_1d_ALLYEARS_1d00_1_2_12.nc') + 0
## dates




dates_tr <- rep(names(date_indices)[1:n_train], times = length(training_ens))
dates_te <- rep(names(date_indices)[(n_train+1):n], times = length(training_ens))
dates_ho <- names(date_indices)

cat("\n len of train dates ", dim(dates_tr))
cat("\n len of test dates ", dim(dates_te))


# save
psl_Z_tr <- psl_pcs$Z_u_train
save(psl_Z_tr, prec_mat_sqrt_tr, dates_tr, 
     file=file.path(rda_data_path, paste("train_", detrend_id, ".rda", sep = "")))

psl_Z_te <- psl_pcs$Z_u_test
save(psl_Z_te, prec_mat_sqrt_te, dates_te,
     file=file.path(rda_data_path, paste("test_", detrend_id, ".rda", sep = "")))

for(i in 1:length(holdout_ens)){
  psl_Z_ho <- psl_pcs$Z_u_ho[[i]] 
  prec_mat_sqrt_ho <- prec_mat_list_ho[[i]]
  save(psl_Z_ho, prec_mat_sqrt_ho, dates_ho, 
       file=file.path(rda_data_path, 
                      paste(holdout_ens[i], "_holdout_", detrend_id, ".rda", sep = "")))
}

save(fitted_psl_v1, warming_trend, 
     file=file.path(rda_data_path, paste("trends_", detrend_id, ".rda", sep = "")))


# json

config = list(
  
  "detrend_id" = detrend_id,
  "psl_index" = psl_index_path,
  "training_files" = training_ens,
  "holdout_files" = holdout_ens,
  "months" = months,
  "npc_psl" = n_pc_psl, 
  "detrend" = detrend,
  "detrend_file" = ifelse(detrend, psl_ensmean_path, "None"),
  "scale" = do_scale,
  "test_split" = ifelse(by_frac, n_train_fraction, n_train_obs),
  "spline_dof" = spline_dof,
  "spline_lamda" = spline_lambda,
  "mask_to_eobs" = mask_to_eobs,
  "mask_slp_to_canesm" = mask_slp_to_canesm,
  "n_CORDEX" = n_CORDEX,
  "n_SMHI" = n_SMHI,
  "cut_holdout" = cut_holdout, 
  "n_HOLDOUT" = n_HOLDOUT,
  "scale_train" = scale_train,
  "scale_holdout" = scale_holdout,
  "scale_factor" = scale_factor
)

jsonData <- toJSON(config, pretty = TRUE, auto_unbox = TRUE)
write(jsonData, paste(save_path, "config.json", sep = "/"))