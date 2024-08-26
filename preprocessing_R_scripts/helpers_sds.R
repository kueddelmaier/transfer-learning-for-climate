
compute_pcs_ho_list <- function(temp_mat_tr, temp_mat_te, temp_mat_ho_list, n_pc_temp, non_zero_spatial_indexes, train_slp_detrend_values, test_slp_detrend_values, holdout_slp_detrend_list,
                                n_train=NULL, n_ens=NULL, fitted_vals=NULL, detrend=FALSE){

  ## flatten
  n_tr <- dim(temp_mat_tr)[3]
  p <- dim(temp_mat_tr)[1]*dim(temp_mat_tr)[2]
  temp_mat_flat_tr <- t(matrix(temp_mat_tr, nrow=p, ncol=n_tr))
  # sum(complete.cases(temp_mat_flat_tr)) # 0
  

  n_te <- dim(temp_mat_te)[3]
  temp_mat_flat_te <- t(matrix(temp_mat_te, nrow=p, ncol=n_te))

  temp_mat_flat_ho_list <- vector("list", length(temp_mat_ho_list))

  for(i in 1:length(temp_mat_ho_list)){
   
    n_ho <- dim(temp_mat_ho_list[[i]])[3]
    cat('\n dim temp_mat_ho_list[[[i]]: ', dim(temp_mat_ho_list[[i]]))
    temp_mat_flat_ho_list[[i]] <- t(matrix(temp_mat_ho_list[[i]], nrow=p, ncol=n_ho))
  }
  ### for some columns (817 - 3914) complete data 
  # range(which(complete.cases(t(temp_mat_flat_tr))))
  temp_mat_flat_tr_complete <- temp_mat_flat_tr[,non_zero_spatial_indexes]
  # dim(temp_mat_flat_tr_complete) 
  
  temp_mat_flat_te_complete <- temp_mat_flat_te[,non_zero_spatial_indexes]
  # dim(temp_mat_flat_te_complete) 
  
  temp_mat_flat_ho_complete_list <- vector("list", length(temp_mat_ho_list))
  for(i in 1:length(temp_mat_ho_list)){
    temp_mat_flat_ho_complete_list[[i]] <- temp_mat_flat_ho_list[[i]][,non_zero_spatial_indexes]
  }
  # dim(temp_mat_flat_ho_complete) 

  ## center before applying svd
  temp_mat_flat_tr_c <- scale(temp_mat_flat_tr_complete, center=TRUE, scale=FALSE)
  temp_mat_flat_te_c <- scale(temp_mat_flat_te_complete, center=TRUE, scale=FALSE)
  
  temp_mat_flat_ho_c_list <- vector("list", length(temp_mat_ho_list))
  for(i in 1:length(temp_mat_ho_list)){
    temp_mat_flat_ho_c_list[[i]] <-  scale(temp_mat_flat_ho_complete_list[[i]], center=TRUE, scale=FALSE)
  }
  
  temp_mat_flat_svd <- svd(temp_mat_flat_tr_c)
  
  ## take U matrix, so that all variables are on same scale
  ## scale to yield unit variance
  Ztemp_u_full <- scale(temp_mat_flat_svd$u)
  Ztemp_u_train <- Ztemp_u_full[,1:n_pc_temp]
  
  Ztemp_test_full <- temp_mat_flat_te_c %*% temp_mat_flat_svd$v 
  Ztemp_test_u_full <- scale(Ztemp_test_full %*% diag(1/temp_mat_flat_svd$d))
  Ztemp_u_test <- Ztemp_test_u_full[,1:n_pc_temp]
  
  Ztemp_u_ho_list <- vector("list", length(temp_mat_ho_list))
  for(i in 1:length(temp_mat_ho_list)){
    Ztemp_ho_full <- temp_mat_flat_ho_c_list[[i]] %*% temp_mat_flat_svd$v 
    Ztemp_ho_u_full <- scale(Ztemp_ho_full %*% diag(1/temp_mat_flat_svd$d))
    Ztemp_u_ho_list[[i]] <- Ztemp_ho_u_full[,1:n_pc_temp]
  }
  
  
  if(detrend){
    # inp <- rep(fitted_vals[1:n_train], times=n_ens)
    # fit <- lm(Ztemp_u_train ~ inp)
    fit <- lm(Ztemp_u_train ~ train_slp_detrend_values)
    Ztemp_u_train <- residuals(fit) 
    
    # inp <- rep(fitted_vals[(n_train+1):n], n_ens)
    # fit <- lm(Ztemp_u_test ~ inp)
    fit <- lm(Ztemp_u_test ~ test_slp_detrend_values)
    Ztemp_u_test <- residuals(fit)
    

    for(i in 1:length(temp_mat_ho_list)){

      cat("\n dim of holdout ",i, " is", dim(Ztemp_u_ho_list[[i]]))
      cat("\n dim of fittet_vals_reduced  is ", length(holdout_slp_detrend_list[[i]]))
      fit <- lm(Ztemp_u_ho_list[[i]] ~ holdout_slp_detrend_list[[i]])
      Ztemp_u_ho_list[[i]] <- residuals(fit)
    }   
  }
  

  return(list(Z_u_train = Ztemp_u_train, 
              Z_u_test = Ztemp_u_test, 
              Z_u_ho = Ztemp_u_ho_list))
}

project_eof <- function(mat, eof){
  ## flatten
  n <- dim(mat)[3]
  p <- dim(mat)[1]*dim(mat)[2]
  mat_flat <- t(matrix(mat, nrow=p, ncol=n))
  ### for some columns (817 - 3914) complete data 
  mat_flat_compl <- mat_flat[,which(complete.cases(t(mat_flat)))]
  ## center before applying svd
  mat_flat_c <- scale(mat_flat_compl, center=TRUE, scale=FALSE)
  projected <- scale(mat_flat_c %*% eof)
  return(projected)
}

create_id <- function(n = 5000) {
  a <- do.call(paste0, replicate(5, sample(LETTERS, n, TRUE), FALSE))
  paste0(a, sprintf("%04d", sample(9999, n, TRUE)), sample(LETTERS, n, TRUE))
}





