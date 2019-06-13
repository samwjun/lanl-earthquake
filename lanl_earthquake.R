library(moments)
library(zoo)
library(readr)
library(TSEntropies)
library(tsfeatures)
library(dplyr)
library(doSNOW)
library(kernlab)
library(CVST)
library(Matrix)
library(caret)
library(lightgbm)


Sys.setenv('R_MAX_VSIZE'=64000000000)
options(digits=12)


#feature engineering
create_features <- function(seg_x,seg_y){
 new_df <- data.frame(
                      "mean" = mean(seg_x),
                      "std" = sd(seg_x),
                      "max" = max(seg_x),
                      "min" = min(seg_x),
                      "kurt" = moments::kurtosis(seg_x),
                      "skew" = moments::skewness(seg_x),
                      "mean_change_abs" = mean(diff(seg_x)),
                      
                      "q1" = quantile(seg_x,0.01),
                      "q2" = quantile(seg_x,0.02),
                      "q3" = quantile(seg_x,0.03),
                      "q4" = quantile(seg_x,0.04),
                      "q5" = quantile(seg_x,0.05),
                      "q6" = quantile(seg_x,0.06),
                      "q7" = quantile(seg_x,0.07),
                      "q8" = quantile(seg_x,0.08),
                      "q9" = quantile(seg_x,0.09),
                      "q91" = quantile(seg_x,0.91),
                      "q92" = quantile(seg_x,0.92),
                      "q93" = quantile(seg_x,0.93),
                      "q94" = quantile(seg_x,0.94),
                      "q95" = quantile(seg_x,0.95),
                      "q96" = quantile(seg_x,0.96),
                      "q97" = quantile(seg_x,0.97),
                      "q98" = quantile(seg_x,0.98),
                      "q99" = quantile(seg_x,0.99),
                      
                      "ApEn" = TSEntropies::FastApEn_C(seg_x),
                      "SampEn" = TSEntropies::FastSampEn(seg_x),
                      
                      "Shannon_En" = tsfeatures::entropy(seg_x),
                      "stability" = tsfeatures::stability(seg_x),
                      "lumpiness" = tsfeatures::lumpiness(seg_x),
                      "hurst" = tsfeatures::hurst(seg_x),
                      "crossing_pts" = tsfeatures::crossing_points(seg_x),
                      "flat_spots" = tsfeatures::crossing_points(seg_x),
                      "nonlinearity" = tsfeatures::nonlinearity(seg_x),
                      "arch_stat" = tsfeatures::arch_stat(seg_x),
                      "motiftwo_entro3" = tsfeatures::motiftwo_entro3(seg_x),
                      "trev_num" = tsfeatures::trev_num(seg_x),
                      "walker_propcross" = tsfeatures::walker_propcross(seg_x),
                      "std1st_der" = tsfeatures::std1st_der(seg_x)
                      
                      ) 
 
 stl <- tsfeatures::stl_features(seg_x)
 #new_df[,"nperiods"] = stl[1]
 #new_df[,"seasonal_period"] =  stl[2]
 new_df[,"trend"] = stl[3]
 new_df["spike"] = stl[4]
 new_df["linearity"] = stl[5]
 new_df["curvature"] = stl[6]
 new_df["e_acf1"] = stl[7]
 new_df["eacf10"] = stl[8]

 acff <- tsfeatures::acf_features(seg_x)
 new_df[,"x_acf1"] = acff[1]
 new_df[,"x_acf10"] = acff[2]
 new_df[,"diff1_acf1"] = acff[3]
 new_df[,"diff1_acf10"] = acff[4]
 new_df[,"diff2_acf1"] = acff[5]
 new_df[,"diff2_acf10"] = acff[6]
 
 pacff <- tsfeatures::pacf_features(seg_x)
 new_df[,"x_pacf5"] = pacff[1]
 new_df[,"diff1x_pacf5"] = pacff[2]
 new_df[,"diff2x_pacf5"] = pacff[3]
 
 holt <- tsfeatures::holt_parameters(seg_x)
 new_df[,"holt_alpha"] = holt[1]
 new_df[,"holt_beta"] = holt[2]
 
 garch <- tsfeatures::heterogeneity(ts(seg_x))
 new_df[,"arch_acf"] = garch[1]
 new_df[,"garch_acf"] = garch[2]
 new_df[,"arch_r2"] = garch[3]
 new_df[,"garch_r2"] = garch[4]
 
 
 for (w in c(10,50,100,200,1000)){
   roll_mean = zoo::rollmean(seg_x,w)
   roll_sd = zoo::rollapplyr(seg_x,w,sd)
   
   new_df[,c(paste0("mean_roll_std_",w))]= mean(roll_sd)
   new_df[,c(paste0("std_roll_std_",w))] = sd(roll_sd)
   new_df[,c(paste0("max_roll_std_",w))] = max(roll_sd)
   new_df[,c(paste0("min_roll_std_",w))] = min(roll_sd)
   new_df[,c(paste0("q1_roll_std_",w))] = quantile(roll_sd,0.01)
   new_df[,c(paste0("q5_roll_std_",w))] = quantile(roll_sd,0.05)
   new_df[,c(paste0("q95_roll_std_",w))] = quantile(roll_sd,0.95)
   new_df[,c(paste0("q99_roll_std_",w))] = quantile(roll_sd,0.99)
   
   new_df[,c(paste0("mean_roll_mean_",w))]= mean(roll_mean)
   new_df[,c(paste0("std_roll_mean_",w))] = sd(roll_mean)
   new_df[,c(paste0("max_roll_mean_",w))] = max(roll_mean)
   new_df[,c(paste0("min_roll_mean_",w))] = min(roll_mean)
   new_df[,c(paste0("q1_roll_mean_",w))] = quantile(roll_mean,0.01)
   new_df[,c(paste0("q5_roll_mean_",w))] = quantile(roll_mean,0.05)
   new_df[,c(paste0("q95_roll_mean_",w))] = quantile(roll_mean,0.95)
   new_df[,c(paste0("q99_roll_mean_",w))] = quantile(roll_mean,0.99)
   new_df[,c(paste0("kurt_roll_mean_",w))] = moments::kurtosis(roll_mean)
   new_df[,c(paste0("skew_roll_mean_",w))] = moments::skewness(roll_mean)
  
 }
 
smoothed_spec <- spec.pgram(seg_x,kernel("daniell"),200)
new_df[,"mean_spec"] = mean(smoothed_spec$spec)
new_df[,"sd_spec"] = sd(smoothed_spec$spec)
new_df[,"peak_freq"] = smoothed_spec$freq[which.max(smoothed_spec$spec)]

if (!missing(seg_y)) {new_df[,"time2failure"] = seg_y[length(seg_y)]}

return (new_df)
}




#split train data into data frames with 150,000 rows and extract features
trainlist <-as.data.frame(read_csv("train.csv"))
numrows<-floor(nrow(trainlist)/150000)
trainlist <- split(trainlist,c(rep(1:numrows,times=1,each=150000),rep("NA",nrow(trainlist)%%150000)))
trainlist[[length(trainlist)]] <- NULL
trainlist <- trainlist[order(as.numeric(names(trainlist)))]


cl <- snow::makeCluster(3,outfile="")
registerDoSNOW(cl)
pb <- txtProgressBar(max=length(trainlist),style = 3)
progress <- function(n) setTxtProgressBar(pb,n)
opts <- list(progress = progress)

traindf <- foreach(i = 1:length(trainlist), .combine=rbind, .options.snow = opts, .verbose = T) %dopar% {
gc()
create_features(seg_x=trainlist[[i]][,1],seg_y=trainlist[[i]][,2])
}
close(pb)
snow::stopCluster(cl)
rm(cl)
gc()

tmp <- list.files(path=".", pattern="*.csv")
cl <- snow::makeCluster(7,outfile="")
registerDoSNOW(cl)
testdf <- foreach(i = 1:length(tmp), .combine=rbind, .verbose = T) %dopar% {
  gc()
  create_features(seg_x=as.data.frame(readr::read_csv(paste0("C:\\Users\\wjun\\Downloads\\test\\",tmp[i])))$acoustic_data)
}
snow::stopCluster(cl)
rm(cl)
gc()





#create stratified cv folds

#tquake <- which(diff(traindf$time2failure) > 1)
#quake_idx <- c(rep(1:16,c(tquake[1],diff(tquake))),rep(1,nrow(traindf)-tail(tquake,1)))
ttf_idx <- round(traindf$time2failure,0)
ttf_fold <- caret::createFolds(factor(ttf_idx[-c(outliers)]),k=5,list=FALSE)


#outliers based on visual inspection (spikes in the middle of the earthquake cycle)
outliers <- c(213,533,1096,1827,3727,4082)

#remove constant features
zero_sd<-names(which(apply(traindf,2,sd)==0))

traindf<-traindf[,!names(traindf) %in% zero_sd]
testdf<-testdf[,!names(testdf) %in% zero_sd]

#remove duplicate/linearly dependent features (cor=1)
cor.mat <- cor(traindf[,names(traindf)!="time2failure"])
t<-as.data.frame(which(cor.mat==1,arr.ind=TRUE))
t<-t[t$row!=t$col,]

cols_to_remove <- c()
unique_rows <- unique(t$row)
for (i in 1:length(unique_rows)){
  if(unique_rows[i] %in% cols_to_remove){cols_to_remove=cols_to_remove}
  else {cols_to_remove <- c(cols_to_remove,t[t$row==unique_rows[i],"col"])}
}
traindf<-traindf[,-c(cols_to_remove)]
testdf<-testdf[,-c(cols_to_remove+1)]

#remove highly correlated features
cor.mat <- cor(traindf[,names(traindf)!="time2failure"])
highlyCorrelated <- findCorrelation(cor.mat, cutoff=0.99)
traindf<-traindf[,-c(highlyCorrelated)]
testdf<-testdf[,-c(highlyCorrelated+1)]

#Near Zero Var
nzv <- nearZeroVar(traindf[,names(traindf)!="time2failure"], saveMetrics = TRUE)
cols_to_remove <- which(nzv$nzv==TRUE)
traindf<-traindf[,-c(cols_to_remove)]
testdf<-testdf[,-c(cols_to_remove+1)]




#lasso regression

lasso.cv<-cv.glmnet(as.matrix(traindf[-c(outliers),names(traindf)!="time2failure"]), 
                    traindf$time2failure[-c(outliers)], 
                    alpha=1, standardize=TRUE, type.measure='mae',foldid=ttf_fold)
coeff<-as.matrix(coef(lasso.cv, s=lasso.cv$lambda.min))
min(lasso.cv$cvm)
lasso.cv$lambda.min

lasso_test_pred <-predict(lasso.cv,newx=as.matrix(testdf[,names(testdf)!="tmp"]))
lasso_test_pred[lasso_test_pred < 0] <- 0






#KRR

#tried reducing features with pca/kpca in the cv but ended up not using in the final model
#regular PCA
quakepca <- function(train,val,num_feat){
  trainpca <- prcomp(train,center=TRUE,scale=TRUE)
  reduced_train <- trainpca$x[,1:num_feat]
  rot_mat <-trainpca$rot[,1:num_feat]
  train_feat_mean <- apply(train,2,mean)
  train_feat_sd <- apply(train,2,sd)
  val_centered_scaled <-as.matrix(sweep(sweep(val,2,train_feat_mean,"-"),2,train_feat_sd,"/"))
  reduced_val <- val_centered_scaled %*% rot_mat
  return (list(reduced_train=reduced_train,reduced_val=reduced_val))
}

#Scree plot
pc_train_all <- prcomp(traindf[,names(traindf)!="time2failure"],center=TRUE,scale=TRUE)
plot(pc_train_all$sdev,type="l",ylab="SD of PC",xlab="PC number")
x<-(pc_train_all$sdev/sum(pc_train_all$sdev))
cumsum(x) 

#kernel PCA
quakekpca <- function(train,val,num_feat,kernel = "rbfdot", kpar = list(sigma=0.0005)){
  x <- as.matrix(scale(train))
  kpc <- kernlab::kpca(x,kernel=kernel,kpar=kpar)
  reduced_train <- kernlab::rotated(kpc)[,1:num_feat]
  train_feat_mean <- apply(train,2,mean)
  train_feat_sd <- apply(train,2,sd)
  val_scaled <-as.matrix(sweep(sweep(val,2,train_feat_mean,"-"),2,train_feat_sd,"/"))
  reduced_val <- kernlab::predict(kpc,as.matrix(val_scaled))[,1:num_feat]
  return (list(reduced_train=reduced_train,reduced_val=reduced_val))
}

#cross validation with KRR
krr.cv <-function(traindf,sigma, lambda, pc_num, fold){
  train_MAE <- c()
  test_MAE <- c()
  train_MAE_name <- c()
  test_MAE_name <- c()
  train_weight <- c()
  test_weight <- c()
  
  
  for (k in 1:length(unique(fold))){
    
    print (paste0("Fold ",k))
    
    fold_k <- which(fold==k)
    train_x <-traindf[-fold_k,names(traindf)!="time2failure"]
    val_x <- traindf[fold_k,names(traindf)!="time2failure"]
    train_y <-traindf[-fold_k,names(traindf)=="time2failure"]
    val_y <- traindf[fold_k,names(traindf)=="time2failure"]
    
    if (pc_num != 0){
      train_xf <- quakekpca(train=train_x,val=val_x,num_feat=pc_num)$reduced_train
      val_xf <- quakekpca(train=train_x,val=val_x,num_feat=pc_num)$reduced_val
    }
    else {
      train_xf<- scale(train_x,center=TRUE,scale=TRUE)
      train_feat_mean <- apply(train_x,2,mean)
      train_feat_sd <- apply(train_x,2,sd)
      val_xf <-as.matrix(sweep(sweep(val_x,2,train_feat_mean,"-"),2,train_feat_sd,"/"))
    }
    
    train_CVST <- CVST::constructData(x=train_xf,y=train_y)
    val_CVST <- CVST::constructData(x=val_xf,y=val_y)
    #krr = constructKRRLearner()
    
    param <- list(kernel="rbfdot",sigma=sigma,lambda=lambda)
    krr_model <- CVST::constructKRRLearner()$learn(train_CVST,param)
    krr_train_pred <- CVST::constructKRRLearner()$predict(krr_model,train_CVST)
    krr_test_pred <- CVST::constructKRRLearner()$predict(krr_model,val_CVST)
    
    train_MAE[k] <- sum(abs(krr_train_pred - train_y))/length(train_y)
    test_MAE[k] <- sum(abs(krr_test_pred - val_y))/length(val_y)
    train_weight[k] <- length(train_y)/((length(unique(fold)) - 1) *nrow(traindf))
    test_weight[k] <- length(val_y)/nrow(traindf)
    train_MAE_name <- c(train_MAE_name,paste0("train_MAE_",k))
    test_MAE_name <- c(test_MAE_name,paste0("test_MAE_",k))
  }
  train_MAE <- c(train_MAE,sum(train_MAE*train_weight))
  test_MAE <- c(test_MAE,sum(test_MAE*test_weight))
  names(train_MAE) <- c(train_MAE_name,"train_MAE_tot")
  names(test_MAE) <- c(test_MAE_name,"test_MAE_tot")
  
  return (cbind.data.frame(t(train_MAE),t(test_MAE)))
}

#grid search
grid_param = expand.grid(sigma=c(0.002990950),
                         lambda=c(8.451179e-05),
                         pc_num=c(0))

cl <- snow::makeCluster(7,outfile="")
registerDoSNOW(cl)

mae_grid_param <- foreach(i = 1:nrow(grid_param), .combine=rbind, .verbose = T) %dopar% {
  gc()
  krr.cv(traindf=traindf_lasso5[-c(outliers),],
         fold = ttf_fold,
         sigma=grid_param[i,"sigma"],
         lambda=grid_param[i,"lambda"],
         pc_num=grid_param[i,"pc_num"]
  )
}

snow::stopCluster(cl)
rm(cl)
gc()
mae_grid_param <- cbind.data.frame(grid_param,mae_grid_param)
View(mae_grid_param[order(mae_grid_param$test_MAE_tot),])

#predict on test data using fitted KRR
x<-traindf[,names(traindf)!="time2failure"]
y<-traindf$time2failure
train <- scale(x,center=TRUE, scale = TRUE)
train_CVST <- constructData(x=train[-c(outliers),],y=y[-c(outliers)])
krr = constructKRRLearner()

param <- list(kernel="rbfdot",sigma = 0.002990950, lambda = 8.451179e-05)
krr_model <- krr$learn(train_CVST,param)
test_scaled <- as.matrix(sweep(sweep(testdf[,names(testdf)!="tmp"],2,train_feat_mean,"-"),2,train_feat_sd,"/"))
test_CVST <- constructData(x=test_scaled,y=rep(0,nrow(testdf)))
krr_test_pred <- krr$predict(krr_model,test_CVST)





#xgboost
k=6
ttf_fold_list <- caret::createFolds(factor(ttf_idx[-c(outliers)]),k=k,returnTrain = TRUE)

grid_xgb <- expand.grid(
  nrounds = c(100),
  max_depth = c(4),
  eta = c(0.033),
  gamma = c(2),
  colsample_bytree = c(0.9),
  min_child_weight = c(1),
  subsample = c(0.35)
)

#Used for grid search
#control <- trainControl(index=t,method="adaptive_cv", verboseIter  = TRUE, number = k, allowParallel = T, 
#                        preProcOptions = list(c("center", "scale")))

#Used after grid search
control <- trainControl(method="none")


x = traindf[-c(outliers),names(traindf)!="time2failure"]
y = traindf$time2failure[-c(outliers)]
model <- train(x = x, y = y, 
               method="xgbTree", trControl=control, tuneGrid=grid_xgb, metric = "MAE", verbose = TRUE)

#predict on test data
xgb_test_pred <- predict(model, newdata = testdf[,names(testdf)!="tmp"])






#model stacking
#The CV has data leak, but used here anyways because there is only ~4000 rows of data and is used often in practice.

#using only the predictions from the models as features
k=5
ttf_fold <- caret::createFolds(factor(ttf_idx),k=k,list=FALSE)

df_stack_train <- data.frame(krr = double(), lasso = double(), xgb = double(), time2failure = double())
for (k in 1:length(unique(ttf_fold))){
  
  #krr
  fold_k <- which(ttf_fold==k)
  x_krr_tr<-traindf[-fold_k,names(traindf)!="time2failure"]
  x_krr_val <- traindf[fold_k,names(traindf)!="time2failure"]
  y<-traindf$time2failure[-fold_k]
  y_val <- traindf$time2failure[fold_k]
  
  train <- scale(x_krr_tr,center=TRUE, scale = TRUE)
  train_CVST <- constructData(x=train[-c(outliers),],y=y[-c(outliers)])
  krr = constructKRRLearner()
  
  param <- list(kernel="rbfdot",sigma = 0.002990950, lambda = 8.451179e-05)
  krr_model <- krr$learn(train_CVST,param)
  
  
  train_feat_mean <- apply(x_krr_tr,2,mean)
  train_feat_sd <- apply(x_krr_tr,2,sd)
  dat <- as.matrix(sweep(sweep(x_krr_val,2,train_feat_mean,"-"),2,train_feat_sd,"/"))
  dat_CVST <- constructData(x=dat,y=y_val)
  
  
  krr_pred <- krr$predict(krr_model,dat_CVST)
  krr_pred[krr_pred < 0] <- 0
  df_stack_train[fold_k,c("krr")] <- krr_pred
  df_stack_train[fold_k,c("time2failure")] <- y_val
  
  #lasso
  x_lasso_tr <- traindf[-fold_k,names(traindf)!="time2failure"]
  x_lasso_val <- traindf[fold_k,names(traindf)!="time2failure"]
  lasso.mod <- glmnet(x = as.matrix(x_lasso_tr[-c(outliers),]), y = y[-c(outliers)], alpha = 1, lambda = 0.02606788, standardize=TRUE)
  lasso_pred <- predict(lasso.mod,newx = as.matrix(x_lasso_val))
  lasso_pred[lasso_pred<0] <- 0
  df_stack_train[fold_k,c("lasso")] <- lasso_pred
  
  #xgb
  x_xgb_tr <- traindf[-fold_k,names(traindf)!="time2failure"]
  x_xgb_val <- traindf[fold_k,names(traindf)!="time2failure"]
  fitControl <- trainControl(method = "none")
  params <- expand.grid(
    nrounds = c(100),
    max_depth = c(4),
    eta = c(0.033),
    gamma = c(2),
    colsample_bytree = c(0.9),
    min_child_weight = c(1),
    subsample = c(0.35)
  )
  xgb.mod <- train(x = x_xgb_tr, y = y, 
                   method="xgbTree", trControl = fitControl, tuneGrid=params, metric = "MAE", verbose = TRUE)
  df_stack_train[fold_k,c("xgb")] <- predict(xgb.mod,newdata=x_xgb_val)
  
}

df_stack_test <- data.frame(krr= krr_test_pred, lasso = lasso_test_pred, xgb = xgb_test_pred, id = testdf$tmp)


#Adding walker_propcross feature from the original traindf
df_stack_train2 <- cbind.data.frame(traindf$walker_propcross,df_stack_train)
grid_param = expand.grid(sigma=c(0.001),
                         lambda=c(3.769231e-07),
                         pc_num=c(0))


#Used KRR as a final model to stack the 3 models
cl <- snow::makeCluster(7,outfile="")
registerDoSNOW(cl)

mae_grid_param <- foreach(i = 1:nrow(grid_param), .combine=rbind, .verbose = T) %dopar% {
  gc()
  krr.cv(traindf=df_stack_train2,
         fold = ttf_fold,
         sigma=grid_param[i,"sigma"],
         lambda=grid_param[i,"lambda"],
         pc_num=grid_param[i,"pc_num"]
  )
}

snow::stopCluster(cl)
rm(cl)
gc()
mae_grid_param <- cbind.data.frame(grid_param,mae_grid_param)
View(mae_grid_param[order(mae_grid_param$test_MAE_tot),])

train <- scale(df_stack_train2[,names(df_stack_train2)!="time2failure"],center=TRUE, scale = TRUE)
train_CVST <- constructData(x=train[-c(outliers),],y=y[-c(outliers)])
krr = constructKRRLearner()

param <- list(kernel="rbfdot",sigma = 0.001, lambda = 3.769231e-07)
krr_model <- krr$learn(train_CVST,param)

df_stack_test2 <- cbind.data.frame(testdf$walker_propcross,df_stack_test[,c(1:3)])
train_feat_mean <- apply(df_stack_train2[,names(df_stack_train2)!="time2failure"],2,mean)
train_feat_sd <- apply(df_stack_train2[,names(df_stack_train2)!="time2failure"],2,sd)
dat <- as.matrix(sweep(sweep(df_stack_test2,2,train_feat_mean,"-"),2,train_feat_sd,"/"))
dat_CVST <- constructData(x=dat,y=y_val)


krr_pred <- krr$predict(krr_model,dat_CVST)
krr_pred[krr_pred < 0] <- 0
submission <- cbind.data.frame(testdf$tmp,krr_pred)
write.csv(submission,"submission_final.csv")


#lightgbm
# lgb_cv_fun <- function(params,traindf,fold){
#   train_MAE <- c()
#   test_MAE <- c()
#   train_MAE_name <- c()
#   test_MAE_name <- c()
#
#
#   for (k in 1:length(unique(fold))){
#
#     print (paste0("Fold ",k))
#
#     fold_k <- which(fold==k)
#     train_x <-traindf[-fold_k,names(traindf)!="time2failure"]
#     val_x <- traindf[fold_k,names(traindf)!="time2failure"]
#     train_y <-traindf[-fold_k,names(traindf)=="time2failure"]
#     val_y <- traindf[fold_k,names(traindf)=="time2failure"]
#
#     dtrain <-lightgbm::lgb.Dataset(as.matrix(train_x),label=train_y)
#     gbm <- lightgbm::lgb.train(params = as.list(params), data = dtrain)
#
#     test_pred <- predict(gbm,as.matrix(val_x))
#     train_pred <- predict(gbm,as.matrix(train_x))
#     train_MAE[k] <- sum(abs(train_pred - train_y))/length(train_y)
#     test_MAE[k] <- sum(abs(test_pred - val_y))/length(val_y)
#     train_MAE_name <- c(train_MAE_name,paste0("train_MAE_",k))
#     test_MAE_name <- c(test_MAE_name,paste0("test_MAE_",k))
#   }
#
#   train_MAE <- c(train_MAE,mean(train_MAE))
#   test_MAE <- c(test_MAE,mean(test_MAE))
#   names(train_MAE) <- c(train_MAE_name,"train_MAE_tot")
#   names(test_MAE) <- c(test_MAE_name,"test_MAE_tot")
#
#   return (cbind.data.frame(t(train_MAE),t(test_MAE)))
# }
# 
# grid_param = expand.grid(objective = "regression",
#                          feature_fraction = c(0.7),
#                          bagging_fraction = c(0.7870858),
#                          bagging_freq = 5,
#                          max_depth = c(5),
#                          min_data = c(150),
#                          lambda_l1 = 3.5,
#                          lambda_l2 = 1.8,
#                          min_gain_to_split = c(5),
#                          learning_rate = c(0.32),
#                          num_leaves = 14,
#                          max_bin = 200,
#                          metric = "mae",
#                          stringsAsFactors = FALSE)
# 
# cl <- snow::makeCluster(7,outfile="")
# registerDoSNOW(cl)
# pb <- txtProgressBar(max=nrow(grid_param),style = 3)
# progress <- function(n) setTxtProgressBar(pb,n)
# opts <- list(progress = progress)
# 
# mae_grid_param <- foreach(i = 1:nrow(grid_param), .combine=rbind, .options.snow = opts, .verbose = T) %dopar% {
#   gc()
#   lgb_cv_fun(params = grid_param[i,],
#              traindf = traindf[-c(outliers)],
#              fold = ttf_fold
#   )
# }
# close(pb)
# snow::stopCluster(cl)
# rm(cl)
# gc()
# lgbm_grid_result <- cbind.data.frame(grid_param,mae_grid_param)
# View(lgbm_grid_result[order(lgbm_grid_result$test_MAE_tot),])

# param = data.frame(objective = "regression",
#                    feature_fraction = c(0.7),
#                    bagging_fraction = c(0.7870858),
#                    bagging_freq = 5,
#                    max_depth = c(5),
#                    min_data = c(150),
#                    lambda_l1 = 3.5,
#                    lambda_l2 = 1.8,
#                    min_gain_to_split = c(5),
#                    learning_rate = c(0.32),
#                    num_leaves = 14,
#                    max_bin = 200,
#                    metric = "mae",
#                    stringsAsFactors = FALSE
# )

# dtrain <- lgb.Dataset(as.matrix(traindf[-c(outliers),names(traindf!="time2failure")]),label=traindf$time2failure)
# 
# lgb_model <- lgb.train(params = as.list(param), data = dtrain)
# lgb_test_pred <- predict(lgb_model,testdf[,names(testdf!="tmp")])
