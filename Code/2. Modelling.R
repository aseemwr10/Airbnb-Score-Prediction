library(tidyverse)
library(tidytext)
library(stringr)
library(dplyr)
library(tibble)
library(tidyr)
library(readxl)
library(data.table)
library(gdata)
library(ggplot2)
library(caret)
library(foreach)
library(doParallel)
library(parallel)
library(caret)
library(gbm)
library(class)
library(tree)
library(randomForest)
library(ISLR)
library(glmnet)
library(xgboost)


airbnb_train <- read_csv("Final Train Data.csv")
airbnb_test <- read_csv("Final Test Data.csv")

airbnb_train_sub <- airbnb_train %>% select(-c(source,count_market,count_property_type)) %>% 
  group_by(jurisdiction_names) %>% 
  mutate(jur_count = n(),
         jurisdiction_names = if_else(jur_count < 100, "Other",jurisdiction_names)) %>% 
  ungroup() %>% select(-jur_count)

count_unique <- as.data.frame(sapply(airbnb_train_sub, function(x) length(unique(x))))

col_class <- as.data.frame(sapply(airbnb_train_sub, function(x) class(x))) %>% 
  mutate(ID = row_number()) %>% 
  pivot_longer(accommodates:perfect_score, names_to = 'column_name',values_to = "value") %>% 
  pivot_wider(names_from = ID, values_from = value)

count_unique$column_name <- row.names(count_unique)
names(count_unique)[1] <- "count_unique"                           

data_summary <- left_join(col_class,count_unique, by = 'column_name')

airbnb_train_sub <- airbnb_train_sub %>% select(-c(host_location,host_neighbourhood,zipcode,neighbourhood,
                                                   smart_location))

na_count <-sapply(airbnb_train_sub, function(y) sum(length(which(is.na(y)))))
na_count <- as.data.frame(na_count)
na_count$var <- row.names(na_count)
na_count <- na_count %>% mutate(total = nrow(airbnb_train_sub), per_na = na_count/total) %>% 
  select(-total)

set.seed(1)

dummy_var <- dummyVars( ~ . , data=airbnb_train_sub)
one_hot_airbnb <- data.frame(predict(dummy_var, newdata = airbnb_train_sub))

one_hot_train$sourcetest <- NULL
one_hot_train$sourcetrain <- NULL

one_hot_test$sourcetest <- NULL
one_hot_test$sourcetrain <- NULL
one_hot_test$perfect_score <- NULL


train_indices <- sample(nrow(one_hot_airbnb),.7*nrow(one_hot_airbnb))

model_train <- one_hot_airbnb[train_indices,]
model_valid <- one_hot_airbnb[-train_indices,]

confusion_generator <- function(actuals, scores, cutoff){
  
  classifications <- if_else(scores > cutoff, 1, 0)
  gen_CM <- table(actuals, classifications)
  
  TP <- gen_CM[2,2]
  TN <- gen_CM[1,1]
  FP <- gen_CM[1,2]
  FN <- gen_CM[2,1]
  
  TPR <- TP/(TP+FN)
  FPR <- FP/(TN+FP)
  accuracy <- (TP+TN)/(TP+TN+FP+FN)
  
  return(c(accuracy, TPR, FPR))
}




#Random Forest
model_train$perfect_score <- as.factor(model_train$perfect_score)

rf.mod <- randomForest(perfect_score ~ ., data = model_train,mtry = 20, ntree=500,importance=TRUE)

rf_preds <- predict(rf.mod, newdata=model_valid, type = "prob")
rf_class <- rf_preds[,2]

cutoffs <- seq(0.44,0.45,0.001)
for (i in cutoffs){
  print(i)
  print(confusion_generator(model_valid$perfect_score,rf_class,i))
}

#cutoff = 0.448
rf_acc <- mean(ifelse(rf_preds==model_valid$perfect_score,1,0))

rf.mod
rf_acc

#Nested Cross Validation

cv_sample <- sample(nrow(one_hot_train),10000)

airbnb_cv <- one_hot_train[cv_sample,]
airbnb_cv$perfect_score <- as.factor(airbnb_cv$perfect_score)

RF_predict_eval <- function(ntree_val, valid_data, train_data){
  rf.mod <- randomForest(perfect_score ~ ., 
                         data=train_data,
                         mtry=20, ntree=ntree_val)
  rf_preds <- predict(rf.mod, newdata=valid_data)
  accuracy <- mean(ifelse(rf_preds==valid_data$perfect_score,1,0))
  return(accuracy)
}

ntree_grid <-  c(50,100,200,500,1000)

k_outer <- 5
k_inner <- 3

outer_accs <- rep(0, k_outer)

outer_folds <- cut(seq(1,nrow(airbnb_cv)),breaks=k_outer,labels=FALSE)

parameter_count <- 0

for (i in c(1:k_outer)){
  
  outer_valid_inds <- which(outer_folds == i, arr.ind=TRUE)
  outer_valid_fold <- airbnb_cv[outer_valid_inds,]
  outer_train_fold <- airbnb_cv[-outer_valid_inds,]
  
  inner_folds <- cut(seq(1,nrow(outer_train_fold)),breaks=k_inner,labels=FALSE)
  
  print(paste("outer fold = ", i))
  
  grid_accs <- rep(0, length(ntree_grid))
  for (n_ind in c(1:length(ntree_grid))){
    nval <- ntree_grid[n_ind]
    inner_accs <- rep(0, k_inner)
    
    for (j in c(1:k_inner)){
      
      
      
      inner_valid_inds <- which(inner_folds == j, arr.ind=TRUE)
      inner_valid_fold <- airbnb_cv[inner_valid_inds, ]
      inner_train_fold <- airbnb_cv[-inner_valid_inds, ]
      
      inner_acc <- RF_predict_eval(nval, inner_valid_fold, inner_train_fold)
      parameter_count <- parameter_count+1
      
      inner_accs[j] <- inner_acc
      print(paste("grid value = ", nval, ", inner fold = ", j, "accuracy = ", inner_acc))
      
    }
    grid_accs[n_ind] <- mean(inner_accs)
  }
  
  print("")
  print("mean 3-fold inner CV accuracy for each ntree value is: ")
  print(grid_accs)
  print("")
  
  best_ntree_ind <- which.max(grid_accs) 
  best_ntree <- ntree_grid[best_ntree_ind]
  
  outer_acc <- RF_predict_eval(best_ntree, outer_valid_fold, outer_train_fold)
  parameter_count <- parameter_count+1
  outer_accs[i] <- outer_acc
  
  
  print(paste("for outer fold = ", i, ", best value for ntrees was: ", best_ntree, ", outer accuracy of model trained with this param = ", outer_acc))
  print("")
  
} # Nested CV for RF

overall_accuracy <- mean(outer_accs)
print(paste("overall accuracy across 10 folds: ", overall_accuracy, ", number of times parameters learned = ", parameter_count))


model1 <- glm(perfect_score~., data = model_train, family = "binomial")

cutoffs <- seq(0.49,0.5,0.001)

y <- predict(model1,newdata = model_valid,type = "response")

model_valid$perfect_score <- as.factor(model_valid$perfect_score)
for (i in cutoffs){
  print(i)
  print(confusion_generator(model_valid$perfect_score,y,i))
}

y_class <- if_else(y  > 0.495, 1,0)
y_class <- data.frame(y_class)


#tree
mycontrol = tree.control(nobs = nrow(model_train), mincut = 10, minsize = 20, mindev = 0)
cl_tree <- tree(perfect_score ~ . ,
                         data = model_train, 
                         control = mycontrol)

# for different cuts
for (i in 2:20) {
  prune_tree <- prune.tree(cl_tree , best = 5*i)
  tree_preds <- as.data.frame(predict(cl_tree,newdata=model_valid))
  tree_probs=tree_preds[,1]
  print(5*i)
  for (k in seq(0.6,0.8,0.02)) {
    print(k)
    print(confusion_generator(model_valid$perfect_score,tree_probs, k))
  }
  
} # getting cutoff and pruning values for tpr and fpr



tree_predict_eval <- function(mincut, valid_data, train_data){
  mycontrol = tree.control(nobs = nrow(model_train), mincut = mincut, minsize = 2*mincut, mindev = 0)
  cl_tree <- tree(perfect_score ~ . ,
                  data = train_data, 
                  control = mycontrol)
  tree_preds <- as.data.frame(predict(cl_tree,newdata=valid_data))
  tree_probs=tree_preds[,1]
  classification <- ifelse(tree_probs > 0.5 , 1, 0)
  accuracy <- mean(ifelse(classification==valid_data$perfect_score,1,0))
  return(accuracy)
}


#gbm

set.seed(10)

airbnb_test$perfect_score <- NA

full_data <- rbind(airbnb_train, airbnb_test)

full_data_sub <- full_data %>% select(-c(source,count_market,count_property_type,host_location,host_neighbourhood,zipcode,neighbourhood,
                                         smart_location))

dummy_var <- dummyVars( ~ . + price:property_type + room_type:accommodates, data=full_data_sub)
one_hot_airbnb <- data.frame(predict(dummy_var, newdata = full_data_sub))

test_index <- which(is.na(one_hot_airbnb$perfect_score))
one_hot_airbnb_train <- one_hot_airbnb[-test_index,]
one_hot_airbnb_test <- one_hot_airbnb[test_index,]

#Splitting the training set into train and test
insts <- sample(nrow(one_hot_airbnb_train), 0.7*nrow(one_hot_airbnb_train))
train_data <- one_hot_airbnb_train[insts,]
valid_data <- one_hot_airbnb_train[-insts,]

#Top 50 predictors
top_50_features <- as_tibble(summary.gbm(boost.mod)[c(1:50),])

#random search for choosing values of n.trees and interaction.depth
#Highest accuracy for n.trees=1404 and interaction.depth=10
ntree_random <- sample(100:1000, 10, replace = T)
int_depth_random <- sample(1:10, 10, replace = T)

for (i in c(1:10)){
  ntree <- ntree_random[i]
  int_depth <- int_depth_random[i]
  
  boost1 <- gbm(perfect_score~., data = train_data, 
                distribution = "bernoulli",
                n.trees = ntree,
                interaction.depth = int_depth)
  
  boost_preds <- predict(boost1,newdata=valid_data_x,type='response')
  cf <- confusion_generator(valid_data_y, boost_preds, 0.475)
  print(paste("Num trees = ",ntree,"interaction.depth =", int_depth, "acc = ",cf[1], "tpr = ", cf[2], "fpr = ", cf[3]))
  
}

#Boosting for up weighing misclassified instances
boost.mod <- gbm(perfect_score~.,data=train_data,
                 distribution="bernoulli",
                 n.trees=465,
                 interaction.depth=9)

valid_data_x <- valid_data %>%
  select(-perfect_score)

valid_data_y <- valid_data$perfect_score

boost_preds <- predict(boost.mod,newdata=valid_data_x,type='response')

confusion_generator(valid_data_y, boost_preds, 0.515)

boost.mod.full <- gbm(perfect_score~.,data=one_hot_airbnb_train,
                      distribution="bernoulli",
                      n.trees=465,
                      interaction.depth=9)

test_data_x <- one_hot_airbnb_test %>%
  select(-perfect_score)

boost_preds <- predict(boost.mod.full,newdata=test_data_x,type='response')

classify_final_test <- ifelse(boost_preds > 0.505, 1, 0)
classify_final_test <- as.data.frame(classify_final_test)

write_csv(classify_final_test, "Final_Output_3.csv")

#knn
set.seed(1)
train.X = model_train %>%
  select(-perfect_score)
valid.X= model_valid %>%
  select(-perfect_score)

train.y=model_train$perfect_score
valid.y=model_valid$perfect_score

knn.winning.1 <- knn(train.X, valid.X, train.y, k = 5, prob = TRUE)
knn.winning.2 <- knn(train.X, valid.X, train.y, k = 10, prob = TRUE)
knn.winning.3 <- knn(train.X, valid.X, train.y, k = 15, prob = TRUE)
knn.winning.4 <- knn(train.X, valid.X, train.y, k = 20, prob = TRUE)
knn.winning.5 <- knn(train.X, valid.X, train.y, k = 25, prob = TRUE)
knn.winning.6 <- knn(train.X, valid.X, train.y, k = 30, prob = TRUE)
knn.winning.7 <- knn(train.X, valid.X, train.y, k = 40, prob = TRUE)
knn.winning.8 <- knn(train.X, valid.X, train.y, k = 45, prob = TRUE)
knn.winning.9 <- knn(train.X, valid.X, train.y, k = 50, prob = TRUE)
knn.winning.10 <- knn(train.X, valid.X, train.y, k = 55, prob = TRUE)
knn.winning.11 <- knn(train.X, valid.X, train.y, k = 60, prob = TRUE)

#these are the proportions of the winning class
knn.probs.1 <- attr(knn.winning.1, "prob")
knn.probs.2 <- attr(knn.winning.2, "prob")
knn.probs.3 <- attr(knn.winning.3, "prob")
knn.probs.4 <- attr(knn.winning.4, "prob")
knn.probs.5 <- attr(knn.winning.5, "prob")
knn.probs.6 <- attr(knn.winning.6, "prob")
knn.probs.7 <- attr(knn.winning.7, "prob")
knn.probs.8 <- attr(knn.winning.8, "prob")
knn.probs.9 <- attr(knn.winning.9, "prob")
knn.probs.10 <- attr(knn.winning.10, "prob")
knn.probs.11 <- attr(knn.winning.11, "prob")

#these are the probabilities that y=1 (aka what we want)
knn.prob_of_1.1 <- ifelse(knn.winning.1 == 1, knn.probs.1, 1-knn.probs.1)
knn.prob_of_1.2 <- ifelse(knn.winning.2 == 1, knn.probs.2, 1-knn.probs.2)
knn.prob_of_1.3 <- ifelse(knn.winning.3 == 1, knn.probs.3, 1-knn.probs.3)
knn.prob_of_1.4 <- ifelse(knn.winning.4 == 1, knn.probs.4, 1-knn.probs.4)
knn.prob_of_1.5 <- ifelse(knn.winning.5 == 1, knn.probs.5, 1-knn.probs.5)
knn.prob_of_1.6 <- ifelse(knn.winning.6 == 1, knn.probs.6, 1-knn.probs.6)
knn.prob_of_1.7 <- ifelse(knn.winning.7 == 1, knn.probs.7, 1-knn.probs.7)
knn.prob_of_1.8 <- ifelse(knn.winning.8 == 1, knn.probs.8, 1-knn.probs.8)
knn.prob_of_1.9 <- ifelse(knn.winning.9 == 1, knn.probs.9, 1-knn.probs.9)
knn.prob_of_1.10 <- ifelse(knn.winning.10 == 1, knn.probs.10, 1-knn.probs.10)
knn.prob_of_1.11 <- ifelse(knn.winning.11 == 1, knn.probs.11, 1-knn.probs.11)


compare.1 <- cbind(knn.prob_of_1.1, valid.y)
compare.2 <- cbind(knn.prob_of_1.2, valid.y)
compare.3 <- cbind(knn.prob_of_1.3, valid.y)
compare.4 <- cbind(knn.prob_of_1.4, valid.y)
compare.5 <- cbind(knn.prob_of_1.5, valid.y)
compare.6 <- cbind(knn.prob_of_1.6, valid.y)
compare.7 <- cbind(knn.prob_of_1.7, valid.y)
compare.8 <- cbind(knn.prob_of_1.8, valid.y)
compare.9 <- cbind(knn.prob_of_1.9, valid.y)
compare.10 <- cbind(knn.prob_of_1.10, valid.y)
compare.11 <- cbind(knn.prob_of_1.11, valid.y)

for (i in seq(0.2,0.9,0.05)) {
  print(i)
  print(confusion_generator(valid.y, knn.prob_of_1.1, i))
}

#Lasso Ridge

set.seed(1)
airbnb_train_sub <- airbnb_train %>% select(-c(source,count_market,count_property_type))

airbnb_train_sub <- airbnb_train_sub %>% select(-c(host_location,host_neighbourhood,zipcode,neighbourhood,
                                                   smart_location))


x <- model.matrix(perfect_score~.,airbnb_train_sub)
y <- airbnb_train_sub$perfect_score
train <- sample(nrow(airbnb_train_sub),.7*nrow(airbnb_train_sub))
valid <- airbnb_train_sub[-train,]
x_train <- x[train,]
x_valid <-  x[-train,]

y_train <- y[train]
y_valid <- y[-train]

grid <- 10^seq(10,-2,length=100)

k<-5

#ridge
cv.out <- cv.glmnet(x_train, y_train, family="binomial", alpha=0, lambda=grid, nfolds=k)
bestlam <- cv.out$lambda.min
pred <- predict(cv.out, s=bestlam, newx = x_valid,type="response")

for (i in seq(0.4,0.6,0.01)) {
  print(i)
  print(confusion_generator(y_valid, pred, i))
}


#lasso
cv.out1 <- cv.glmnet(x_train, y_train, family="binomial", alpha=1, lambda=grid, nfolds=k)
bestlam1<- cv.out1$lambda.min
pred1 <- predict(cv.out1, s=bestlam1, newx = x_valid,type="response")

for (i in seq(0.4,0.6,0.01)) {
  print(i)
  print(confusion_generator(y_valid, pred1, i))
}




#XG Boost
gc()

dummy_var <- dummyVars( ~ . + price:property_type , data=airbnb_train_sub)
one_hot_airbnb <- data.frame(predict(dummy_var, newdata = airbnb_train_sub))
gc()

tr_insts <- sample(nrow(one_hot_airbnb), 0.7*nrow(one_hot_airbnb))
train.X <- one_hot_airbnb[tr_insts, c(1:418)]
valid.X <- one_hot_airbnb[-tr_insts, c(1:418)]
train.y <- one_hot_airbnb[tr_insts, 419]
valid.y <- one_hot_airbnb[-tr_insts, 419]



train.X_m <- as.matrix(train.X)
train.y_m <- as.matrix(train.y)
valid.X_m <- as.matrix(valid.X)
valid.y_m <- as.matrix(valid.y)
xgboost <- xgboost(train.X_m, label = train.y_m, objective = "binary:logistic", max.depth = 3,
                   eta = 0.1, nrounds = 15)

mat <- xgb.importance(feature_names = colnames(train.X_m),model = xgboost)
xgb.plot.importance(importance_matrix = mat[1:20]) 

preds <- predict(xgboost, newdata = valid.X_m)
classify <- ifelse(preds > 0.5, 1, 0)
corrects <- ifelse(classify == valid.y_m, 1, 0)
acc <- sum(corrects)/length(corrects)

CM <- table(valid.y, classify)
CM
FP <- CM[1,2]
FN <- CM[2,1]
TP <- CM[2,2]
TN <- CM[1,1]

TPR <- TP/(TP+FN) 
TNR <- TN/(TN+FP) #0.1528
FPR <- 1-TNR #0.0280

random_nrounds <- sample(5:10, 10, replace = T)
max_depth <- sample(1:10, 10, replace = T)
cutoff <- runif(10, 0.44, 0.48)
eta <- runif(15, 0.1, 0.7)

accuracies <- c()
tpr <- c()
fpr <- c()

one_hot_airbnb_m <- as.matrix(one_hot_airbnb)
labeled_shuffle <- one_hot_airbnb_m[sample(nrow(one_hot_airbnb_m)),]

c(1:5,7:9)
rm(airbnb_train)

#separate into 10 equally-sized folds
#cut() will assign "fold numbers" to each instance
folds <- cut(seq(1,nrow(labeled_shuffle)),breaks=10,labels=FALSE)
for (i in c(1:10)){
  valid_inds <- which(folds==i,arr.ind=TRUE)
  
  train.X <- labeled_shuffle[-valid_inds, c(1:418, 420:438)]
  valid.X <- labeled_shuffle[valid_inds, c(1:418, 420:438)]
  train.y <- labeled_shuffle[-valid_inds, 419]
  valid.y <- labeled_shuffle[valid_inds, 419]
  
  nrounds <- random_nrounds[i]
  max_d <- random_nrounds[i]
  c <- cutoff[i] #0.47
  
  xgboost <- xgboost(train.X_m, label = train.y_m, objective = "binary:logistic", max.depth = 3,
                     eta = 0.1, nrounds = 10)
  
  preds <- predict(xgboost, newdata = valid.X_m)
  classify <- ifelse(preds > c, 1, 0)
  corrects <- ifelse(classify == valid.y_m, 1, 0)
  acc <- sum(corrects)/length(corrects)
  
  accuracies[i] <- acc
  
  CM <- table(valid.y_m, classify)
  if(ncol(CM)==1){
    i <- i-1
  } else{
    FP <- CM[1,2]
    FN <- CM[2,1]
    TP <- CM[2,2]
    TN <- CM[1,1]
    
    TPR <- TP/(TP+FN)
    TNR <- TN/(TN+FP)
    FPR <- 1-TNR
    
    tpr[i] <- TPR
    fpr[i] <- FPR
    
    
    print(paste("acc = ",acc, "tpr = ", TPR, "fpr = ", FPR, "Cutoff = ", c))
  }
  
}

ind <- which.max(tpr)
ind
cutoff[7]
fpr[7]
max_depth[7]
random_nrounds[7]

tpr
fpr
cutoff









