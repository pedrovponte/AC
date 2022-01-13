library(mlbench)
library(performanceEstimation)
library(Rcpp)

data("PimaIndiansDiabetes")
dataset <- PimaIndiansDiabetes
formula <- diabetes ~ .
summary(PimaIndiansDiabetes)

# example <- function(form, train, test, opt=100, ...){
#   tgt <- which(colnames(train) == as.character(form[[2]]))
#   
#   # to be done
#   
#   res <- list(trues=test[,tgt], preds=p) # p must be defined
#   res
# }

# exp1 <- performanceEstimation::performanceEstimation(
#  PredTask(formula, dataset), #formula and dataset must be defined
#  c(workflowVariants("example",opt=c(1,2,3))),
#  # example is the name of the function previously defined.
#  EstimationTask(metrics = "auc", method = CV(nReps = 1, nFolds = 10), evaluator="AUC")
#)

# performanceEstimation::rankWorkflows(exp1, top = 3, maxs = TRUE)

# Function to calculate AUC
AUC <- function(trues,preds,...) {
  library(AUC)
  c(auc=AUC::auc(roc(preds, trues)))
}

########################### Exercise 1.1 ##################################

od.if <- function(form,train,test,ntrees=100,...) {
  require(solitude)
  tgt <- which(colnames(train)==as.character(form[[2]]))
  iso = isolationForest$new(num_trees=ntrees)
  iso$fit(train,...)
  p <- iso$predict(test)
  p <- p$anomaly_score
  p <- scale(p)
  res <- list(trues=test[,tgt], preds=p)
  res 
}

exp.od.if <- performanceEstimation::performanceEstimation(
  PredTask(formula,dataset),
  c(workflowVariants("od.if",ntrees=c(100,250,500,1000))),
  EstimationTask(metrics="auc",method=CV(nReps = 1, nFolds=10), evaluator="AUC"))

performanceEstimation::rankWorkflows(exp.od.if, top = 4, maxs = TRUE)

plot(exp.od.if)

getWorkflow("od.if.v3", exp.od.if)


########################### Exercise 1.2 ##################################
od.svm.linear <- function(form, train, test, classMaj, n=0.1) {
  require(e1071)
  tgt <- which(colnames(train)==as.character(form[[2]]))
  j <- which(train[,tgt]==classMaj)
  m.svm <- svm(form, train[j,], type='one-classification', kernel='linear', nu=n)
  p <- predict(m.svm,test)
  res <- list(trues=test[,tgt], preds=p)
  res 
}

exp.od.svm.linear <- performanceEstimation::performanceEstimation(
  PredTask(formula,dataset),
  c(workflowVariants("od.svm.linear", classMaj='neg',
                     n=c(seq(0.1,0.9,by=0.1)))),
  EstimationTask(metrics="auc",method=CV(nReps = 1, nFolds=10), evaluator="AUC"))

performanceEstimation::rankWorkflows(exp.od.svm.linear, top = 9, maxs = TRUE)

plot(exp.od.svm.linear)

getWorkflow("od.svm.linear.v3", exp.od.svm.linear)

########################### Exercise 1.3 ##################################

od.svm.radial <- function(form, train, test, classMaj, g, n=0.1) {
  require(e1071)
  tgt <- which(colnames(train)==as.character(form[[2]]))
  j <- which(train[,tgt]==classMaj)
  m.svm <- svm(form, train[j,], type='one-classification',
               kernel='radial', gamma=g, nu=n)
  p <- predict(m.svm,test)
  res <- list(trues=test[,tgt], preds=p)
  res 
}

exp.od.svm.radial <- performanceEstimation::performanceEstimation(
  PredTask(formula,dataset),
  c(workflowVariants("od.svm.radial", classMaj='neg',
                     g=c(1,2,4,8,16,32), n=c(seq(0.1,0.9,by=0.1)))),
  EstimationTask(metrics="auc",method=CV(nReps = 1, nFolds=10), evaluator="AUC"))

performanceEstimation::rankWorkflows(exp.od.svm.radial, top = 54, maxs = TRUE)

plot(exp.od.svm.radial)

getWorkflow("od.svm.radial.v50", exp.od.svm.radial)

########################### Exercise 1.4 ##################################

od.svm.sigmoid <- function(form, train, test, classMaj, g, c0, n=0.1) {
  require(e1071)
  tgt <- which(colnames(train)==as.character(form[[2]]))
  j <- which(train[,tgt]==classMaj)
  m.svm <- svm(form, train[j,], type='one-classification',
               kernel='sigmoid', gamma=g, coef0=c0, nu=n)
  p <- predict(m.svm,test)
  res <- list(trues=test[,tgt], preds=p)
  res 
}

exp.od.svm.sigmoid <- performanceEstimation::performanceEstimation(
  PredTask(formula,dataset),
  c(workflowVariants("od.svm.sigmoid", classMaj='neg',g=c(1,2,4,8,16,32),
                     c0= c(1,2,4,8,16,32),n=c(seq(0.1,0.9,by=0.1)))),
  EstimationTask(metrics="auc",method=CV(nReps = 1, nFolds=10), evaluator="AUC"))

performanceEstimation::rankWorkflows(exp.od.svm.sigmoid, top = 324, maxs = TRUE)  

plot(exp.od.svm.sigmoid)

getWorkflow("od.svm.sigmoid.v300", exp.od.svm.sigmoid)
