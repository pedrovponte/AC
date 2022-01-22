library(tidyverse)
library(caret)

set.seed(69)
data("iris")
dataset <- iris %>% select(Sepal.Length, Sepal.Width, Species) %>% mutate(Species = ifelse(Species == 'versicolor', 1, 0))
ggplot(dataset, aes(x=Sepal.Length, y=Sepal.Width, col=as.factor(Species))) + geom_point()
train_idx <- sample(nrow(dataset), 0.8*nrow(dataset))
train <- dataset[train_idx,]
test <- dataset[-train_idx,]

X_train <- scale(train[, c(1:2)])
y_train <- train$Species
dim(y_train) <- c(length(y_train), 1) # add extra dimension to vector
X_test <- scale(test[, c(1:2)])
y_test <- test$Species
dim(y_test) <- c(length(y_test), 1) # add extra dimension to vector
X_train <- as.matrix(X_train, byrow=TRUE)
X_train <- t(X_train)
y_train <- as.matrix(y_train, byrow=TRUE)
y_train <- t(y_train)
X_test <- as.matrix(X_test, byrow=TRUE)
X_test <- t(X_test)
y_test <- as.matrix(y_test, byrow=TRUE)
y_test <- t(y_test)

getLayerSize <- function(X, y, hidden_n) {
  size <- list("n_x" = dim(X)[1],
               "n_h" = hidden_n,
               "n_y" = dim(y)[1])
  return (size)
}

layer_size <- getLayerSize(X_train, y_train, 4)

initializeParameters <- function(X, list_layer_size){
  m <- dim(data.matrix(X))[2]
  n_x <- list_layer_size$n_x
  n_h <- list_layer_size$n_h
  n_y <- list_layer_size$n_y
  W1 <- matrix(runif(n_h * n_x), nrow = n_h, ncol = n_x, byrow = TRUE) * 0.01
  b1 <- matrix(rep(0, n_h), nrow = n_h)
  W2 <- matrix(runif(n_y * n_h), nrow = n_y, ncol = n_h, byrow = TRUE) * 0.01
  b2 <- matrix(rep(0, n_y), nrow = n_y)
  params <- list("W1" = W1,
                 "b1" = b1,
                 "W2" = W2,
                 "b2" = b2)
  return (params)
}

init_params <- initializeParameters(X_train, layer_size)
lapply(init_params, function(x) dim(x))

sigmoid <- function(x) {
  return (1 / (1 + exp(-x)))
}

forwardPropagation <- function(X, params, list_layer_size){
  m <- dim(X)[2]
  n_h <- list_layer_size$n_h
  n_y <- list_layer_size$n_y
  W1 <- params$W1
  b1 <- params$b1
  W2 <- params$W2
  b2 <- params$b2
  b1_new <- matrix(rep(b1, m), nrow = n_h)
  b2_new <- matrix(rep(b2, m), nrow = n_y)
  Z1 <- W1 %*% X + b1_new
  A1 <- sigmoid(Z1)
  Z2 <- W2 %*% A1 + b2_new
  A2 <- sigmoid(Z2)
  cache <- list("Z1" = Z1,
                "A1" = A1,
                "Z2" = Z2,
                "A2" = A2)
  return (cache)
}

fwd_prop <- forwardPropagation(X_train, init_params, layer_size)
lapply(fwd_prop, function(x) dim(x))
  
computeCost <- function(X, y, cache) {
  m <- dim(X)[2]
  A2 <- cache$A2
  logprobs <- (log(A2) * y) + (log(1-A2) * (1-y))
  cost <- -sum(logprobs/m)
  return (cost)
}
cost <- computeCost(X_train, y_train, fwd_prop)
cost

backwardPropagation <- function(X, y, cache, params, list_layer_size){
  m <- dim(X)[2]
  n_x <- list_layer_size$n_x
  n_h <- list_layer_size$n_h
  n_y <- list_layer_size$n_y
  A2 <- cache$A2
  A1 <- cache$A1
  W2 <- params$W2
  dZ2 <- A2 - y
  dW2 <- 1/m * (dZ2 %*% t(A1))
  db2 <- matrix(1/m * sum(dZ2), nrow = n_y)
  db2_new <- matrix(rep(db2, m), nrow = n_y)
  dZ1 <- (t(W2) %*% dZ2) * (1 - A1^2)
  dW1 <- 1/m * (dZ1 %*% t(X))
  db1 <- matrix(1/m * sum(dZ1), nrow = n_h)
  db1_new <- matrix(rep(db1, m), nrow = n_h)
  grads <- list("dW1" = dW1,
                "db1" = db1,
                "dW2" = dW2,
                "db2" = db2)
  return(grads)
}

back_prop<-backwardPropagation(X_train,y_train,fwd_prop,init_params,layer_size)
lapply(back_prop, function(x) dim(x))

updateParameters <- function (grads, params, learning_rate){
  W1 <- params$W1
  b1 <- params$b1
  W2 <- params$W2
  b2 <- params$b2
  dW1 <- grads$dW1
  db1 <- grads$db1
  dW2 <- grads$dW2
  db2 <- grads$db2
  W1 <- W1 - learning_rate * dW1
  b1 <- b1 - learning_rate * db1
  W2 <- W2 - learning_rate * dW2
  b2 <- b2 - learning_rate * db2
  updated_params <- list("W1" = W1,
                         "b1" = b1,
                         "W2" = W2,
                         "b2" = b2)
  return (updated_params)
}

update_params <- updateParameters(back_prop, init_params, learning_rate = 0.01)
lapply(update_params, function(x) dim(x))

trainModel <- function(X, y, num_iteration, hidden_neurons, lr){
  layer_size <- getLayerSize(X, y, hidden_neurons)
  init_params <- initializeParameters(X, layer_size)
  cost_history <- c()
  for (i in 1:num_iteration) {
    fwd_prop <- forwardPropagation(X, init_params, layer_size)
    cost <- computeCost(X, y, fwd_prop)
    back_prop <- backwardPropagation(X, y, fwd_prop, init_params, layer_size)
    update_params <- updateParameters(back_prop, init_params, learning_rate = lr)
    init_params <- update_params
    cost_history <- c(cost_history, cost)
    if (i %% 10000 == 0) cat("Iteration", i, " | Cost: ", cost, "\n")
  }
  model_out <- list("updated_params" = update_params,
                    "cost_hist" = cost_history)
  return (model_out)
}

EPOCHS = 60000
HIDDEN_NEURONS = 40
LEARNING_RATE = 0.1
train_model <- trainModel(X_train, y_train, hidden_neurons = HIDDEN_NEURONS, num_iteration = EPOCHS, lr = LEARNING_RATE)

nn_lr09 <- trainModel(X_train, y_train, hidden_neurons = HIDDEN_NEURONS, num_iteration = EPOCHS, lr = 0.9)
nn_lr01 <- trainModel(X_train, y_train, hidden_neurons = HIDDEN_NEURONS, num_iteration = EPOCHS, lr = 0.1)

loss_lr09 <- tibble(epochs=1:EPOCHS, loss=nn_lr09$cost_hist)
ggplot(loss_lr09, aes(x=epochs,y=loss)) + geom_point()

logreg_model <- glm(Species~.,data=train)
logreg_preds <- predict(logreg_model,test[,1:2])
logreg_preds <- round(logreg_preds)
layer_size <- getLayerSize(X_train,y_test,HIDDEN_NEURONS)
params <- nn_lr01 $updated_params
fwd_prop <- forwardPropagation(X_test,params,layer_size)
mm_pred <- round(fwd_prop $A2)

confusionMatrix(as.factor(y_test),as.factor(logreg_preds))
confusionMatrix(as.factor(y_test),as.factor(mm_pred))

