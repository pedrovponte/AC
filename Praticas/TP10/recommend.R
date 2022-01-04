library(recommenderlab)
library(dplyr)
library(readr)

setwd("D:/U. Porto/4 ano/1 semestre/Aprendizagem Computacional/Praticas/TP10")

log <- read_csv("log1.csv",col_types = list(col_factor(),col_factor()))
brm <- as(as.data.frame(log), "binaryRatingMatrix")
brm_offline <- brm[1:6,]

getData.frame(brm_offline)
getRatingMatrix(brm_offline)
inspect(getRatingMatrix(brm_offline))
rowCounts(brm_offline)
colCounts(brm_offline)
image(brm_offline)

#Association Rules
model <- Recommender(brm_offline, "AR")
getModel(model)

rules <- getModel(model)$rule_base

inspect(rules)

brm_u7 <- brm[7,]

recsAR <- predict(model, brm_u7, n=2)
recsAR
getList(recsAR)
r <- subset(rules, lhs %in% c("C", "F"))
inspect(r)

recsAR <- predict(model, brm[8,], n=2)
recsAR
getList(recsAR)
r <- subset(rules, lhs %in% c("C"))
inspect(r)

modelPop <- Recommender(brm_offline, "POPULAR")
recsPop <- predict(modelPop, brm_u7, n=2)
recsPop
getList(recsPop)

recsPop <- predict(modelPop, brm_u8, n=2)
recsPop
getList(recsPop)

## Colaborative Filtering

simJac_users <- similarity(brm_offline, method="jaccard")
simJac_users
simCos_users <- similarity(brm_offline, method="cosine")
simCos_users

simCos_items <- similarity(brm_offline, method="cosine", which="items")


