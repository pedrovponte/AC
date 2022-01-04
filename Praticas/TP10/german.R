library(dplyr)
library(tidyverse)

setwd("D:/U. Porto/4 ano/1 semestre/Aprendizagem Computacional/Praticas/TP10")

df <- read_csv('german_credit.csv')

ds <- df %>%
  mutate(duration_in_month=cut(duration_in_month,4,labels=c("short", "med-short","med-long","long")),
         credit_amount=cut()
         age=cut(age,4,lables=c("young adult","adult","senior","golden")))

df <- df %>% mutate_if(is.numeric,as.factor)

dfT <. as(df,"transactions")
itemInfo(dfT)

subset(itemInfo(dfT), variables == "age")
subset(itemInfo(dfT), variables %in% c("age","credite_amount","duration_in_month"))

rules <- apriori(dfT)