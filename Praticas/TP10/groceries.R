library(arules)
library(arulesViz)

data(Groceries)
Groceries
unique(Groceries)
class(Groceries)
summary(Groceries)
unique(Groceries)
inspect(Groceries[1:5])
size(Groceries)

boxplot(size(Groceries))

itemInfo(Groceries)

itemFrequency(Groceries)
itemFrequency(Groceries, type="absolute")

# top 5 more frequent items (without mining)
itemFrequencyPlot(Groceries, topN=5)
# items that appear in at least 10% of transactions
itemFrequencyPlot(Groceries, support=0.1)

fsets <- apriori(Groceries,
                 parameter=list(supp=0.01, minlen=1, target="frequent itemsets"))

# fsets <- apriori(Groceries, 
#                  parameter=list(supp=0.1,target="frequent itemsets"))

class(fsets)

# inspect(fsets)
inspect(sort(fsets)[1:5])

quality(fsets)

fsetsclosed <- fsets[is.closed(fsets)]
fsetmax <- fsets[is.maximal(fsets)]

inspect(fsetsmax[1:5])

rules <- apriori(Groceries)

rules <- apriori(Groceries, parameter=list(supp=0.01, conf=0.5))

rules <- apriori(Groceries, parameter=list(supp=0.01, conf=0.25))

summary(rules)
inspect(rules)[1:5]
rules.sub <- subset(rules, subset=lift>2)
inspect(rules.sub)

# by default sort of association rules has decreasing=TRUE
rules.sub <- subset(rules, subset=rhs %in% c("yogurt", "whole milk") & lift>2)
rules.sort <- sort(rules.sub, by="lift")
inspect(rules.sort)

plot(rules.sort)

rules.milk <- subset(rules, subset=rhs %in% "whole milk" & size(rules) > 2)

plot(rules.sort, method="graph")

plot(rules.sort, method="grouped")

