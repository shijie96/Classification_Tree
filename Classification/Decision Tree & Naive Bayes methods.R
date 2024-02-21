# Use classification methods to explore dataset of heartdisease.
# date : 10/04/2023
# author: Shijie Geng

library(rpart)
library(rpart.plot)
library(e1071)
library(caret)
library(tidyverse)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(klaR)

# Data manipulation
## rename a name of a column

heartdisease <- heartdisease %>%
  rename('HD' = 'target')

## To remove a column.

heartdisease[,'target'] <- NULL

summary(heartdisease)
unique(heartdisease$thal)
heartdisease$HD <- ifelse(heartdisease$HD == '1',
                              'Yes',
                              'No')
## change the type of columns of HD in datasets.

test_data$HD <- as.factor(test_data$HD)
train_data$HD <- as.factor(train_data$HD)
heartdisease$HD <- as.factor(heartdisease$HD)

# To split dataset into train_data and test_data.

split <- sample(nrow(heartdisease),floor(nrow(heartdisease)*0.7))
split
train_data <- heartdisease[split,]
test_data <- heartdisease[-split,]

# Create a tree model using train_data 
tree_model <- rpart(HD~.,data = train_data,method = 'class',parms = list(prior=c(0.3,0.7)))
predictiontest <- predict(tree_model,test_data,type = 'class')
confusionMatrix(predictiontest,test_data$HD,positive = 'Yes')
prp(tree_model, faclen=0, cex=0.7, extra=4, main="yourID Decision Tree")

# To generate a ROC curve
library(ROCR)
val1 <- predict(tree_model, test_data, type = 'prob')
predictionvalues <- prediction(val1[,2], test_data$HD)

## To display the probability under the curve.
performancevalues <- performance(predictionvalues, 'auc')
auc <- performancevalues@y.values[[1]]

## plot ROC curve
performancevalues_1 <- performance(predictionvalues, 'tpr','fpr')
plot(performancevalues_1)


# Naive bayes Classification
NBmodel <- naiveBayes(HD~., data = train_data)
y_pred <- predict(NBmodel, test_data)
confusionMatrix(y_pred, test_data$HD, positive = 'Yes')


