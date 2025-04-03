#independent variable 

age <- runif(400, 25, 80)
age
Age<- c(0, 25, 50, 90)
labels<- c("Young", "Adult", "Senior")
Age_grp <- cut(Age, breaks = Age, labels = labels, 
               include.lowest = TRUE)
Age_grp

gender <- factor(c("female", "male"))
gender

Family_history <- sample(c(0, 1),400, replace = TRUE)
Family_history_cat <- factor(Family_history, levels = c(0, 1), 
                             labels = c("Yes", "No"))

Family_history_cat

Genetic_mutations<- c("Missense", "Nonsense", "Frameshift", "Silent")
G_mutations <- sample(Genetic_mutations, size = 400, replace = TRUE)
print(G_mutations)



environmental_factors <- runif(400, 0, 1)
environmental_factors

cut_points <- quantile(environmental_factors, probs = c(0, 0.33, 0.66, 1))
labels <- c("Low", "Medium", "High")
environmental_factors_cat<- cut(environmental_factors, breaks = cut_points, 
                                labels = labels, include.lowest = TRUE)
environmental_factors_cat


lifestyle_choices <- c("Smoking", "Physical inactivity", "Unhealthy diet", "Alcohol consumption")
random_indices <- sample(length(lifestyle_choices), 400, replace = TRUE)
lifestyle_factors <- lifestyle_choices[random_indices]

lifestyle_factors


previous_cancer_history <- sample(c(0, 1),400, replace = TRUE)
previous_cancer_history_cat <- factor(previous_cancer_history, levels = c(0, 1), 
                                      labels = c("Yes", "No"))

previous_cancer_history_cat


immunodeficiency <- sample(c(0, 1), 400, replace = TRUE)
immunodeficiency_cat<- factor(immunodeficiency , levels = c(0, 1), 
                              labels = c("Yes", "No"))

immunodeficiency_cat


hormonal_factors <- rnorm(400, 50, 10)
cut_points <- quantile(hormonal_factors, probs = c(0, 0.25, 0.75, 1))
labels <- c("Low", "Medium", "High")
hormonal_factors_cat <- cut(hormonal_factors, breaks = cut_points, 
                            labels = labels, include.lowest = TRUE)
hormonal_factors_cat


viral_infections <- c("HPV", "Hepatitis B", "Hepatitis C", "EBV", "HIV")
random_indices <- sample(length(viral_infections), 400, replace = TRUE)
viral_infection_factors <- viral_infections[random_indices]
print(viral_infection_factors)


#independent variable

cancer_diagnosis <- sample(c(0, 1),400, replace = TRUE)
cancer_diagnosis
cancer_diagnosis_cat <- factor(cancer_diagnosis, levels = c(0, 1), 
                               labels = c("Yes", "No"))

cancer_diagnosis_cat
y<-cancer_diagnosis_cat
y



df<- data.frame(Age_grp, gender, Family_history_cat, G_mutations, environmental_factors_cat,
                lifestyle_factors, previous_cancer_history_cat, immunodeficiency_cat,
                hormonal_factors_cat, viral_infection_factors, y)

df



df$y<-as.factor(df$y)
str(df)


df$viral_infection_factors<- as.factor(df$viral_infection_factors)
str(df)
df$ lifestyle_factors<-as.factor(df$ lifestyle_factors)
str(df)
df$G_mutations<-  as.factor(df$G_mutations)
str(df)



#support vector machine
#support vector machine

install.packages("e1071")
library(e1071)

library(caret)
library(ggplot2)
library(lattice)
library(DescTools)
library(e1071)
data<- df
data
Abstract(data)






set.seed(123)
index <- sample(1:nrow(data), round(0.8*nrow(data)) ,replace = F)
data_train <- data[index,]
data_test <- data[-index,]




set.seed(123)

svm_model<- 
  svm(y ~ ., 
      data = data_train, 
      type = "C-classification", 
      kernel = "linear",
      scale = FALSE)


svm_model
summary(svm_model)

test_pred <- predict(svm_model, newdata = data_test)
test_pred

CF<- confusionMatrix(table(test_pred, data_test$y))
CF


#random forest
set.seed(123)
train_index <- createDataPartition(data$y, p = 0.8, list = FALSE, times = 1)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]


rf <- randomForest(y~., data=train_data, proximity=TRUE) 
print(rf)


p1 <- predict(rf, train_data)
cm <- confusionMatrix(p1, train_data$ y)
cm


p2 <- predict(rf, test_data)
cm1<- confusionMatrix(p2, test_data$ y)
cm1



#naive bayes model 

install.packages(naivebayes)
library(naivebayes)
library(dplyr)
library(ggplot2)
library(psych)

install.packages("e1071")
library(caret)
library(DescTools)
library(e1071)

Abstract(data)

set.seed(123)
index <- sample(1:nrow(data), round(0.8*nrow(data)) ,replace = F)
data_train <- data[index,]
data_test <- data[-index,]

nb_model <- naiveBayes(y ~ ., data = train_data)
nb_model



predictions <- predict(nb_model, train_data)
predictions

cm_train <- confusionMatrix(predictions, train_data$y)
cm_train 


predictions1 <- predict(nb_model, data_test)
predictions1

cm_train1 <- confusionMatrix(predictions1, data_test$y)
cm_train1 



#knn model

library(caret)
library(pROC)
library(mlbench)
library(ggplot2)
library(lattice)



set.seed(123)
index <- sample(1:nrow(data), round(0.8*nrow(data)) ,replace = F)
data_train <- data[index,]
data_test <- data[-index,]

trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3)
metric <- "Accuracy"


set.seed(5)
fit.knn <- train(y~., 
                 data=data_train, 
                 method="knn",
                 metric=metric,
                 trControl=trainControl)

knn.k1 <- fit.knn$dataset 
print(fit.knn)

plot(fit.knn)

set.seed(7)
p1 <- predict(fit.knn, train_data)
cm_TRAIN <- confusionMatrix(p1, train_data$y)
cm_TRAIN


p2 <- predict(fit.knn, data_test)
cm_test <- confusionMatrix(p2, data_test$y)
cm_test


#decision tree
install.packages("rpart")
library(rpart)
install.packages(c("rpart", "rpart.plot"))
library(caret)
library(DescTools)
library(rpart)
library(rpart.plot)
install.packages("caret")
install.packages("yardstick")
library(caret)
library(yardstick)
library(ggplot2)
library( lattice)

data<- df
data
set.seed(123)
train_index <- createDataPartition(data$y, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]


tree_model <- rpart(y ~ ., data = data, method = "class")
tree_model

tree_model$variable.importance

prp(x = tree_model, extra =2) 
base.trpreds <- predict(object = tree_model, 
                        newdata = train_data, 
                        type = "class") 

base.trpreds



DT_train_conf <- confusionMatrix(data = base.trpreds, # predictions
                                 reference = train_data$y, # actual
                                 positive = "Yes",
                                 mode = "everything")
DT_train_conf


base.trpreds <- predict(object = tree_model, 
                        newdata = test_data, 
                        type = "class") 

base.trpreds



DT_test_conf <- confusionMatrix(data = base.trpreds, # predictions
                                reference = test_data$y, # actual
                                positive = "Yes",
                                mode = "everything")
DT_test_conf


#glm model 


library(caret)
library(ggplot2)
library(lattice)


set.seed(123)
index <- sample(1:nrow(data), round(0.8*nrow(data)) ,replace = F)
data_train <- data[index,]
data_test <- data[-index,]


glm_model<- glm(y ~ Age_grp+gender+G_mutations+environmental_factors_cat+
                  lifestyle_factors+previous_cancer_history_cat+immunodeficiency_cat+
                  hormonal_factors_cat+viral_infection_factors,
                data = data_train, family = binomial(link = "logit"))

glm_model
summary(glm_model)


logitModelPred <- predict(glm_model, data_test, type = "response")

classify50 <- ifelse(logitModelPred > 0.5,"Yes","No")
classify50 <- ordered(classify50, levels = c("Yes", "No"))
data_test$y <- ordered(data_test$y, levels = c("Yes", "No"))

cm <- table(Predicted = classify50, Actual = data_test$y)
cm


cf<-confusionMatrix(cm)
cf























