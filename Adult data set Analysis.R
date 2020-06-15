---
  title: "IDV Project on Adult data set"
author: "Kanchan Singh"
date: "6/5/2020"
output:
  pdf_document:
  latex_engine: xelatex
html_document:
  df_print: paged
fontsize: 12pt
---
  
  **Introduction**
  
  
  *The adult dataset is from  UCI Machine Learning Repository. I will predict whether individual’s annual income exceeds $50,000 using the set of variables in this data set. The question is inspected in two different approaches – traditional statistical modeling and machine learning techniques. Logistic regression is used as the statistical modeling tool as the outcome is binary. Two different machine learning techniques – regression tree and random forest are used to answer the same question. *
  
  
  *First, I will load the dataset and do data cleaning, data exploration and visualization.*
  
  
  **Important Library**
  
  
  ```{r,warning=FALSE,message=FALSE}
library(tidyverse)
library(tidyr)
library(rvest)
library(ggplot2)
library(plyr)
library(ROCR)
library(data.table)
library(rvest)
library(lattice)
library(caret)
library(randomForest)
library(knitr)
library(datasets)

```



**#Exploratory Data Analysis and Data Cleaning**
  
  
  **#Imported data in R and then used attach function**
  
  
  
  ```{r}
adult<-read.csv('/Users/kanchansingh/R from Harvard/capstone project/Individual project/adult.data')
head(adult)
```



**# There was no header name so changed variable name into specific column name**
  
  
  
  ```{r}
colnames(adult) <- c('age', 'workclass', 'fnlwgt', 'education', 
                     'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 
                     'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income')

```



```{r}
colnames(adult)
```



**# Finalweight(fnlwgt), education and relationship column were not providing much information so deleted these columns from data set.**
  
  
  
  ```{r}
adult<-adult%>%select(-c('fnlwgt','education','relationship'))%>%
  mutate(adult$income==as.factor(data$income))
head(adult)

```



**# Now , I am doing some exploratory analysis**
  
  
  
  ```{r}
class(adult$income)
```



**# Histogram of age by income group**
  
  
  
  ```{r}
ggplot(adult) + aes(x=as.numeric(age), group=income, fill=income) + 
  geom_histogram(binwidth=5, color='brown')
```



*# Age by income group  plot shows that mostly income was less than 50k. People earned more than 50K income between age 45 and 60.*
  
  
  
  **# Histogram of education_num by income group**
  
  
  
  ```{r}
ggplot(adult) + aes(x=as.numeric(education_num), group=income, fill=income) + 
  geom_histogram(binwidth=5, color='brown')
```



*# This plot shows that education has positive effect on income . Person with higher education tends to earn higher income.*
  
  
  
  **# Histogram of age by gender group**
  
  
  
  ```{r}
ggplot(adult) + aes(x=as.numeric(age), group=sex, fill=sex) + 
  geom_histogram(binwidth=2, color='black')
```



*# This plot shows that majority of the observations make less than $50,000 a year. For those do make over $50,000 annually, they are mainly in midcareer. Interestingly,females are well represented. This could be possibly caused by census bias.*
  
  
  
  **# Performing Logistic Regression**
  
  
  **# Dividing data in Training and Testing Datasets**
  
  
  *# Let's divide adult data set into 80% training and 20% test data set.*
  
  
  
  ```{r}
index<-createDataPartition(adult$age,p=0.80,list = F)
# argument 'list=F' is added so that it takes only indexes of the observations and not make a list row wise
train_adult<-adult[index,]
test_adult<-adult[-index,]
dim(train_adult)
dim(test_adult)
```



**# Binary Logistic Regression model implementation**
  
  
  *# A logistic regression using income as the response variable, and all other 8 variables as predictors is fitted.Its parameter estimates and confidence intervals are reported as below.*
  
  
  
  ```{r}
train_adult$income <-as.factor(train_adult$income)
adult_ml<-glm(income~.,data = train_adult,family = "binomial")
# argument (family = "binomial") is necessary as we are creating a model with dichotomous result
```



```{r}
summary(adult_ml)
```


```{r}
str(adult)
```



*# To check how well is our model built we need to calculate predicted porobabilities, also our calculated probabilities need to be classified. In order to do that we also need to decide the threshold that best classifies our predicted results.*
  
  
  
  ```{r}
train_adult$pred_prob_income<-fitted(adult_ml) 
# this coloumn will have predicted probabilties of being 1
head(train_adult) # run the command to check if the new coloumn is added

```



* # Now, I will predict and plot*
  
  
  
  ```{r}
pred<-prediction(train_adult$pred_prob_income,train_adult$income)
# compares predicted values with actual values in training dataset

perf<-performance(pred,"tpr","fpr")
# stores the measures with respect to which we want to plot the ROC graph

plot(perf,colorize=T,print.cutoffs.at=seq(0.1,by=0.05))
```



*# We assign that threshold where sensitivity and specificity have almost similar values after observing the ROC graph*
  
  
  
  ```{r}
train_adult$pred_income<-ifelse(train_adult$pred_prob_income<0.3,0,1) 
# this coloumn will classify probabilities we calculated and classify them as 0 or 1 based on our threshold value (0.3) and store in this coloumn
head(train_adult)
```



**#Creating Confusion Matrix**
  
  
  
  ```{r}
table(train_adult$income,train_adult$pred_income)

dim(train_adult)
```



```{r}
accuracy<-(4848+16798)/26051;accuracy # formula- (TP+TN)/total possibilities

sensitivity<-4848/(1388+4848);sensitivity # formula TP/(TP+FN)

specificity<-16798/(16798+3017);specificity
```



*# We can conclude that we are getting an accuracy of 83.09% which is good.*
  
  
  **#Decision Tree**
  
  
  *#We need to remove the extra coloumns we added while performing Logrithimic model before implementing Decision tree*
  
  
  
  ```{r}
train_adult$pred_income<-NULL
train_adult$pred_prob_income<-NULL
test_adult$pred_income<-NULL
test_adult$pred_prob_income<-NULL
```


**# Implementing Decision tree**
  
  
  
  ```{r}
library(rpart)
tree_adult_model<-rpart(income~.,data = train_adult)

test_adult$pred_income<-predict(tree_adult_model,test_adult,type = "class")
# an extra argument (type = "class") is required to directly classify prediction into classes

head(test_adult)
```



**# Creating Confusion Matrix**
  
  
  
  ```{r}
table(test_adult$income,test_adult$pred_income)

dim(test_adult)

accuracy<-(4661+839)/6510;accuracy # formula- (TP+TN)/total possibilities

```



*#We are getting an accuracy of 84.49% which is good.*
  
  
  **# Plot the decision tree**
  
  
  
  ```{r}
library(rpart.plot)
rpart.plot(tree_adult_model,cex = 0.6) # cex argument was just to adjust the resolution
```


**# Random forest**
  
  *# In Random Forest the data is not required to split into training and testing*
  
  ```{r}
library(randomForest)
adult$income<-as.factor(adult$income)
rf_adult_model<-randomForest(income~.,data = adult)
rf_adult_model
```



*# Here the Out of Bag error (OOB) gives us the miscalssification rate (MCR) of the model.In this case it comes out to be 13.44%, which gives us the accuracy of 86.56%*
  
  
  *# To check classwise error*
  
  
  
  ```{r}
plot(rf_adult_model)
```



*#Red line represents MCR of class <=50k, green line represents MCR of class >50k and black line represents overall MCR or OOB error. Overall error rate is what we are interested in which seems considerably good.*
  
  
  
  **Conclusion*
  
  *After performing various classification techniques and taking into account their accuracies, I can conclude all the models had an accuracy ranging from 81% to 84%. Out of which Random forest gave a slightly better accuracy of 84.56%*
  
  
  
  
  
  
  
  
  
  
  

