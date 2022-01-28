# Practical Machine Learning Project


---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Objective and Contents : 

The aim of this project is to predict the 'classe' variable, of doing the exercice for each partcicpant.
The data used is collected from accelerometers on the belt, forearm, arm, and dumbell.
In this project we will proceed through the following steps : 

   1. Preprocessing the data Exploratory data Analysis
   2. Cross validation, training/test split & Exploratory data analysis
   3. Model fitting, Evaluation & model selection
   4. Predictions on the final test set

In steps 1 to 3, we will be using only the training data : 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'.

We will keep the test data : 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
apart until step 4 (Predictions). Let's start !


load the data and packages : 
```{r }
suppressMessages({
library(caret)
library(ggplot2)
library(skimr)
library(dplyr)
})
data = read.csv("pml-training.csv")
dim(data)
```


## 1. Preprocessing the data

We execute the skim function that gives the summary of the columns by type (character and numeric). We have 37 character columns and 123 numeric ones.
At first glance on the summary (we won't print the complete summary because it takes a lot of space. Instead we will load the summary into a data frame and explore it's columns),  

```{r }
data_summary = skim(data)  ###-gives a complete and very useful summary of the data
print("Informations given by the skim functions") ;names(data_summary)
print("Number of columns by type") ;table(data_summary$skim_type)
print("Number of missings in numeric columns") ;table(data_summary$n_missing)
print("Number of empty cells in character columns") ;table(data_summary$character.empty)
```
The first result displayed above, shows us the colnames of our summary data (different sets of information like the type of the column : "skim_type", the name of the column : "skim_variable", and summaries of numerical variables). We have 37 character variables and 123 numerical one.
the columns n_missings and character.empty contains the number of missing or empty values for numeric or character columns. As we can see above, we have 67 numeric columns containing 19216 missing values and 33 character columns with the same number of empty cells. These columns can be removed without significant information loss !


Below, we display a subset of the summary (only the 40 first columns to have an example display of the function).
```{r}
data_summary[1:40,] ### the skimed summary of the 40 first columns
```




Removing the columns with high missing rate :
```{r}
maxNA = round(3*nrow(data)/4)   ##-set max NA authorized to be 75% of the data
##-split the data into numeric/character
nmiss = 1:ncol(data)
for( i in 1 : length(nmiss)) {
   nmiss[i]  = length(which(is.na(data[,i]))) + 
      length(which(data[,i] == ""))
}
names_miss = names(data[,which(nmiss >= maxNA)])
complete_data  = data %>% select(-all_of(names_miss))
dim(complete_data)   ####-60 columns with no high missing values
```
We can re execute skim(complete_data) and see that there are no missing or empty values in complete_data.
there are 56 numeric variables and 4 character one (the 'classe' variable + 3 others).

```{r}
###-get rid of non interesting variables
nzv_columns = nearZeroVar(complete_data, names = TRUE)
complete_data2 = complete_data %>%
   select(- all_of(nzv_columns) ) %>%  ###-get rid of near zero variance column
   select(-c("X"))                     ###-get rid of indexes
```



## 2. Cross validation, training/test split & Exploratory data analysis

in this section we split our processed training data into new train/test sets. This cross validation will allows us to validate the model before predicting on the final test set:

```{r}
set.seed(12354)
index = createDataPartition(y = complete_data2$classe, p=0.75, list = FALSE)
train = complete_data2[index, ] ; test = complete_data2[-index, ]
dim(train) ; dim(test)
```

We then proceed to some exploratory data analysis on the training set in order to comprehend our features :
charactar features, numeric features, features engineering on train/test sets

```{r}
####-character features
round((table(train$classe)/nrow(train))*100)  
```
The classe A is 28% percent of the data, the other classes are aound 18%. The data is quite balanced around the outcome variable. The 2 other categorical variables are names and cvtd_timestamp. We won't be interested to these variables. We will focus on the numerical variables.

```{r}
###-check correlation with the classe variable
correlations_data = data.frame(cor(train[,-c(1,4,58)], as.numeric(as.factor(train$classe))) ); colnames(correlations_data) = "correlations"
corr = {correlations_data %>%
  mutate(variables = rownames(correlations_data),
         corr_pos = abs(correlations)) %>%
  arrange(desc(corr_pos)) %>%
  select("variables", "corr_pos")}[1:5,]
  
corr
```

The correlations between our class variable and the numeric features are not very high. the highly correlated one (as we can see above) is "pitch_forearm" with only 0.34 correlation. the top 5 correlated features are the only one with correlation above 0.2. let's visualize the relation between our outcome variable and "pitch_forearm" (the highest correlated feature).

```{r}
###we first plot the distribution of the highest correlated feature by classe
###the we plot the second correlated feature vs the third one and color by classe 
suppressMessages(library(gridExtra))
p1 = qplot(data = train, classe, pitch_forearm, fill = classe ) +
  geom_boxplot()
p2 = qplot(data = train, magnet_arm_x, magnet_belt_y ,  colour = classe)
grid.arrange(p1,p2, ncol = 2)
```
from the two graphs, we can see that only a slight relation can be seen between pitch_forearm and classe (left side multi boxplot) and in the second graph we can't really detect any pattern by plotting magnet_arm_x vs magnet_belt_y points and color by classe. Thus we will have to use a model in order to understand the relation between "classe" and the features.


But before, let's check the correlation between our numeric variables
```{r}
suppressMessages(library(corrplot))
train_num = train %>% select_if(sapply(train,  is.numeric))%>%
  mutate( "classe" = as.factor(train$classe))
test_num = test %>% select_if(sapply(test,  is.numeric))%>%
  mutate( "classe" = as.factor(test$classe))
corr = cor(train_num[,-56])  ###-correlation matrix of our numeric features
corrplot(corr, method = "circle", type = "lower" ,  tl.cex = 0.55, tl.col = 'black',  tl.srt = 0.2,
         order = "hclust"
)
```

From the correlation plot above, we can see that several ones of our features are highly correlated with each others (the dark red and blue circles). We will use a PCA to reduce our data dimensions (usefull when having a lot of numeric and correlated features).

### PCA

we will create a new training data with the classe outcome and the principal components 

```{r}
set.seed(123522)
pca = preProcess(train_num[,-56], method = "pca")
train_pca = predict(pca, newdata = train_num[,-56])
train_pca = train_pca %>%
  mutate("classe" = as.factor(train_num$classe))
test_pca = predict(pca, newdata = test_num[, - 56])
test_pca = test_pca %>%
  mutate("classe" = as.factor(test_num$classe))
```

We have created two new training and test data based on a pca preprocessing.
The idea is to use this data to fit, evaluate a model and compare with model fitted on the data before pca processing.


## 3. Model fitting, Evaluation & model selection


we will fit several models and select our best one.
let's start with a simple decision tree model, fit it on the training data and compare with the fitting on the preprocessed pca data "train_pca" :

```{r}
set.seed(12354)
suppressMessages(library("rpart"))
tree = rpart(classe~., data = train_num, method = "class")
pred_tree = predict(tree, newdata = test_num, type = "class")
confusionMatrix(pred_tree, test_num$classe)  ###-accuracy on the test set = 84%
```
fit on the pca training table :

```{r}
set.seed(12354)
tree_pca = rpart(classe~., data = train_pca, method = "class")
pred_tree_pca = predict(tree_pca, newdata = test_pca, type = "class")
confusionMatrix(pred_tree_pca, test_pca$classe)  ###-accuracy on the test set = 52%
```
the decision tree with pca is less accurate than the simple one. So the use of PCA won't be useful in improving our model. We will continue with our simple decision tree model and compare with a random forest model.


```{r}
suppressMessages(library(randomForest))
set.seed(123654)
rf = randomForest(classe~., data = train_num)
pred_rf = predict(rf, newdata = test_num)
confusionMatrix(pred_rf, test_num$classe) 
```


We have a very high accuracy when predicting with a random forest model. We will choose this model to predict on the final test set. 

### The OOB error of the model

the Out of bag error of our model is 0.13% error rate :
```{r}
print(rf)
```



## 4. Predictions on the final test set


```{r}
final_test = read.csv("pml-testing.csv")
dim(final_test)
```
Process the final test data and apply the prediction model : 

```{r}
predictions_data = final_test %>%
  select("user_name", "problem_id")
final_test  = final_test %>% select(-all_of(names_miss))   ###-exclude NA variables
final_test = final_test %>%
   select(- all_of(nzv_columns) ) %>%  ###-get rid of near zero variance column
   select(-c("X"))                     ###-get rid of indexes
final_test = final_test %>% select_if(sapply(final_test,  is.numeric))%>%
  select(-"problem_id")
final_predictions = predict(rf, newdata = final_test)
predictions_data = cbind(predictions_data, final_predictions)
```

