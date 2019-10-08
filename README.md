# Kaggle - Titanic
Machine Learning from Disaster

## Getting Started
I used four steps to achieve this submission.
* 1. Clean the dataFirst, I found the columns 'Age' and 'Embarked' have a null value in the dataset. The method I used to deal with 'Age' label is found the mean age for men and women separately. Then, if the passenger lack of 'Age' value, I can assign the value by gender.For the 'Embarked' label, I assigned the 'U' to the null value directly. ('U' means unknown).
* 2. Find whether the association between 'Survived' and other columnsIn this step, I used the Chi-square test to determine whether the association exist between two categorical data or not. On the other hand, I used the logistic regression to find the association between categorical data and continuous data.
* 3. Build the modelAfter step 2, I can determine the columns which are significant to the dependent variable (Survived). The significant labels are 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare' , 'Embarked'. The model accuracy is approximately 76%.
* 4. Predict the test dataFinally, I got the model to estimate the testing data. The result is as the submission file.

## Result
<img src = "Titanic%20-%20My%20ranking%20.png" width='900' heigh='600'>
