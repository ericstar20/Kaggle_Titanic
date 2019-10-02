import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Titanic_1_clean import clean_data

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def train_logistic_regression(train_x, train_y):
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(train_x, train_y)
    return logistic_regression_model

# Logistic regression model  accuracy calculation
def model_accuracy(trained_model, features, targets):
    """
    Get the accuracy score of the model
    :param trained_model:
    :param features:
    :param targets:
    :return:
    """
    accuracy_score = trained_model.score(features, targets)
    return accuracy_score

def main():
    df = clean_data()
    df.columns
    df.dtypes
    # 1. Data creation for modeling and testing
    training_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare' , 'Embarked']
    target = 'Survived'

    # 2. Train , Test data split
    train_x, test_x, train_y, test_y = train_test_split(df[training_features], df[target], train_size = 0.7)
    '''
    print("train_x size :: ", train_x.shape)
    print("test_x size :: ", test_x.shape)
    print("train_y size :: ", train_y.shape)
    print("test_y size :: ", test_y.shape)
    '''

    # 3. Converting Categorical Features
    sex = pd.get_dummies(train_x['Sex'],drop_first=True)
    embark = pd.get_dummies(train_x['Embarked'],drop_first=True)
    #drop the sex,embarked,name and tickets columns
    train_x.drop(['Sex','Embarked'],axis=1,inplace=True)
    #concatenate new sex and embark column to our train dataframe
    train_x = pd.concat([train_x,sex,embark],axis=1)
    #check the head of dataframe
    train_x.head()

    # covert the testing data
    sex_t = pd.get_dummies(test_x['Sex'],drop_first=True)
    embark_t = pd.get_dummies(test_x['Embarked'],drop_first=True)
    #drop the sex,embarked,name and tickets columns
    test_x.drop(['Sex','Embarked'],axis=1,inplace=True)
    #concatenate new sex and embark column to our train dataframe
    test_x = pd.concat([test_x,sex_t,embark_t],axis=1)
    #check the head of dataframe
    test_x.head()

    # Training Logistic regression model
    trained_logistic_regression_model = train_logistic_regression(train_x, train_y)
    '''
    # Logistic regression model accuracy on train dataset
    train_accuracy = model_accuracy(trained_logistic_regression_model, train_x, train_y)

    # Logistic regressuib model accuracy on test dataset
    test_accuracy = model_accuracy(trained_logistic_regression_model, test_x, test_y)

    print('Train data accuracy : {} \n Test data accuracy : {}'.format(train_accuracy,test_accuracy))
    '''
    return trained_logistic_regression_model
