import pandas as pd
import numpy as np

df = pd.read_csv('/Applications/詩典修煉手冊/Personal Projects/Kaggle_1_Titanic/titanic/train.csv')
df.head

# 1. Drop the unnecessary columns
to_drop = ['Name','Ticket']
df.drop(columns=to_drop, inplace=True)
df.head

# 2. Changing the Index of a DataFrame
#df['PassengerId'].is_unique
#df = df.set_index('PassengerId')
df.head
#df.loc[1]

# 3. Tidying up Fields in the Data
df.get_dtype_counts()
## 1. age
df.loc[1:, 'Age'].head(10)
df['Age'].isnull().sum()/len(df)
# there are 20% of null value in cell age

## 1-a. Write a mean number into null number.
gender_mean = df.groupby('Sex').Age.mean()
f_mean = gender_mean[0]
m_mean = gender_mean[1]
df.loc[df['Sex'] == 'female', 'Age'] = f_mean
df.loc[df['Sex'] == 'male', 'Age'] = m_mean
df['Age'].isnull().sum()/len(df)

## 2. cabin
df.loc[1:, 'Cabin'].head(10)
df['Cabin'].isnull().sum()/len(df)
# there are 77% of null value in cell cabin
# we will drop this cabin cell since we cannot interpret the meaning
df.drop(columns='Cabin', inplace=True)
df.head

## 3. embarked
df.Embarked.unique()
df['Embarked'] = df['Embarked'].fillna('U')
df['Embarked'].isnull().sum()/len(df)

# move the cleaning DataFrame to the next step
def clean_data():
    return df

#---------- do the same thing at the testing data ------------#
tt = pd.read_csv('/Applications/詩典修煉手冊/Personal Projects/Kaggle_1_Titanic/titanic/test.csv')
tt.head

# 1. Drop the unnecessary columns
to_drop = ['Name','Ticket']
tt.drop(columns=to_drop, inplace=True)
tt.head

# 2. Changing the Index of a DataFrame
#tt['PassengerId'].is_unique
#tt = tt.set_index('PassengerId')

# 3. Tidying up Fields in the Data
tt.get_dtype_counts()
## 1. age
tt.loc[1:, 'Age'].head(10)
tt['Age'].isnull().sum()/len(tt)
# there are 20% of null value in cell age

## 1-a. Write a mean number into null number.
gender_mean = tt.groupby('Sex').Age.mean()
f_mean = gender_mean[0]
m_mean = gender_mean[1]
tt.loc[tt['Sex'] == 'female', 'Age'] = f_mean
tt.loc[tt['Sex'] == 'male', 'Age'] = m_mean
tt['Age'].isnull().sum()/len(tt)

## 2. cabin
tt.loc[1:, 'Cabin'].head(10)
tt['Cabin'].isnull().sum()/len(tt)
# there are 77% of null value in cell cabin
# we will drop this cabin cell since we cannot interpret the meaning
tt.drop(columns='Cabin', inplace=True)
tt.head

## 3. embarked
tt.Embarked.unique()
tt['Embarked'] = tt['Embarked'].fillna('U')
tt['Embarked'].isnull().sum()/len(tt)

## 4. fare
tt[tt['Fare'].isnull()] # the nall value is 3 class
avg_class = tt.groupby('Pclass').Fare.mean()
fare_3 = avg_class[3]
tt['Fare'] = tt['Fare'].fillna(fare_3)
tt.Fare.isnull().sum()

def clean_test():
    return tt
