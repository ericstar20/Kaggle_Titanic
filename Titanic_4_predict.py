import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Titanic_1_clean import clean_test
from Titanic_3_model import main

# Using the model to predict the testing dataset
max_model = main()
tt = clean_test()
tt[tt['Fare'].isnull()]
# Converting Categorical Features
sex = pd.get_dummies(tt['Sex'],drop_first=True)
embark = pd.get_dummies(tt['Embarked'],drop_first=True)
#drop the sex,embarked,name and tickets columns
tt.drop(['Sex','Embarked'],axis=1,inplace=True)
#concatenate new sex and embark column to our train dataframe
tt = pd.concat([tt,sex,embark],axis=1)
tt.head()
tt['U']=0

all_X = tt[['Pclass','Age','SibSp','Parch','Fare','male','Q','S','U']]
tt['PassengerId']
# --- checking the null value
c1 = 1
for coll in tt.columns:
    get_null = tt[coll].isnull().sum()/len(tt)
    print('{}.{}: {}'.format(c1,coll,get_null))
    c1 += 1
filename="submission.csv"
predictions = max_model.predict(all_X)
submission = pd.DataFrame({'PassengerId': tt['PassengerId'], 'Survived': predictions})
submission.to_csv(filename, index=False)
