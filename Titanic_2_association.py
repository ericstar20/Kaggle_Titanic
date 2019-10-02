from Titanic_1_clean import clean_data
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df = clean_data()
df.Embarked.unique()
# 1. find whether the association between Survived (responcse value)
#    and other variables (predicted variables) exist
df.dtypes


# Survived = binary categorical
## a. Pclass - ordinal categorical -> significant
worksurv_by_cls = df[['Survived','Pclass']]
worksurv_by_cls['Survived'].value_counts()
worksurv_by_cls['Pclass'].value_counts()

# Constructing the Contingency Table
contingency_table = pd.crosstab(
    worksurv_by_cls['Survived'],
    worksurv_by_cls['Pclass'],
    margins = True)

# Visualizing the Contingency Table with a Stacked Bar Chart
# Assigns the frequency values
died_c = contingency_table.iloc[0][0:3].values
surv_c = contingency_table.iloc[1][0:3].values

# Plots the bar chart
fig = plt.figure(figsize=(10, 5))
sns.set(font_scale=1.8)
categories = ["1st","2nd","3rd"]
p1 = plt.bar(categories, died_c, 0.55, color='#d62728')
p2 = plt.bar(categories, surv_c, 0.55, bottom=died_c)
plt.legend((p2[0], p1[0]), ('Survived', 'Dead'))
plt.xlabel('The Class level')
plt.ylabel('Number of Survived')
plt.show()

# Calculate Chi-square
f_obs = np.array([contingency_table.iloc[0][0:3].values, contingency_table.iloc[1][0:3].values])
stats.chi2_contingency(f_obs)[0:3]
# since the p-value is low, we reject the null hypothesis,
# There are a association between survival and class.
# ** (chi-square, p-value, degree of freedom)

## b. Sex - binary categorical -> significant
worksurv_by_sex = df[['Survived','Sex']]
worksurv_by_sex['Survived'].value_counts()
worksurv_by_sex['Sex'].value_counts()

# Constructing the Contingency Table
contingency_table_sex = pd.crosstab(
    worksurv_by_sex['Survived'],
    worksurv_by_sex['Sex'],
    margins = True)

# Visualizing the Contingency Table with a Stacked Bar Chart
# Assigns the frequency values
died_c = contingency_table_sex.iloc[0][0:2].values
surv_c = contingency_table_sex.iloc[1][0:2].values

# Plots the bar chart
fig = plt.figure(figsize=(10, 5))
sns.set(font_scale=1.8)
categories = ["female","male"]
p1 = plt.bar(categories, died_c, 0.55, color='#d62728')
p2 = plt.bar(categories, surv_c, 0.55, bottom=died_c)
plt.legend((p2[0], p1[0]), ('Survived', 'Dead'))
plt.xlabel('Gender')
plt.ylabel('Number of Survived')
plt.show()

# Calculate Chi-square
f_obs = np.array([contingency_table_sex.iloc[0][0:2].values, contingency_table_sex.iloc[1][0:2].values])
stats.chi2_contingency(f_obs)[0:2]
# p-value = 1.1973570627755645e-58

## c. Age   - numeric continuous -> association exist
sns.countplot(x="Age", data=df)
plt.show()


## d. SibSp - numeric continuous -> association exist
sns.countplot(x="SibSp", data=df)
plt.show()
## e. Parch - numeric continuous -> association exist
sns.countplot(x="Parch", data=df)
plt.show()
## f. Fare  - numeric continuous
cut_points = [-1,12,50,100,1000]
label_names = ["0-12","12-50","50-100","100+"]
Fare_categories = pd.cut(df["Fare"],cut_points,labels=label_names)

sns.countplot(Fare_categories, hue=df['Survived'])
plt.xlabel('')
plt.show()
print(df['Survived'].groupby(Fare_categories).mean().sort_values())

## g. Embarked - nominal categorical -> significant
worksurv_by_em = df[['Survived','Embarked']]
worksurv_by_em['Survived'].value_counts()
worksurv_by_em['Embarked'].value_counts()

# Constructing the Contingency Table
contingency_table_em = pd.crosstab(
    worksurv_by_em['Survived'],
    worksurv_by_em['Embarked'],
    margins = True)

# Visualizing the Contingency Table with a Stacked Bar Chart
# Assigns the frequency values
died_c = contingency_table_em.iloc[0][0:4].values
surv_c = contingency_table_em.iloc[1][0:4].values

# Plots the bar chart
fig = plt.figure(figsize=(10, 5))
sns.set(font_scale=1.8)
categories = ["Cherbourg","Queenstown","Southampton","Unknown"]
p1 = plt.bar(categories, died_c, 0.55, color='#d62728')
p2 = plt.bar(categories, surv_c, 0.55, bottom=died_c)
plt.legend((p2[0], p1[0]), ('Survived', 'Dead'))
plt.xlabel('Embarked')
plt.ylabel('Number of Survived')
plt.show()

# Calculate Chi-square
f_obs = np.array([contingency_table_sex.iloc[0][0:4].values, contingency_table_sex.iloc[1][0:4].values])
print(stats.chi2_contingency(f_obs)[0:4])
# p-value = 7.573447336653259e-58
