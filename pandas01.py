import pandas as pd
import numpy as np


titanic_survival = pd.read_csv('titanic_train.csv')
print titanic_survival.head(3)

ages =  titanic_survival['Age']
print len([age for age in ages if pd.isnull(age)])

# avg(ages)
print ages.mean()

print pd.pivot_table(titanic_survival, values=['Fare'], index='Pclass')

print titanic_survival.pivot_table(values=['Survived'], index='Pclass', aggfunc=np.mean)

print titanic_survival.pivot_table(values=['Fare', 'Survived'], index='Embarked', aggfunc=np.sum)

print titanic_survival.loc[0, 'Age']

new_titanic_survival = titanic_survival.dropna(axis=0, subset=['Age', 'Sex'])
print new_titanic_survival.loc[0:5]

sorted_titanic_survival = titanic_survival.sort_values(by=['Age'], ascending=False)
sorted_titanic_survival = sorted_titanic_survival.reset_index(drop=True)
print sorted_titanic_survival.head()



# self define function
def count_columns(columns):
    return len([c for c in columns if pd.isnull(c)])

def convert_pclass(row):
    pclass=row['Pclass']
    if pd.isnull(pclass):
        return 'Unknown'
    elif pclass == 1:
        return 'First Class'
    elif pclass == 2:
        return 'Second Class'
    elif pclass == 3:
        return 'Third Class'

print titanic_survival.apply(count_columns)

print titanic_survival.apply(convert_pclass, axis=1).tail()



owner = titanic_survival.pivot_table(values=['Name'], index='Pclass',  aggfunc='count')
owership= titanic_survival.pivot_table(values=['SibSp'], index='Pclass',  aggfunc='sum')
owership2= titanic_survival.pivot_table(values=['Parch'], index='Pclass',  aggfunc='sum')

print type(owner)
print type(owership)

print owner
print owership
print owership2

print owner.add(owership)
