#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd



titanic = pd.read_csv('titanic_train.csv')
print titanic.describe()

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
print titanic.describe()

print titanic['Sex'].unique()
titanic['Sex'] =  titanic['Sex'].map({'male': 0, 'female': 1})
print titanic['Sex'].unique()

print titanic.head()

titanic['Embarked'] = titanic['Embarked'].fillna('S')
print titanic['Embarked'].describe()

