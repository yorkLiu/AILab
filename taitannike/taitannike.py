#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd


titanic = pd.read_csv("titanic_train.csv")
print titanic.head()

# Process 'Age'
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
print titanic['Embarked'].describe()

# Process 'Embarked'
titanic['Embarked'] = titanic['Embarked'].fillna('S')

print titanic['Embarked'].unique()

# convert 'Sex'
# male: 0, female: 1
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
print titanic['Sex'].unique()

# convert 'Embarked'
# S: Southampton, C: Cherbourg, Q: Queenstown
# S: 0, C: 1, Q: 2
titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
print titanic['Embarked'].unique()

# 现在我们用 sklearn来进行 训练 学习
# 1. 用 "线性回归"来学习

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import numpy as np

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


# random_state = 1, 每次拿到的都是一样的结果
# shuffle= true, 是否要洗牌，true为要洗牌
kf = KFold(n_splits=3, random_state=1)

predictions = []

alg = LinearRegression()
for train_index, test_index in kf.split(titanic[predictors]):
    train_predictors = titanic[predictors].iloc[train_index, :]
    train_target = titanic['Survived'].iloc[train_index]

    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test_index, :])
    predictions.append(test_predictions)


predictions = np.concatenate(predictions)

predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1

accuracy = metrics.precision_score(titanic['Survived'], predictions)

print accuracy



from sklearn.linear_model import LogisticRegression

alg = LogisticRegression(random_state=1)
scores = cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=3)
print scores.mean()



from sklearn.ensemble import RandomForestClassifier

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
kf = KFold(n_splits=3, random_state=1)

scores = cross_val_score(alg, titanic[predictors], predictions, cv=kf)
print scores
print scores.mean()





# 增加两列, 统计获救人与 '家庭成员' 及 '名字长度' 是否有什么千丝万缕的联系
# FamilySize & NameLength

titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch']
titanic['NameLength'] = titanic['Name'].apply(lambda x:len(x))

print titanic.head()


import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''

titles = titanic['Name'].apply(get_title)
print titles.unique()
print titles.value_counts()

# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k, v in title_mapping.items():
    titles[titles == k] = v
print '------------------- After converted -------------------'
print titles.value_counts()

titanic['Title'] = titles



#统计一下哪些 特征 对我们评估有用，哪些不那么重要

from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'NameLength', 'Title']

selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic['Survived'])

print selector.pvalues_
print np.log10(selector.pvalues_)
scores = -np.log10(selector.pvalues_)
print scores

plt.figure(figsize=(10,8))
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()



predictors = ["Pclass", "Sex", "Fare", "NameLength", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=10, min_samples_leaf=4)
kf = KFold(n_splits=3, random_state=1)

scores = cross_val_score(alg, titanic[predictors], predictions, cv=kf)
print scores.mean()

print """
与上面用 RandomForestClassifier 所预测的结果基本一致
"""

from sklearn.ensemble import GradientBoostingClassifier

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=5), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

kf = KFold(n_splits=3, random_state=1)

predictions = []

GradientBoostingClassifier().predict_proba()

for train_index, test_index in kf.split(titanic):
    train_target = titanic['Survived'].iloc[train_index]
    full_test_predictions = []
    for alg, predictors in algorithms:
        alg.fit(titanic[predictors].iloc[train_index,:], train_target)
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test_index,:].astype('float'))[:,1]
        full_test_predictions.append(test_predictions)

    test_predictions = full_test_predictions[0] + full_test_predictions[1] / 2.0
    test_predictions[test_predictions > .5] = 1
    test_predictions[test_predictions <= .5] = 0
    predictions.append(test_predictions)


predictions = np.concatenate(predictions)
accuracy = metrics.precision_score(titanic['Survived'], predictions)
print accuracy








