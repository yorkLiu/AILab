# -*- coding: UTF-8 -*- 
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost.sklearn import XGBClassifier

# load data
dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

seed = 7

test_size=0.33
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = XGBClassifier()
eval_set = [(x_test, y_test)]
# early_stopping_rounds: 如果连续N 次结果没有提升,则停止
# eval_metric: 损失函数
# eval_set: A list of (X, y) pairs to use as a validation set for early-stopping
# verbose: print 学习结果
model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)

# make predictions for test data
y_pred = model.predict(x_test)
predictions =[round(value) for value in y_pred]

accuracy = metrics.accuracy_score(y_test, predictions)

print "Accuracy: %.2f %%" % (accuracy*100)

