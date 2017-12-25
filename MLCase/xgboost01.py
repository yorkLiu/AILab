from numpy import loadtxt
import pandas as pd
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
model.fit(X_train, Y_train)

# make predictions for test data
y_pred = model.predict(x_test)
predictions =[round(value) for value in y_pred]

accuracy = metrics.roc_auc_score(y_test, predictions)

print "Accuracy: %.2f %%" % (accuracy*100)

