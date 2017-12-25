# -*- coding: UTF-8 -*-
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from sklearn import metrics
from sklearn.datasets import make_hastie_10_2
from xgboost.sklearn import XGBClassifier
import xgboost as xgb

#
X, y = make_hastie_10_2(random_state=0)
X = DataFrame(X)
y = DataFrame(y)

y.columns= {"label"}

label = {-1: 0, 1:1}
y.label = y.label.map(label)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# XGBoost 自带接口
params = {
    'eta': 0.3,
    'max_depth': 3,
    'min_child_weight': 1,
    'gamma': 0.3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'nthread': 12,
    'scale_pos_weight': 1,
    'lambda': 1,
    'seed': 27,
    'silent': 0,
    'eval_metric': 'auc'
}

d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_test, label=y_test)
d_test = xgb.DMatrix(X_test)

watchlist = [(d_train, "train"), (d_valid, 'valid')]

clf = XGBClassifier()
model_bst = xgb.train(params, d_train, 30, evals=watchlist, early_stopping_rounds=500, verbose_eval=10)
model_sklearn = clf.fit(X_train, y_train)

y_bst = model_bst.predict(d_test)
y_sklearn = model_sklearn.predict_proba(X_test)[:,1]

print("XGBoost_自带接口    AUC Score : %f" % metrics.roc_auc_score(y_test, y_bst))
print("XGBoost_sklearn接口 AUC Score : %f" % metrics.roc_auc_score(y_test, y_sklearn))

print "原始train大小:" , X_train.shape
print "原始test大小:" , X_test.shape

# XGBoost 自带接口生成的新特征
train_new_feature = model_bst.predict(d_train, pred_leaf=True)
test_new_feature = model_bst.predict(d_test, pred_leaf=True)
train_new_feature1 = DataFrame(train_new_feature)
test_new_feature1 = DataFrame(test_new_feature)

print "新的特征集(自带接口):", train_new_feature1.shape
print "新的测试集(自带接口):", test_new_feature1.shape


# sklearn接口生成的新特征
train_sklearn_new_feature = clf.apply(X_train)#每个样本在每颗树叶子节点的索引值
test_sklearn_new_feature = clf.apply(X_test)

train_new_feature2 = DataFrame(train_sklearn_new_feature)
test_new_feature2 = DataFrame(test_sklearn_new_feature)

print "新的特征集(sklearn接口):", train_new_feature2.shape
print "新的测试集(sklearn自带接口):", test_new_feature2.shape

#用XGBoost自带接口生成的新特征训练
new_feature1=clf.fit(train_new_feature1, y_train)
y_new_feature1= clf.predict_proba(test_new_feature1)[:,1]
#用XGBoost自带接口生成的新特征训练
new_feature2=clf.fit(train_new_feature2, y_train)
y_new_feature2= clf.predict_proba(test_new_feature2)[:,1]

print("XGBoost自带接口生成的新特征预测结果 AUC Score : %f" % metrics.roc_auc_score(y_test, y_new_feature1))
print("XGBoost自带接口生成的新特征预测结果 AUC Score : %f" % metrics.roc_auc_score(y_test, y_new_feature2))


from xgboost import plot_tree
from xgboost import plot_importance
import matplotlib.pyplot as plt
from graphviz import Digraph
import pydot

#安装说明：
#pip install pydot
#http://www.graphviz.org/Download_windows.php
#先安装上面下载的graphviz.msi，安装，然后把路径添加到环境变量，重启下
#然后pip3 install graphviz...竟然就可以了...

# macos: brew install graphviz


plot_tree(model_bst, num_trees=0)
plot_importance(model_bst)

plt.show()

plot_tree(model_sklearn, num_trees=0)
plot_importance(model_sklearn)






