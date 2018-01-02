#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import codecs
import os
reload(sys)
sys.setdefaultencoding('utf-8')

import cPickle as pickle

from sklearn.naive_bayes import MultinomialNB

def _readbunchobj(path):
    path = os.path.expanduser(path)
    with open(path, 'rb') as file_obj:
        content = pickle.load(file_obj)

    return content


# 导入 训练 数据
trainpath = '~/Project/AILab/seg_data/train_word_bag/tfdifspace.dat'
train_set = _readbunchobj(trainpath)
print type(train_set.tdm)
print train_set.tdm.shape
print len(train_set.label)
# 导入 测试 数据
testpath = '~/Project/AILab/seg_data/test_word_bag/testspace.dat'
test_set = _readbunchobj(testpath)
print test_set.tdm.shape
print len(test_set.label)



# 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高
clf = MultinomialNB(alpha=0.001)

clf.fit(train_set.tdm, train_set.label)
predicted = clf.predict(test_set.tdm)

for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
    if flabel != expct_cate:
        print file_name, ": 实际类别:", flabel, " -->预测类别:", expct_cate

print "预测完毕!!!"


# 计算分类精度：
from sklearn import metrics
def metrics_score(actual, predict):
    print '精度{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted'))
    print '召回{0:.3f}'.format(metrics.recall_score(actual, predict, average='weighted'))
    print 'F1-Score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted'))

metrics_score(test_set.label, predicted)