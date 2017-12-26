# -*- coding: UTF-8 -*-
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np


#信用卡欺诈检测
# creditcard.csv download url: https://clouda-datasets.s3.amazonaws.com/creditcard.csv.zip
# 如果打不开,也可以在百度盘中下载: https://pan.baidu.com/s/1pLKGzQN

# 对于样本非常不均衡时，有两种方案: 1. 下采样,使得样本量一样的少 2. 过采样, 使得样本量一样多
#
#
#
#
#

data = pd.read_csv('creditcard.csv')

# print data.head()
#
#
# print data['Class'].value_counts()
#
#
# # step 1
# count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
# count_classes.plot(kind='bar')
# plt.title('Fraud class histogram')
# plt.xlabel('Class')
# plt.ylabel('Frequency')
# plt.show()

# step 2
# using sklearn StandardScaler process the data

# data['normalAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
# data.drop(['Time', 'Amount'], axis=1)
# print data.head()


# step 3
# 下采样: 使所有的样本数量都同样的少

X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']


number_records_fraud = len(data[data['Class']==1])
fraud_indices = np.array(data[data['Class']==1].index)

normal_indices = np.array(data[data['Class']==0].index)


random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)

under_sample_indices = np.concatenate([random_normal_indices, fraud_indices])


under_sample_data = data.iloc[under_sample_indices,:]

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

print 'Percentage of normal transactions:', len(under_sample_data[under_sample_data.Class == 0]) / float(len(under_sample_data))
print 'Percentage of fraud transactions: ', len(under_sample_data[under_sample_data.Class == 1]) / float(len(under_sample_data))
print 'Total numbers of transactions in reshape samples:', len(under_sample_data)
