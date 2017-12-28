# -*- coding: UTF-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score


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

data['normalAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data.drop(['Time', 'Amount'], axis=1)
print data.head()


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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print "Number transaction of train dataset:", len(X_train)
print "Number transaction of test dataset:", len(X_test)
print "Total numbers:", len(X_train) + len(X_test)

X_undersample_train, X_undersample_test, y_undersample_train, y_undersample_test = train_test_split(X_undersample, y_undersample,
                                                                                                    test_size=0.3, random_state=0)

print "Number under sample transaction of train dataset:", len(X_undersample_train)
print "Number under sample transaction of test dataset:", len(X_undersample_test)
print "Total number of under sample:", len(X_undersample_train) + len(X_undersample_test)


#交差验证

def print_kfold_scores(x_train_data, y_train_data):
    fold = KFold(len(y_train_data), 5, shuffle=False)
    # kfold = KFold(5, shuffle=False)


    # KFold.split will return train data array and test data array
    # fold = kfold.split(y_train_data)

    # 惩罚系数
    # 数值越小,惩罚力度就越小, 数值越大, 惩罚力度就越大
    c_param_range=[0.01, 0.1, 1, 10, 100]

    result_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_Parameter', 'Mean Recall Score'])
    result_table['C_Parameter'] = c_param_range

    j = 0
    for c_param in c_param_range:
        print "-----------------------------"
        print "C Parameter: ", c_param
        print "-----------------------------"

        recall_accs = []
        for iteration, indices in enumerate(fold, start=1):
            lr = LogisticRegression(C=c_param, penalty='l1')
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())

            y_pred = lr.predict(x_train_data.iloc[indices[1], :].values)

            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred)
            recall_accs.append(recall_acc)
            print ("Iteration:", iteration, ": recall score = ", recall_acc)

        result_table.ix[j, 'Mean Recall Score'] = np.mean(recall_accs)
        j += 1

        print '\nMean recall score:', np.mean(recall_accs)

    # j = 0
    # for c_param in c_param_range:
    #     print "-----------------------------"
    #     print "C Parameter: ", c_param
    #     print "-----------------------------"
    #
    #     recall_accs = []
    #     for iteration, indices in enumerate(fold, start=1):
    #         lr = LogisticRegression(C=c_param, penalty='l1')
    #         lr.fit(x_train_data.iloc[indices[0],:], y_train_data.iloc[indices[0],:].values.ravel())
    #
    #         y_pred = lr.predict(x_train_data.iloc[indices[1],:].values)
    #
    #         recall_acc = recall_score(y_train_data.iloc[indices[1],:].values, y_pred)
    #         recall_accs.append(recall_acc)
    #         print ("Iteration:", iteration, ": recall score = ", recall_acc)
    #
    #     result_table.ix[j, 'Mean Recall Score'] = np.mean(recall_accs)
    #     j+=1
    #
    #     print '\nMean recall score:', np.mean(recall_accs)


    best_c = result_table.loc[result_table['Mean Recall Score'].idxmax()]['C_Parameter']
    print ('*****************************************************************************')
    print ('The best C Parameter is: ', best_c)
    print ('*****************************************************************************')





print_kfold_scores(X_undersample_train, y_undersample_train)

