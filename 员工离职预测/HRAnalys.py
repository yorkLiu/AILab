#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
员工离职分析

目标：
 找出员工离职与所给数据的关系

变量说明:
satisfaction_level: 对公司的满意度
last_evaluation: 绩效评估
number_project: 参加过的项目数量
average_montly_hours: 平均每月工作时长
time_spend_company: 工作年限
Work_accident: 是否发生过工作差错
left: 是否已经离职
promotion_last_5years: 5年内是否得到升职
sales: 职业
salary: 薪资水平

"""


import pandas as pd
import seaborn as sns
import numpy as np

np.count_nonzero()
sns.barplot()


df = pd.read_csv('HR_comma_sep.csv')
print df.describe()
