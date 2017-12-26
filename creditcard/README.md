# 信用卡欺诈检测

## 数据下载
示例中的上传的creditcard.csv 只有2W条数据, 如果要更多的数据请按以下方式下载
- [Download creditcard.csv in Amazon S3](https://clouda-datasets.s3.amazonaws.com/creditcard.csv.zip)
- [Download creditcard.csv in 百度盘](https://pan.baidu.com/s/1pLKGzQN)

```
    # -*- coding: UTF-8 -*-
    import pandas as pd
    import matplotlib.pyplot as plt
    data = pd.read_csv('creditcard.csv')
    
    print data['Class'].value_counts()
    
    count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
    count_classes.plot(kind='bar')
    plt.title('Fraud class histogram')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()
```
如下图所示，0与1的样本数量相差巨大
<div class="row">
<img src="https://github.com/yorkLiu/AILab/blob/master/creditcard/creditcard-anays.png">
</div>