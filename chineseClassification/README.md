中文 文本分类
====
本示例主要基于 **复旦中文文本分类语料库** 
- [戳我下载 训练数据集](https://pan.baidu.com/s/1gfNEQNx)
- [戳我下载 测试数据集](https://pan.baidu.com/s/1bplTvo7)


## 中文文本分类流程
- 1. 预处理
    - 1. 得到训练集语料库: 即已经分好类的文本资料（例如：语料库里是一系列txt文章，这些文章按照主题归入到不同分类的目录中，如 .\art\21.txt）
    - 2. 得到测试语料库: 也是已经分好类的文本资料，与a类型相同，只是里面的文档不同，用于检测算法的实际效果。测试预料可以从a中的训练预料中随机抽取，也可以下载独立的测试语料库
- 2. 中文分词
    - 1. 中文分词的实际操作
    - 2. 中文分词的算法
- 3. 结构化表示--构建词向量空间: 将中文转化成向量
- 4. 权重策略--TF-IDF
- 5. 分类器
- 6. 评价


## 代码
```
pip install jieba, sciking-learn
```
如果是要去爬取网页上的内容，以下代码可以很方面的提取文本
```
#!/usr/bin/env python  
# -*- coding: UTF-8 -*-  

import sys  
from lxml import html  
reload(sys)  
sys.setdefaultencoding('utf-8')  
  
def html2txt(path):  
    with open(path,"rb") as f:  
        content=f.read()   
    r''''' 
    上面两行是python2.6以上版本增加的语法，省略了繁琐的文件close和try操作 
    2.5版本需要from __future__ import with_statement 
    新手可以参考这个链接来学习http://zhoutall.com/archives/325 
    '''  
    page = html.document_fromstring(content) # 解析文件  
    text = page.text_content() # 去除所有标签  
    return text  
  
if __name__  =="__main__":  
    # htm文件路径，以及读取文件  
    path = "1.htm"  
    text=html2txt(path)  
    print text   # 输出去除标签后解析结果  
```

- 1.中文分词示例代码 [corpus_segment.py](corpus_segment.py)
- 2.将中文分词以SkLearn Bunch格式存储 [corpus2Bunch.py](corpus2Bunch.py)
- 3.将中文分词转换成向量 [vector_sapce.py](vector_sapce.py)
- 4.预测 -  使用 **贝叶斯**进行分类 [NBayes_Predict.py](NBayes_Predict.py)

**Note** 详细教程请参见 [相国大人博客](http://blog.csdn.net/github_36326955/article/details/54891204)