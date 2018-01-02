#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@version: python2.7.8
@author: XiangguoSun
@contact: sunxiangguodut@qq.com
@file: TFIDF_space.py
@time: 2017/2/8 11:39
@software: PyCharm
"""
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from sklearn.datasets.base import Bunch
import cPickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def _readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content

def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

def _writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
def vector_space(stopword_path,bunch_path,space_path,train_tfidf_path=None):
    bunch_path = os.path.expanduser(bunch_path)
    space_path = os.path.expanduser(space_path)

    stpwrdlst = _readfile(stopword_path).splitlines()
    bunch = _readbunchobj(bunch_path)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[], vocabulary={})

    if train_tfidf_path is not None:
        trainbunch = _readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)

    else:

        corpus = ["图书 评论 应当 重视 对 书籍装帧 艺术 的 评价",  # 第一类文本切词后的结果，词之间以空格隔开
                            "不难设想 ， 在 这种 缺乏 平等 探讨 的 格局 下",  # 第二类文本的切词结果
                            "小明 硕士 毕业 与 中国 科学院".decode('utf-8'),  # 第三类文本的切词结果
                            "我 爱 北京 天安门".decode('utf-8')]

        # print '--corpus:', corpus
        # print type(corpus)
        # vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        # transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        # print type(bunch.contents)
        # tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

        print '--corpus:', corpus
        print len(bunch.contents)
        contents = bunch.contents[0:100]
        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform(contents)
        print vectorizer.vocabulary_
        print vectorizer.get_feature_names()
        print '---vectorizer:', vectorizer

        # tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        # print vectorizer.get_feature_names()
        # tfidfspace.vocabulary = vectorizer.vocabulary_

    # _writebunchobj(space_path, tfidfspace)
    print "if-idf词向量空间实例创建成功！！！"

if __name__ == '__main__':

    stopword_path = "stop_words.txt"
    bunch_path = "~/Project/AILab/seg_data/train_word_bag/train_set_01.dat"
    space_path = "~/Project/AILab/seg_data/train_word_bag/tfdifspace.dat"
    vector_space(stopword_path,bunch_path,space_path)

    # bunch_path = "test_word_bag/test_set.dat"
    # space_path = "test_word_bag/testspace.dat"
    # train_tfidf_path="train_word_bag/tfdifspace.dat"
    # vector_space(stopword_path,bunch_path,space_path,train_tfidf_path)