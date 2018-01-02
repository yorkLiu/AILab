#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import codecs
import os
reload(sys)
sys.setdefaultencoding('utf-8')

import cPickle as pickle

from sklearn.datasets.base import Bunch


# 将中文分词好的文件以  Bunch的格式存到文件中
# Bunch: 是 sklearn.datasets.base中的类，其实就相当于python中的字典。你往里面传什么，它就存什么

# 让我们看看的我们的数据集（训练集）有哪些信息：
## 1. 类别，也就是所有分类类别的集合，即我们./train_corpus_seg/和./test_corpus_seg/下的所有子目录的名字。
###   我们在这里不妨把它叫做target_name（这是一个列表）
## 2. 文本文件名。例如./train_corpus_seg/art/21.txt，我们可以把所有文件名集合在一起做一个列表，叫做filenames
## 3. 文本标签（就是文本的类别），不妨叫做label（与2中的filenames一一对应
###   例如2中的文本“21.txt”在./train_corpus_seg/art/目录下，则它的标签就是art。
###   文本标签与1中的类别区别在于：文本标签集合里面的元素就是1中类别，而文本标签集合的元素是可以重复的，
###   因为./train_corpus_seg/art/目录下有好多文本，不是吗？相应的，1中的类别集合元素显然都是独一无二的类别
## 4. 文本内容（contens）。
###   上一步代码我们已经成功的把文本内容进行了分词，并且去除掉了所有的换行，得到的其实就是一行词袋（词向量）
###   每个文本文件都是一个词向量。这里的文本内容指的就是这些词向量
#
#
# 那么，用Bunch表示，就是：
## from sklearn.datasets.base import Bunch
## bunch = Bunch(target_name=[],label=[],filenames=[],contents=[])


def _readfile(path):
    with codecs.open(path, 'rb', encoding='utf-8', errors='ignore') as fp:
        content = fp.read()

    return content

def corpus2Bunch(wordbag_path, seg_path):
    wordbag_path = os.path.expanduser(wordbag_path)
    seg_path = os.path.expanduser(seg_path)

    wordbag_basename=os.path.basename(wordbag_path)
    wordbag_basepath = wordbag_path.replace(wordbag_basename, '')

    if not os.path.exists(wordbag_basepath):
        os.makedirs(wordbag_basepath)

    catelist = os.listdir(seg_path) # 获取seg_path下的所有子目录，也就是分类信息
    catelist = [dir for dir in catelist if not dir.startswith('.')]

    # 创建一个Bunch实例
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)
    #  extend(addlist)是python list中的函数，意思是用新的list（addlist）去扩充 原来的list

    for mydir in catelist:
        class_path = os.path.join(seg_path, mydir)
        file_list = os.listdir(class_path)

        for file_name in file_list:
            full_name = os.path.join(class_path, file_name)

            bunch.label.append(mydir)
            bunch.filenames.append(full_name)
            bunch.contents.append(_readfile(full_name))

    # 将bunch存储到wordbag_path路径中
    with open(wordbag_path, 'wb') as fp:
        pickle.dump(bunch, fp)

    print 'bunch 构建结束'


if __name__ == '__main__':
    # 对训练集进行Bunch化操作：
    wordbag_path = "~/Project/AILab/seg_data/train_word_bag/train_set.dat"  # Bunch存储路径
    seg_path = "~/Project/AILab/seg_data/fu-dan-chinese-train-corpus-seg/"  # 分词后分类语料库路径
    corpus2Bunch(wordbag_path, seg_path)

    # 对测试集进行Bunch化操作：
    wordbag_path = "~/Project/AILab/seg_data/test_word_bag/test_set.dat"  # Bunch存储路径
    seg_path = "~/Project/AILab/seg_data/fu-dan-chinese-test-corpus-seg/"  # 分词后分类语料库路径
    corpus2Bunch(wordbag_path, seg_path)


    # wordbag_path = os.path.expanduser("~/Project/AILab/seg_data/train_word_bag/train_set.dat")  # Bunch存储路径
    # print _readfile(os.path.expanduser('~/Project/AILab/seg_data/fu-dan-chinese-train-corpus-seg/C3-Art/C3-Art0001.txt'))

    # full_name = os.path.expanduser('~/Project/AILab/seg_data/fu-dan-chinese-train-corpus-seg/C3-Art/C3-Art0001.txt')
    # seg_path = os.path.expanduser(seg_path)
    #
    # bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    # catelist = os.listdir(seg_path)
    # bunch.target_name.extend(catelist)
    # bunch.label.extend('art')
    # bunch.filenames.extend(full_name)
    # bunch.contents.extend(_readfile(full_name))
    #
    # with open(wordbag_path, 'wb') as fp:
    #     pickle.dump(bunch, fp)


    # with open(wordbag_path, 'rb') as fp:
    #      pickle.load(fp)


    # print os.path.basename(os.path.expanduser(wordbag_path))


