#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import codecs
import os
import jieba
# 配置utf-8输出环境
reload(sys)
sys.setdefaultencoding('utf-8')


# 对训练集和测试集进行分词
# 并存储到指定的目录

__all__ = ['readfile', 'savefile', 'corpus_segment']


def savefile(savepath, content):
    with codecs.open(savepath, 'wb', encoding='utf-8') as f:
        f.write(content)

def readfile(path):
    with open(path, 'rb') as fp:
    # with codecs.open(path, 'r', encoding='utf-8') as fp:
        content = fp.read()

    return content


def corpus_segment(corpus_path, seg_path):
    """
    :param corpus_path: 未分词语料库路径
    :param seg_path: 分词后语料库存储路径
    :return:
    """

    corpus_path = os.path.expanduser(corpus_path)
    seg_path = os.path.expanduser(seg_path)

    catelist = os.listdir(corpus_path) # 获取corpus_path下的所有子目录

    # 其中子目录的名字就是类别名，例如: train_corpus/art/21.txt中，'train_corpus/'是corpus_path，'art'是catelist中的一个成员
    total = len(catelist)
    for index, mydir in enumerate(catelist, start=1):
        # 这里mydir就是train_corpus/art/21.txt中的art（即catelist中的一个类别）

        class_path = os.path.join(corpus_path, mydir) # 拼出分类子目录的路径如：train_corpus/art/
        seg_dir = os.path.join(seg_path, mydir)  # 拼出分词后存贮的对应目录路径如：train_corpus_seg/art/

        # 是否存在分词目录，如果没有则创建该目录
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)

        # 获取未分词语料库中某一类别中的所有文本
        file_list = os.listdir(class_path)
        # train_corpus/art/中的 21.txt, 22.txt,  23.txt
        # .... file_list=['21.txt','22.txt',...]

        for file_path in file_list: # 遍历类别目录下的所有文件

            fullpath = os.path.join(class_path, file_path) # 拼出文件名全路径如：train_corpus/art/21.txt
            save_file_name = os.path.join(seg_dir, file_path)  # 保存文件名
            content = readfile(fullpath) # 读取文件内容
            # 此时，content里面存贮的是原文本的所有字符，例如多余的空格、空行、回车等等，
            # 接下来，我们需要把这些无关痛痒的字符统统去掉，变成只有标点符号做间隔的紧凑的文本内容

            content = content.replace('\r\n', '')
            content = content.replace(' ', '')
            content_seg = jieba.cut(content) #为文件内容分词
            savefile(save_file_name, ' '.join(content_seg)) # 将处理后的文件保存到分词后语料目录

        print ('%s 目录分词结束, index[%s/%s]' % (mydir, index, total))

    print '中文语料分词结束！！！'



if __name__ == '__main__':
    # 对训练集进行分词
    corpus_dir='~/Project/AILab/corpus_data/fu-dan-chinese-corpus'
    seg_dir='~/Project/AILab/seg_data/fu-dan-chinese-train-corpus-seg'
    corpus_segment(corpus_dir, seg_dir)

    # #对测试集进行分词
    test_corpus_dir='~/Project/AILab/corpus_data/fu-dan-chinese-test-corpus'
    test_seg_dir='~/Project/AILab/seg_data/fu-dan-chinese-test-corpus-seg'
    corpus_segment(test_corpus_dir, test_seg_dir)