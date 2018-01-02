# -*- coding: gbk -*-
import csv
import pandas as pd
import jieba
from jieba import analyse
from wordcloud import WordCloud
import matplotlib.pyplot as plt

"""
一个更好玩的例子 "基于复旦大学提供的数据集进行中文分类"
http://blog.csdn.net/github_36326955/article/details/54891204
https://github.com/sunxiangguo/chinese_text_classification
"""

df_news = pd.read_table('val.txt', names=['contenttitle', 'url', 'content'])
df_news = df_news.dropna()

content = df_news.content.values.tolist()

content_S = []

for line in content:
    current_segment = jieba.lcut(line)
    if len(current_segment) > 1 and current_segment != '\r\n':
        content_S.append(current_segment)

df_content = pd.DataFrame({'content_S': content_S})
print df_content.head()


#### TF-IDF 提取关键词

def get_tags_with_tf_idf(content):
    tf_idf_content = content
    print tf_idf_content
    tf_idf_words =  analyse.extract_tags(tf_idf_content, topK=5)
    for w in tf_idf_words:
        print w



# with open('stop_words.txt', 'r') as f:
#     lines = [l.strip() for l in f.readlines() if l]
#
# print len(lines)
# open('stop_words01.txt', 'w').write('\t'.join(lines))


def drop_stopwords(contents, stopwords):
    contents_clean = []
    all_words = []
    for word in contents:
        line_clean = []
        if word in stopwords:
            continue
        line_clean.append(word)
        all_words.append(word)


        # line_clean = []
        # for word in line:
        #     print word
        #     if word in stopwords:
        #         continue
        #     line_clean.append(word)
        #     all_words.append(word)

        contents_clean.append(line_clean)
    return contents_clean, all_words


# stopwords = pd.read_csv('stop_words01.txt', sep='\t', quoting=csv.QUOTE_NONE, header=None, names=['stopword'], encoding='utf-8')
stopwords = open('stop_words01.txt', 'r').read().decode('utf-8', 'ignore').split('\t')

orig_content=content_S[4]
contents_clean, all_words = drop_stopwords(orig_content, stopwords)
result =  ''.join([j for i in contents_clean for j in i])
get_tags_with_tf_idf(result)


contents_clean_df = pd.DataFrame({'Content-Clean':contents_clean})
all_words = pd.DataFrame({'AllWord': all_words})


print contents_clean_df.head()
print '--------------------'

words_count = all_words['AllWord'].value_counts()
words_count_keys = words_count.index.tolist()
words_count_values = words_count.values.tolist()



# wordcloud = WordCloud(font_path='SimHei.ttf', max_font_size=40, background_color='white')
# word_frequency = {words_count_keys[i]:words_count_values[i] for i in range(100)}
# # print word_frequency
# wordcloud.fit_words(word_frequency)
#
# plt.figure()
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()

##################
#  gensim
##################
from gensim import corpora, similarities
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel

# gensim Dictionary format is "list of list" (i.e [[1, 2, 3], [4,5,6]...]
dictionary = corpora.Dictionary(contents_clean)

# words to vectors
corpus = [dictionary.doc2bow(text) for text in contents_clean]

lda = LdaModel(corpus = corpus, num_topics=10, id2word=dictionary)


print lda.print_topic(1, 10)
print lda.print_topic(2, 10)
print lda.print_topic(3, 10)
print '--------------------------'
for topic in lda.print_topics(num_topics=10, num_words=5):
    print topic[1]

print '------------tfidf---------'
tfidf = TfidfModel(corpus=corpus)
print tfidf[dictionary.doc2bow([u'中国', u'报道'])]









