# 新闻分类
- 数据清洗
- 分词
- 过滤掉 『停用词』
- 将词 --> 向量

**Note**: 数据源: http://www.sogou.com/labs/resource/ca.php

## TF-IDF 算法提取关键词
- TF: Term Frequency
- IDF: Inverse Document Frequency
- 有关 [TF-IDF 算法原理请看这里](http://www.cnblogs.com/biyeymyhjob/archive/2012/07/17/2595249.html)

## 主题模型
- LSI
- LDA
- HDP
- DTM
- DIM
- 等等....    


## 分词工具
 - **jieba [Fork jieba on GitHub](https://github.com/whtsky/jieba/)**
   ```
     “结巴”中文分词：做最好的 Python 中文分词组件
    "Jieba" (Chinese for "to stutter") Chinese text segmentation: built to be the best Python Chinese word segmentation module.
    
    支持三种分词模式：

    1. 精确模式，试图将句子最精确地切开，适合文本分析；
    2. 全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
    3. 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
    支持繁体分词
    支持自定义词典
    
    MIT 授权协议
    
   ```
   示例:
   ``` 
   pip install jieba
   ```
   ```
    # encoding=utf-8
    import jieba
    
    seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
    print("Full Mode: " + "/ ".join(seg_list))  # 全模式
    
    seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
    print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
    
    seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
    print(", ".join(seg_list))
    
    seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
    print(", ".join(seg_list))
   ```
    输出:
   ```
    【全模式】: 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
    
    【精确模式】: 我/ 来到/ 北京/ 清华大学
    
    【新词识别】：他, 来到, 了, 网易, 杭研, 大厦    (此处，“杭研”并没有在词典中，但是也被Viterbi算法识别出来了)
    
    【搜索引擎模式】： 小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所, 后, 在, 日本, 京都, 大学, 日本京都
    ```
 
 - **[Gensim](https://github.com/piskvorky/gensim)**
 
   Gensim的作者是Radim Řehůřek,一位来自阿拉伯世界的学者。这个作品源于其博士论文《SCALABILITY OF SEMANTIC ANALYSIS IN NATURAL LANGUAGE PROCESSING》，用兴趣的同学可以到谷歌学术上查找看看，这里就不在细说。给定一篇文档，Gensim可以产生一些列与该文档相似的文档集合，这也是作者将其命名为Gensim(gensim = “generate similar”)原因。另外Gensim在Github上地址为:https://github.com/piskvorky/gensim
   
   - Gensim可以做什么？ 根据Gensim的官方API描述，Gensim提供如下函数： 
        - [interfaces](https://radimrehurek.com/gensim/interfaces.html) – Core gensim interfaces
        - [utils](https://radimrehurek.com/gensim/utils.html) – Various utility functions
        - [matutils](https://radimrehurek.com/gensim/matutils.html) – Math utils
        - [corpora.bleicorpus](https://radimrehurek.com/gensim/corpora/bleicorpus.html) – Corpus in Blei’s LDA-C format
        - [corpora.dictionary](https://radimrehurek.com/gensim/corpora/dictionary.html) – Construct word<->id mappings
        - [corpora.hashdictionary](https://radimrehurek.com/gensim/corpora/hashdictionary.html) – Construct word<->id mappings
        - [corpora.lowcorpus](https://radimrehurek.com/gensim/corpora/lowcorpus.html) – Corpus in List-of-Words format
        - [corpora.mmcorpus](https://radimrehurek.com/gensim/corpora/mmcorpus.html) – Corpus in Matrix Market format
        - [corpora.svmlightcorpus](https://radimrehurek.com/gensim/corpora/svmlightcorpus.html) – Corpus in SVMlight format
        - [corpora.wikicorpus](https://radimrehurek.com/gensim/corpora/wikicorpus.html) – Corpus from a Wikipedia dump
        - [corpora.textcorpus](https://radimrehurek.com/gensim/corpora/textcorpus.html) – Building corpora with dictionaries
        - [corpora.ucicorpus](https://radimrehurek.com/gensim/corpora/ucicorpus.html) – Corpus in UCI bag-of-words format
        - [corpora.indexedcorpus](https://radimrehurek.com/gensim/corpora/indexedcorpus.html) – Random access to corpus documents
        - [models.ldamodel](https://radimrehurek.com/gensim/models/ldamodel.html) – Latent Dirichlet Allocation
        - [models.ldamulticore](https://radimrehurek.com/gensim/models/ldamulticore.html) – parallelized Latent Dirichlet Allocation
        - [models.ldamallet](https://radimrehurek.com/gensim/models/ldamallet.html) – Latent Dirichlet Allocation via Mallet
        - [models.lsimodel](https://radimrehurek.com/gensim/models/lsimodel.html) – Latent Semantic Indexing
        - [models.tfidfmodel](https://radimrehurek.com/gensim/models/tfidfmodel.html) – TF-IDF model
        - [models.rpmodel](http://radimrehurek.com/gensim/models/rpmodel.html) – Random Projections
        - [models.hdpmodel](https://radimrehurek.com/gensim/models/hdpmodel.html) – Hierarchical Dirichlet Process
        - [models.logentropy_model](https://radimrehurek.com/gensim/models/logentropy_model.html) – LogEntropy model
        - [models.lsi_dispatcher](https://radimrehurek.com/gensim/models/lsi_dispatcher.html) – Dispatcher for distributed LSI
        - [models.lsi_worker](https://radimrehurek.com/gensim/models/lsi_worker.html) – Worker for distributed LSI
        - [models.lda_dispatcher](https://radimrehurek.com/gensim/models/lda_dispatcher.html) – Dispatcher for distributed LDA
        - [models.lda_worker](https://radimrehurek.com/gensim/models/lda_worker.html) – Worker for distributed LDA
        - [models.word2vec](https://radimrehurek.com/gensim/models/word2vec.html) – Deep learning with word2vec
        - [models.doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html) – Deep learning with paragraph2vec
        - [models.dtmmodel](https://radimrehurek.com/gensim/models/dtmmodel.html) – Dynamic Topic Models (DTM) and Dynamic Influence Models (DIM)
        - [models.phrases](https://radimrehurek.com/gensim/models/phrases.html) – Phrase (collocation) detection
        - [similarities.docsim](https://radimrehurek.com/gensim/similarities/docsim.html) – Document similarity queries 
        - [simserver](https://radimrehurek.com/gensim/similarities/simserver.html) – Document similarity server
        - [How It Works](https://radimrehurek.com/gensim/similarities/docsim.html#how-it-works)
   - 从上述描述我们可以总结出，除了具备基本的语料处理功能外，Gensim还提供了LSI、LDA、HDP、DTM、DIM等主题模型、TF-IDF计算以及当前流行的深度神经网络语言模型word2vec、paragraph2vec等算法，可谓是方便之至。
   
   
 - **[WordCloud](https://github.com/amueller/word_cloud) 构建词云**
   - [Fork WorkCloud on GitHub](https://github.com/amueller/word_cloud)
   
   ```
    pip install wordcloud
   ```
   
   ```
    import wordcloud
   
    words_count = all_words['AllWord'].value_counts()
    words_count_keys = words_count.index.tolist()
    words_count_values = words_count.values.tolist()
    
    
    
    wordcloud = WordCloud(font_path='SimHei.ttf', max_font_size=40, background_color='white')
    word_frequency = {words_count_keys[i]:words_count_values[i] for i in range(100)}
    # print word_frequency
    wordcloud.fit_words(word_frequency)
    
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
   
   ```
   示例代码: [news_C.py](https://github.com/yorkLiu/AILab/blob/master/newsCategory/news_C.py), 运行效果如下图:
   更多关于 [wordcloud示例请参见官方](https://github.com/amueller/word_cloud)
   ![Screenshot](wordcloud-01.png)
   
  
   
   
   