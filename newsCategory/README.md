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

## 分词工具
 - jieba [Fork jieba on GitHub](https://github.com/whtsky/jieba/)
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
 - WordCloud 构建词云
   - [Fork WorkCloud on GitHub](https://github.com/amueller/word_cloud)
   
   ```
    pip install wordcloud
   
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
   示例代码: [news_C.py]
   运行效果如下图:
   ![Screenshot](https://github.com/yorkLiu/AILab/tree/master/newsCategory/wordcloud-01.png)
   
   更多关于 [wordcloud示例请参见官方](https://github.com/amueller/word_cloud)
   
   
   