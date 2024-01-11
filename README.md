# Douban_Sentiment_Analyzer

## 使用说明

1. 环境配置：使用`pip install -r requirements.txt`安装依赖包
2. 运行`main.py`文件
3. 分析结果将保存在`save`文件夹下，每个子文件夹（为电影编号）对应一个电影的分析结果

## 算法介绍

### 情感检测算法

#### 1. 词典情感分析

参考链接：
- [基于词典的情感分析](https://zhuanlan.zhihu.com/p/201028746)
- [Hello-NLP](https://zhuanlan.zhihu.com/p/142011031)

#### 2. SO-PMI

参考链接：
- [SentimentWordExpansion](https://github.com/liuhuanyong/SentimentWordExpansion/tree/master)
- [SO-PMI算法详解](https://blog.csdn.net/qq_35357274/article/details/109027337)

#### 3. TextCNN

参考链接：
- [Text-CNN+Word2vec+电影评论情感分析实战](https://blog.csdn.net/qq_43391414/article/details/118557836)
- [文本分类之TextCNN模型原理和实现](https://blog.csdn.net/GFDGFHSDS/article/details/105295247)