"""
对影评进行聚类，使用基于密度的DBSCAN算法
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
import re
import jieba
import jieba.posseg as pseg
import math
import codecs
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
matplotlib.rc("font",family='YouYuan')
# import hdbscan

def Cluster_dbscan(file_path='res\\comments\\剧情\\千与千寻_200条影评.txt', file_name='千与千寻', save_path='res\\output\\cluster\\kmeans\\' ,eps=0.05, min_samples=3):
    """
    对影评进行聚类，使用基于密度的DBSCAN算法
    init中输入一个已经经过preprocess和feature_extraction的features list，即可进行聚类
    """
    corpus = []

    # 读取预料,去除首位缩进，一行预料为一个文档
    for line in open(file_path, 'r', encoding='utf-8').readlines():
        sentence_seged = jieba.posseg.cut(line.strip())
        outstr = ''
        outstr_ = ''
        for x in sentence_seged:
            if x.flag == 'n' or x.flag == 'v' or x.flag == 'a':
                outstr += "{},".format(x.word)
            outstr += "{},".format(x.word)
            # outstr_ += "{},".format(x.flag)
        corpus.append(outstr)
    # print(corpus)
    with open('res\\dicts\\sentiment-words.txt', 'r', encoding='utf-8') as f1:
        sentiment_words = f1.readlines()
        # 去除英文和空格和标点
        sentiment_words = [word.strip() for word in sentiment_words]
        # print(sentiment_words)
    for i in range(len(sentiment_words)):
        word = sentiment_words[i]
        index = word.index(',')
        word = word[0:index]
        sentiment_words[i] = word


    # 计算sentiment_words在corpus中出现的次数


    vectorizer = CountVectorizer(vocabulary=sentiment_words, min_df=3)
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)
    sentiment_words_ = []
    # 输出特征词和对应的次数
    for word, count in zip(feature_names, counts):
        # print(f"{word}: {count}")
        if count > 0:
            sentiment_words_.append(word)


    vectorizer2 = CountVectorizer(min_df=5)
    transformer2 = TfidfTransformer()
    tfidf2 = transformer2.fit_transform(vectorizer2.fit_transform(corpus))
    word2 = vectorizer2.get_feature_names_out()
    for word in word2:
        if word not in sentiment_words_:
            token = jieba.posseg.cut(word)
            for x in token:
                if x.flag == 'n' or x.flag == 'v' or x.flag == 'a':
                    # print('666')
                    sentiment_words_.append(word)
                    break

    vectorizer = CountVectorizer(vocabulary=sentiment_words_)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names_out()
    weight = tfidf.toarray()
    # resName = "res\\output\\SOPMI\\tfidf_Result.txt"
    # result = codecs.open(resName, 'w', 'utf-8')
    # for i in range(len(weight)):
    #     for j in range(len(word)):
    #         result.write(str(weight[i][j]) + ' ')
    #     result.write('\r\n\r\n')
    # result.close()

    # 创建DBSCAN对象
    db = DBSCAN(eps=eps, min_samples=min_samples)
    # 进行聚类
    db.fit(weight)
    # 聚类结果
    labels = db.labels_
    # 轮廓系数评价聚类的好坏
    sihouette = silhouette_score(weight, labels)
    calinski_harabasz = calinski_harabasz_score(weight, labels)
    davies_bouldin = davies_bouldin_score(weight, labels)
    print("轮廓系数:", sihouette)
    print("Calinski-Harabasz Index:", calinski_harabasz)
    print("Davies-Bouldin Index:", davies_bouldin)
    # 每个样本所属的簇
    print("每个样本所属的簇:", labels)

    # 绘制聚类图片
    pca = PCA(n_components=2).fit(weight)
    datapoint = pca.transform(weight)
    plt.figure(figsize=(8, 5))
    plt.figure(1)
    plt.clf()
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=labels, cmap=plt.cm.nipy_spectral, edgecolor='k')
    plt.title("DBSCAN " + file_name)
    plt.xlabel("feature space for the 1st feature")
    plt.ylabel("feature space for the 2nd feature")
    plt.savefig(save_path + '_' + file_name + '_DBSCAN.png')

def draw_dbscan(comments_list, save_path, file_name, eps=0.05, min_samples=3):
    corpus = []
    for comment in comments_list:
        sentence_seged = jieba.posseg.cut(comment.strip())
        outstr = ''
        for x in sentence_seged:
            if x.flag == 'n' or x.flag == 'v' or x.flag == 'a':
                outstr += "{},".format(x.word)
        corpus.append(outstr)
    with open('res\\dicts\\sentiment-words.txt', 'r', encoding='utf-8') as f1:
        sentiment_words = f1.readlines()
        # 去除英文和空格和标点
        sentiment_words = [word.strip() for word in sentiment_words]

    for i in range(len(sentiment_words)):
        word = sentiment_words[i]
        index = word.index(',')
        word = word[0:index]
        sentiment_words[i] = word

    vectorizer = CountVectorizer(vocabulary=sentiment_words, min_df=3)
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)
    sentiment_words_ = []
    # 输出特征词和对应的次数
    for word, count in zip(feature_names, counts):
        if count > 0:
            sentiment_words_.append(word)
    # print('sentiment_words_len', sentiment_words_.__len__())

    vectorizer2 = CountVectorizer(min_df=5)
    transformer2 = TfidfTransformer()
    tfidf2 = transformer2.fit_transform(vectorizer2.fit_transform(corpus))
    word2 = vectorizer2.get_feature_names_out()
    # print('word2', word2)
    for word in word2:
        if word not in sentiment_words_:
            token = jieba.posseg.cut(word)
            for x in token:
                if x.flag == 'n' or x.flag == 'v' or x.flag == 'a':
                    # print('666')
                    sentiment_words_.append(word)
                    break
    # print('sentiment_words_', sentiment_words_)
    # print('sentiment_words_len', sentiment_words_.__len__())

    vectorizer = CountVectorizer(vocabulary=sentiment_words_)

    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names_out()
    weight = tfidf.toarray()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    # 进行聚类
    db.fit(weight)
    # 聚类结果
    labels = db.labels_

    # 每个样本所属的簇
    print("每个样本所属的簇:", labels)

    # 绘制聚类图片
    pca = PCA(n_components=2).fit(weight)
    datapoint = pca.transform(weight)
    plt.figure(figsize=(8, 5))
    plt.figure(1)
    plt.clf()
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=labels, cmap=plt.cm.nipy_spectral, edgecolor='k')
    plt.title("DBSCAN " + file_name)
    plt.xlabel("feature space for the 1st feature")
    plt.ylabel("feature space for the 2nd feature")
    plt.savefig(save_path)

if __name__ == '__main__':
    folder_path1 = 'res\\comments\\剧情'
    folder_path2 = 'res\\comments\\喜剧'
    for root, dirs, files in os.walk(folder_path1):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            index = file_name.index('_')
            file_name_base = file_name[0:index]
            save_path = 'res\\output\\clustering_result\\dbscan\\剧情\\'
            Cluster_dbscan(file_path=file_path, file_name=file_name_base, save_path=save_path)
    for root, dirs, files in os.walk(folder_path2):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            index = file_name.index('_')
            file_name_base = file_name[0:index]
            save_path = 'res\\output\\clustering_result\\dbscan\\喜剧\\'
            Cluster_dbscan(file_path=file_path, file_name=file_name_base, save_path=save_path)

