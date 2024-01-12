# import pandas as pd
# import codecs
# import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
#
# def kmeans_cluster(n_clusters):
#     """
#     Perform K-Means clustering on the given features.
#
#     Parameters:
#     - features (sparse matrix): Features of the corpus.
#     - n_clusters (int): Number of clusters.
#
#     Returns:
#     - labels (list): Cluster labels of the corpus.
#     """
#     # 创建KMeans对象
#
#
#     corpus = []
#
#     # 读取预料,去除首位缩进，一行预料为一个文档
#     for line in open('res\\comments\\剧情\\千与千寻_200条影评.txt', 'r', encoding='utf-8').readlines():
#         corpus.append(line.strip())
#     print(corpus.__len__())
#     vectorizer = CountVectorizer(min_df=5)
#     transformer = TfidfTransformer()
#     tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
#     word = vectorizer.get_feature_names_out()
#     weight = tfidf.toarray()
#     resName = "res\\output\\SOPMI\\tfidf_Result.txt"
#     result = codecs.open(resName, 'w', 'utf-8')
#     print(len(weight))
#     for i in range(len(weight)):
#         for j in range(len(word)):
#             result.write(str(weight[i][j]) + ' ')
#         result.write('\r\n\r\n')
#     result.close()
#
#     from sklearn.cluster import KMeans
#     clf = KMeans(n_clusters=n_clusters, n_init=10)
#     s = clf.fit(weight)
#     labels = s.labels_
#     # labels中大于0数量
#     print(len(labels[labels[:] > 0]))
#     print(labels)
#     # 每个样本所属的簇
#     label = []
#     i = 1
#     while i <= len(clf.labels_):
#         label.append(clf.labels_[i - 1])
#         i = i + 1
#
#     y_pred = clf.labels_
#
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=2)  # 输出两维
#     newData = pca.fit_transform(weight)  # 载入N维
#
#     xs, ys = newData[:, 0], newData[:, 1]
#     # 设置颜色
#     cluster_colors = {0: 'r', 1: 'yellow', 2: 'b', 3: 'chartreuse', 4: 'purple', 5: '#FFC0CB', 6: '#6A5ACD',
#                       7: '#98FB98'}
#
#     # 设置类名
#     cluster_names = {0: u'类0', 1: u'类1', 2: u'类2', 3: u'类3', 4: u'类4', 5: u'类5', 6: u'类6', 7: u'类7'}
#
#     df = pd.DataFrame(dict(x=xs, y=ys, label=y_pred, title=corpus))
#     groups = df.groupby('label')
#
#     fig, ax = plt.subplots(figsize=(8, 5))  # set size
#     ax.margins(0.02)
#     for name, group in groups:
#         ax.plot(group.x, group.y, marker='o', linestyle='', ms=10, label=cluster_names[name],
#                 color=cluster_colors[name], mec='none')
#     plt.show()
#
#
#     return labels
#
# if __name__ =='__main__':
#     # 读取预料,去除首位缩进，一行预料为一个文档
#     kmeans_cluster(n_clusters=8)

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
# import hdbscan

def Cluster_kmeans(eps=0.05, min_samples=3, file_path='res\\comments\\剧情\\千与千寻_200条影评.txt', file_name='千与千寻', save_path='res\\output\\cluster\\kmeans\\'):
    """
    对影评进行聚类，使用基kmeans算法
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


    # print('sentiment_words', sentiment_words)
    # 计算sentiment_words在corpus中出现的次数


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

    # resName = "res\\output\\SOPMI\\tfidf_Result.txt"
    # result = codecs.open(resName, 'w', 'utf-8')
    # # print(len(weight))
    # for i in range(len(weight)):
    #     for j in range(len(word)):
    #         result.write(str(weight[i][j]) + ' ')
    #     result.write('\r\n\r\n')
    # result.close()

    # kmeans聚类
    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=8)
    # clf = K_Means(k=len(corpus) // 8)
    s = clf.fit(weight)
    labels = clf.labels_

    sihouette = silhouette_score(weight, labels)
    calinski_harabasz = calinski_harabasz_score(weight, labels)
    davies_bouldin = davies_bouldin_score(weight, labels)
    print("轮廓系数:", sihouette)
    print("Calinski-Harabasz Index:", calinski_harabasz)
    print("Davies-Bouldin Index:", davies_bouldin)
    print(clf.labels_)

    pca = PCA(n_components=2).fit(weight)
    datapoint = pca.transform(weight)
    plt.figure(figsize=(8, 5))
    plt.figure(1)
    plt.clf()
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=labels, cmap=plt.cm.nipy_spectral, edgecolor='k')
    plt.title("KMEANS " + file_name)
    plt.xlabel("feature space for the 1st feature")
    plt.ylabel("feature space for the 2nd feature")
    plt.savefig(save_path + '_' + file_name + '_kmeans.png')
    # plt.show()

def draw_kmeans(comments_list, save_path, file_name):
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

    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=8)
    # clf = K_Means(k=len(corpus) // 8)
    s = clf.fit(weight)
    print(clf.labels_)
    labels = clf.labels_
    pca = PCA(n_components=2).fit(weight)
    datapoint = pca.transform(weight)
    plt.figure(figsize=(8, 5))
    plt.figure(1)
    plt.clf()
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=labels, cmap=plt.cm.nipy_spectral, edgecolor='k')
    plt.title("KMEANS " + file_name)
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
            save_path = 'res\\output\\clustering_result\\kmeans\\剧情\\'
            Cluster_kmeans(file_path=file_path, file_name=file_name_base, save_path=save_path)
    for root, dirs, files in os.walk(folder_path2):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            index = file_name.index('_')
            file_name_base = file_name[0:index]
            save_path = 'res\\output\\clustering_result\\kmeans\\喜剧\\'
            Cluster_kmeans(file_path=file_path, file_name=file_name_base, save_path=save_path)
# if __name__ == '__main__':
#     folder_path1 = 'res\\comments\\剧情'
#     file_path = 'res\\comments\\剧情\\千与千寻_200条影评.txt'
#     file_name = '千与千寻'
#     save_path = 'res\\output\\clustering_result\\kmeans\\剧情\\'
#     Cluster_kmeans(file_path=file_path, file_name=file_name, save_path=save_path)


