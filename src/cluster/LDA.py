import jieba
import jieba.posseg as jp
from gensim import corpora, models
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import matplotlib
matplotlib.rc("font",family='YouYuan')
# Global Dictionary
new_words = ['奥预赛', '折叠屏']  # 新词
stopwords = {' ', '再', '的', '们', '为', '时', '：', '，', ',', '是', '了'}  # 停用词
synonyms = {'韩国': '南朝鲜', '传言': '流言'}  # 同义词
words_nature = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 可用的词性


def add_new_words():  # 增加新词
    for i in new_words:
        jieba.add_word(i)


def remove_stopwords(ls):  # 去除停用词
    return [word for word in ls if word not in stopwords]


def replace_synonyms(ls):  # 替换同义词
    return [synonyms[i] if i in synonyms else i for i in ls]

def Cluster_LDA(num_topics=8, save_path='res\\output\\cluster\\LDA\\', file_name='千与千寻', file_path='res\\comments\\剧情\\千与千寻_200条影评.txt'):
    documents = []
    # file_path = 'res\\comments\\剧情\\千与千寻_200条影评.txt'
    for line in open(file_path, 'r', encoding='utf-8').readlines():
        sentence_seged = jieba.posseg.cut(line.strip())
        outstr = ''
        outstr_ = ''
        for x in sentence_seged:
            if x.flag in words_nature:
                outstr += "{},".format(x.word)
            outstr += "{},".format(x.word)
            # outstr_ += "{},".format(x.flag)
        documents.append(outstr)
    add_new_words()
    words_ls = []
    for text in documents:
        words = replace_synonyms(remove_stopwords([w.word for w in jp.cut(text)]))
        words_ls.append(words)

    # 生成语料词典
    dictionary = corpora.Dictionary(words_ls)
    # 生成稀疏向量集
    corpus = [dictionary.doc2bow(words) for words in words_ls]
    # LDA模型，num_topics设置聚类数，即最终主题的数量
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    # 展示每个主题的前5的词语
    for topic in lda.print_topics(num_words=5):
        print(topic)
    # 推断每个语料库中的主题类别
    print('推断：')
    labels = []
    for e, values in enumerate(lda.inference(corpus)[0]):
        topic_val = 0
        topic_id = 0
        for tid, val in enumerate(values):
            if val > topic_val:
                topic_val = val
                topic_id = tid
        # print(topic_id, '->', documents[e])
        labels.append(topic_id)
    print(labels)
    # 画图
    tsne_model = TSNE(n_components=2, random_state=42)
    doc_topic_tsne = tsne_model.fit_transform(lda.inference(corpus)[0])
    plt.figure(figsize=(8, 5))
    plt.figure(1)
    plt.clf()
    plt.scatter(doc_topic_tsne[:, 0], doc_topic_tsne[:, 1], c=labels, cmap='viridis', marker='.')
    plt.title('LDA Clustering with t-SNE Visualization')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(save_path + '_' + file_name + '_LDA.png')
    # plt.cycler()
    # plt.show()


def draw_LDA(comments_list, save_path, file_name):
    documents = []
    for line in comments_list:
        sentence_seged = jieba.posseg.cut(line.strip())
        outstr = ''
        outstr_ = ''
        for x in sentence_seged:
            if x.flag in words_nature:
                outstr += "{},".format(x.word)
            outstr += "{},".format(x.word)
            # outstr_ += "{},".format(x.flag)
        documents.append(outstr)
    add_new_words()
    words_ls = []
    for text in documents:
        words = replace_synonyms(remove_stopwords([w.word for w in jp.cut(text)]))
        words_ls.append(words)

    # 生成语料词典
    dictionary = corpora.Dictionary(words_ls)
    # 生成稀疏向量集
    corpus = [dictionary.doc2bow(words) for words in words_ls]
    # LDA模型，num_topics设置聚类数，即最终主题的数量
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=8)
    # 展示每个主题的前5的词语
    for topic in lda.print_topics(num_words=5):
        print(topic)
    # 推断每个语料库中的主题类别
    print('推断：')
    labels = []
    for e, values in enumerate(lda.inference(corpus)[0]):
        topic_val = 0
        topic_id = 0
        for tid, val in enumerate(values):
            if val > topic_val:
                topic_val = val
                topic_id = tid
        # print(topic_id, '->', documents[e])
        labels.append(topic_id)
    print(labels)
    # 画图
    tsne_model = TSNE(n_components=2, random_state=42)
    doc_topic_tsne = tsne_model.fit_transform(lda.inference(corpus)[0])
    plt.figure(figsize=(8, 5))
    plt.figure(1)
    plt.clf()
    plt.scatter(doc_topic_tsne[:, 0], doc_topic_tsne[:, 1], c=labels, cmap='viridis', marker='.')
    plt.title('LDA Clustering with t-SNE Visualization' + file_name)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(save_path)

if __name__ == '__main__':
    folder_path1 = 'res\\comments\\剧情\\'
    folder_path2 = 'res\\comments\\喜剧\\'
    for root, dirs, files in os.walk(folder_path1):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            index = file_name.index('_')
            file_name_base = file_name[0:index]
            save_path = 'res\\output\\clustering_result\\LDA\\剧情\\'
            Cluster_LDA(save_path=save_path, file_name=file_name_base, file_path=file_path)
    for root, dirs, files in os.walk(folder_path2):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            index = file_name.index('_')
            file_name_base = file_name[0:index]
            save_path = 'res\\output\\clustering_result\\LDA\\喜剧\\'
            Cluster_LDA(save_path=save_path, file_name=file_name_base, file_path=file_path)



