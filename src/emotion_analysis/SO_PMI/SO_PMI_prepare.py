
import pandas as pd

# 打开dev\豆瓣电影评论数据集\comments.txt
# 读取前10行

with open('dev\豆瓣电影评论数据集\comments.txt', 'r', encoding='utf-8') as file:
    comments = file.readlines()

import sys
sys.path.append('src\preprocessing')
import preprocessing as pp

# 对每一行进行分词
comments_cut = pp.cut_words(comments)

# 将分词结果写入csv文件
comments = []
for c in comments_cut:
    try:
        c = ','.join(c)
        with open('dev\豆瓣电影评论数据集\comments_cut.csv', 'a', encoding='utf-8') as file:
            file.write(c + '\n')
        comments.append(c)
    except:
        print(c, type(c))
        continue


comments = []
with open('dev\豆瓣电影评论数据集\comments_cut.csv', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        comments.append(line.strip().split(','))

# 去除停用词
comments = pp.remove_stop_words(comments, 'res\dicts\stop-words.txt')

# 获得词频
word_frequency = pp.get_word_frequency(comments, 500)

# 保存到csv文件
word_frequency = pd.DataFrame(word_frequency, columns=['word', 'frequency'])
word_frequency.to_csv('dev\豆瓣电影评论数据集\word_frequency.csv', index=False, encoding='utf-8')