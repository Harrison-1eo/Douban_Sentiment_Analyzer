import jieba
import re
# 获取评论内容
# file_path: str，文件路径
def get_comment_content(file_path: str) -> list:
    with open(file_path, 'r', encoding='utf-8') as f:
        comments = f.read()
        comments = comments.split('\n\n')
    
    comments = [c for c in comments if c.strip() != '']
    return comments
    
# 分词
# content: list of str，每个元素是一条评论
def cut_words(content: list) -> list:
    allpuncs = "，_《。》、？；：‘’＂“”【「】」·！@￥…（）—,<.>/?;:\'\"[]{}~`!@#$%^&*()-=\+] \n\t\r"
    allpuncs = list(allpuncs[::])
    # print(allpuncs)
    words = []
    for c in content:
        words.append(jieba.lcut(c))
        # 去除标点符号
        words[-1] = [w for w in words[-1] if w not in allpuncs]
        # 去除空字符
        words[-1] = [w for w in words[-1] if w.strip() != '']
    return words

# 去除停用词
# words: list of list of str，每个元素是一条评论，每条评论是一个词列表
# stop_words_path: str，停用词表路径
def remove_stop_words(words: list, stop_words_path: str) -> list:
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        stop_words = f.read()
        stop_words = stop_words.split('\n')
    
    stop_words_dict = {}
    for w in stop_words:
        stop_words_dict[w] = 1

    for index, word in enumerate(words):
        words[index] = [w for w in word if w not in stop_words_dict]
    
    words = [word for word in words if word != []]
    return words

# 获得词频，并按照词频降序排序，输出前n个词
# words: list of list of str，每个元素是一条评论，每条评论是一个词列表
# n: int，输出前n个词，及其词频
def get_word_frequency(words: list, n: int) -> list:
    word_frequency = {}
    for word in words:
        for w in word:
            if w not in word_frequency:
                word_frequency[w] = 1
            else:
                word_frequency[w] += 1
    word_frequency = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
    return [w for w in word_frequency[:n]]

if __name__ == "__main__":
    comments = []

    # 获取评论内容
    # 遍历'./res/comments'所有文件夹下的文件
    import os
    for root, dirs, files in os.walk('./res/comments'):
        for file in files:
            print(os.path.join(root, file))
            comments += get_comment_content(os.path.join(root, file))

    comments = cut_words(comments)
    comments = remove_stop_words(comments, 'res\dicts\stop-words.txt')
    for index, c in enumerate(comments):
        print(index)
        print(c)
        print('---------------------------')

    word_frequency = get_word_frequency(comments, 200)
    # 输出到文件中
    with open('res\dicts\extra.txt', 'w+', encoding='utf-8') as f:
        for w in word_frequency:
            f.write(w[0] + '\t' + 'pos' + '\n')

