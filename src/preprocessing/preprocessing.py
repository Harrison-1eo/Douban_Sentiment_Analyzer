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
    print(allpuncs)
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
    words = [[w for w in word if w not in stop_words] for word in words]
    words = [word for word in words if word != []]
    return words

if __name__ == "__main__":
    comments = get_comment_content('./res/comments/剧情/肖申克的救赎_200条影评.txt')
    comments = cut_words(comments)
    comments = remove_stop_words(comments, 'res\dicts\stop-words.txt')
    for index, c in enumerate(comments):
        print(index)
        print(c)
        print('---------------------------')
