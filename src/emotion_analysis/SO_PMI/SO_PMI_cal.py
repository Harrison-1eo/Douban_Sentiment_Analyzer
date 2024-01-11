import jieba

class SoPmiSentiment:
    def __init__(self):
        # 初始化时加载所有的词典
        self.load_dictionaries()

    def load_dictionaries(self):
        # 词典路径
        dicts_paths = {
            'pos_dict': 'res\\dicts\\sopmi\\pos.txt',
            'neg_dict': 'res\\dicts\\sopmi\\neg.txt',
            'not_dict': 'res\\dicts\\sopmi\\not_dict.txt',
            'degree_dict': 'res\\dicts\\sopmi\\degree_dict.txt'
        }

        # 情感词影响力，即情感词的权重。排序小于100的情感词权重为3，100-1000的为2，1000-5000的为1
        influence = {
            100: 3,
            1000: 2,
            10000: 1
        }

        def get_influence(x):
            for key in influence.keys():
                if x < key:
                    return influence[key]
            else:
                return 0.5

        self.not_dict = {}
        with open(dicts_paths['not_dict'], mode='r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            if line.strip() == '':
                continue
            l = line.strip()
            self.not_dict[l[0]] = 1

        self.degree_dict = {}
        with open(dicts_paths['degree_dict'], mode='r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            if line.strip() == '':
                continue
            l = line.strip().split(' ')
            self.degree_dict[l[0]] = l[1]

        self.sen_dict = {}
        with open(dicts_paths['pos_dict'], mode='r', encoding='utf-8') as file:
            lines = file.readlines()
        for index, line in enumerate(lines[:10000]):
            if line.strip() == '':
                continue
            l = line.strip().split(',')
            self.sen_dict[l[0]] = get_influence(index)
        
        with open(dicts_paths['neg_dict'], mode='r', encoding='utf-8') as file:
            lines = file.readlines()
        for index, line in enumerate(lines[:10000]):
            if line.strip() == '':
                continue
            l = line.strip().split(',')
            if l[0] not in self.sen_dict:
                self.sen_dict[l[0]] = -get_influence(index)
            else:
                self.sen_dict[l[0]] -= get_influence(index)

    def word_cut(self, mytext):
        # 使用jieba进行分词
        return jieba.lcut(mytext)
            
    def score(self, words, positive_score=1, negative_score=-1):
        # words: list of str, 待打分的语句，已经分词

        score = 0
        
        # 遍历每个词，只保留在三个词典中出现的词
        words = [word for word in words if word in self.sen_dict or word in self.not_dict or word in self.degree_dict]
        lastsenloc = -1
        for i in range(0, len(words)):
            W = positive_score
            if words[i] in self.sen_dict:
                for other in words[lastsenloc + 1:i]:
                    if other in self.not_dict:
                        W *= negative_score
                    elif other in self.degree_dict:
                        W *= int(self.degree_dict[other])

                score += W * int(self.sen_dict[words[i]])
                lastsenloc = i

        return score
    
# 使用示例
if __name__ == "__main__":
    import sys
    sys.path.append('src\preprocessing')
    import preprocessing as pp
    comments = pp.get_comment_content('./res/comments/剧情/泰坦尼克号_200条影评.txt')

    scores = []
    sopmi = SoPmiSentiment()

    for index, c in enumerate(comments):
        c_cut = sopmi.word_cut(c)
        scores.append(sopmi.score(c_cut))
        if scores[index] <= 0:
            print(c, scores[index])
            w = [word for word in c_cut if word in sopmi.sen_dict or word in sopmi.not_dict or word in sopmi.degree_dict]
            print(w)
        


    good_comments = [c for index, c in enumerate(comments) if scores[index] > 0]
    bad_comments = [c for index, c in enumerate(comments) if scores[index] < 0]
    mid_comments = [c for index, c in enumerate(comments) if scores[index] == 0]
    print(f'正面评论数：{len(good_comments)}')
    print(f'负面评论数：{len(bad_comments)}')
    print(f'中性评论数：{len(mid_comments)}')

