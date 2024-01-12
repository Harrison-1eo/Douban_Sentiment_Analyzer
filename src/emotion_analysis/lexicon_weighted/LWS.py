
import pandas as pd
import jieba

class LexiconWeightedSentiment:
    def __init__(self):
        # 初始化时加载所有的词典
        self.load_dictionaries()

    def load_dictionaries(self):
        # 中英文程度词和情感词的路径
        dicts_paths = {
            'English_degree': '.\\res\\dicts\\zhihu\\程度级别词语（英文）.txt',
            'Chinese_degree': '.\\res\\dicts\\zhihu\\程度级别词语（中文）.txt',
            'English_negative_comments': '.\\res\\dicts\\zhihu\\负面评价词语（英文）.txt',
            'Chinese_negative_comments': '.\\res\\dicts\\zhihu\\负面评价词语（中文）.txt',
            'English_negative_emotions': '.\\res\\dicts\\zhihu\\负面情感词语（英文）.txt',
            'Chinese_negative_emotions': '.\\res\\dicts\\zhihu\\负面情感词语（中文）.txt',
            'English_positive_comments': '.\\res\\dicts\\zhihu\\正面评价词语（英文）.txt',
            'Chinese_positive_comments': '.\\res\\dicts\\zhihu\\正面评价词语（中文）.txt',
            'English_positive_emotions': '.\\res\\dicts\\zhihu\\正面情感词语（英文）.txt',
            'Chinese_positive_emotions': '.\\res\\dicts\\zhihu\\正面情感词语（中文）.txt'
        }

        # 加载所有词典
        for key, path in dicts_paths.items():
            setattr(self, key, self.load_words_from_file(path))

        # 处理程度级别词
        self.process_degree_words()

        # 合并中英文负面和正面词语
        self.negative_words = self.English_negative_comments + self.Chinese_negative_comments + self.English_negative_emotions + self.Chinese_negative_emotions
        self.positive_words = self.English_positive_comments + self.Chinese_positive_comments + self.English_positive_emotions + self.Chinese_positive_emotions

        #读入停用词表
        Chinese_English_stopwords_path = '.\\res\\dicts\\zhihu\\Chinese_English_stopwords.txt'
        with open(Chinese_English_stopwords_path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            self.Chinese_English_stopwords = [line.strip() for line in lines]


    def load_words_from_file(self, path):
        with open(path, mode='r', encoding='gbk') as file:
            lines = file.readlines()
            return [line.strip() for line in lines if line.strip()]

    def process_degree_words(self):
        degree_categories = ['most', 'very', 'more', 'ish', 'insufficiently', 'over']
        for language in ['English', 'Chinese']:
            degree_list: list = getattr(self, f"{language}_degree")
            for category in degree_categories:
                index = degree_list.index(f'{degree_categories.index(category) + 1}.')
                next_index = degree_list.index(f'{degree_categories.index(category) + 2}.') if category != 'over' else len(degree_list)
                setattr(self, f'{language}_degree_{category}', degree_list[index + 1:next_index])
    
    def word_cut(self, mytext):
        # 使用jieba进行分词
        return ','.join(jieba.cut(mytext))

    def remove_stopwords(self, segmented_text):
        #字符串转为列表
        segmented_lst = segmented_text.split(',')
        #去停用词
        words_without_stopwords = []
        for w in segmented_lst:
            if w not in self.Chinese_English_stopwords:
                words_without_stopwords.append(w)
        return words_without_stopwords

    # def score(self, words):
    #     most_socre = 8
    #     very_socre = 6
    #     more_socre = 4
    #     ish_socre = 0.6
    #     insufficiently_socre = -1.5
    #     over_socre = 2
    #     positive_socre = 1
    #     negative_socre = -1
    #     no_attitude_score = 0
    #     PS = []#句子中正向得分
    #     NS = []#句子中负向得分
    #     NAS = []#非态度词得分

    #     for w in words:
    #         if w in negative_words:
    #             if words[words.index(w) -1] in most_degree:
    #                 NS.append(most_socre * negative_socre)
    #             elif words[words.index(w) -1] in very_degree:
    #                 NS.append(very_socre * negative_socre)
    #             elif words[words.index(w) -1] in more_degree:
    #                 NS.append(more_socre * negative_socre)
    #             elif words[words.index(w) -1] in ish_degree:
    #                 NS.append(ish_socre * negative_socre)
    #             elif words[words.index(w) -1] in insufficiently_degree:
    #                 NS.append(insufficiently_socre * negative_socre)
    #             elif words[words.index(w) -1] in over_degree:
    #                 NS.append(over_socre * negative_socre)
    #             else:
    #                 NS.append(negative_socre)

    #         elif w in positive_words:
    #             if words[words.index(w) -1] in most_degree:
    #                 PS.append(most_socre * positive_socre)
    #             elif words[words.index(w) -1] in very_degree:
    #                 PS.append(very_socre * positive_socre)
    #             elif words[words.index(w) -1] in more_degree:
    #                 PS.append(more_socre * positive_socre)
    #             elif words[words.index(w) -1] in ish_degree:
    #                 PS.append(ish_socre * positive_socre)
    #             elif words[words.index(w) -1] in insufficiently_degree:
    #                 PS.append(insufficiently_socre * positive_socre)
    #             elif words[words.index(w) -1] in over_degree:
    #                 PS.append(over_socre * positive_socre)
    #             else:
    #                 PS.append(positive_socre)
    #         else:
    #             NAS.append(no_attitude_score)

    #     final_score = sum(NS) + sum(PS)
    #     return final_score

    def score(self, words, positive_score=1, negative_score=-1):
        # 词汇的分数映射
        degree_scores = {
            'most': 8,
            'very': 6,
            'more': 4,
            'ish': 0.6,
            'insufficiently': -1.5,
            'over': 2
        }

        PS, NS, NAS = [], [], []  # 分别存储正向得分、负向得分和非态度词得分

        for i, w in enumerate(words):
            prev_word = words[i - 1] if i > 0 else None

            if w in self.negative_words or w in self.positive_words:
                sentiment = positive_score if w in self.positive_words else negative_score
                score_multiplier = self.get_degree_score(prev_word, degree_scores)
                score = score_multiplier * sentiment
                PS.append(score) if sentiment > 0 else NS.append(score)
            else:
                NAS.append(0)

        final_score = sum(NS) + sum(PS)
        return final_score


    def get_degree_score(self, word, degree_scores):
        # 检查程度词并返回相应的分数
        for degree, score in degree_scores.items():
            if word in getattr(self, f'{degree}_degree', []):
                return score
        return 1  # 如果没有程度词，返回基本分数 1


    def analyze_sentiment_score(self, comments):
        # comments: list of str，每个元素是一条评论
        # 分析评论的情感得分
        scores = []
        for comment in comments:
            segmented_text = self.word_cut(comment)
            words = self.remove_stopwords(segmented_text)
            score = self.score(words)
            scores.append(score)
        return scores



# 使用示例
if __name__ == "__main__":
    import sys
    sys.path.append('src')
    import preprocessing as pp
    comments = pp.get_comment_content('./res/comments/剧情/泰坦尼克号_200条影评.txt')

    lws = LexiconWeightedSentiment()
    scores = lws.analyze_sentiment_score(comments)

    for index, c in enumerate(comments):
        print(c, scores[index])

    good_comments = [c for index, c in enumerate(comments) if scores[index] > 0]
    bad_comments = [c for index, c in enumerate(comments) if scores[index] < 0]
    mid_comments = [c for index, c in enumerate(comments) if scores[index] == 0]
    print(f'正面评论数：{len(good_comments)}')
    print(f'负面评论数：{len(bad_comments)}')
    print(f'中性评论数：{len(mid_comments)}')

    from util import draw_pie_chart
    draw_pie_chart(
        [len(good_comments), len(bad_comments), len(mid_comments)], 
        ['正面评论', '负面评论', '中性评论'], 
        '泰坦尼克号_200条影评情感分析结果饼图',
        'pie_chart.png'
    )
    