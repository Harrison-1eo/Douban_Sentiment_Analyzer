import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gensim.models import keyedvectors

device="cpu"
Embedding_size = 50
Batch_Size = 32
Kernel = 3
Filter_num = 10#卷积核的数量。
Epoch = 60
Dropout = 0.5
Learning_rate = 1e-3

sequence_length = 62

num_classs = 2

with open(os.path.join("res\\dicts\\textcnn\\word_freq.txt"), encoding='utf-8') as fin:
    vocab = [i.strip() for i in fin]
vocab=set(vocab)
word2idx = {i:index for index, i in enumerate(vocab)}
idx2word = {index:i for index, i in enumerate(vocab)}

w2v=keyedvectors.load_word2vec_format(os.path.join("res\\dicts\\textcnn\\wiki_word2vec_50.bin"),binary=True)

for word in vocab:
    try:
        # 尝试获取词的向量表示
        _ = w2v[word]
    except KeyError:
        # 如果词不存在于Word2Vec模型中，随机生成一个向量
        w2v[word] = np.random.randn(Embedding_size)

def word2vec(x):
    x2v=np.ones((len(x),x.shape[1],Embedding_size))
    for i in range(len(x)):
        x2v[i]=w2v[[idx2word[j.item()] for j in x[i]]]
    return torch.tensor(x2v).to(torch.float32)

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
#         self.W = nn.Embedding(vocab_size, embedding_dim=Embedding_size)
        out_channel = Filter_num #可以等价为通道的解释。
        self.conv = nn.Sequential(
                    nn.Conv2d(1, out_channel, (2, Embedding_size)),#卷积核大小为2*Embedding_size,默认当然是步长为1
                    nn.ReLU(),
                    nn.MaxPool2d((sequence_length-1,1)),
        )
        self.dropout = nn.Dropout(Dropout)
        self.fc = nn.Linear(out_channel, num_classs)
    
    def forward(self, X):
        batch_size = X.shape[0]
        embedding_X =  word2vec(X)
        embedding_X = embedding_X.unsqueeze(1)
        conved = self.conv(embedding_X)
        conved = self.dropout(conved)
        flatten = conved.view(batch_size, -1)
        output = self.fc(flatten)
        #2分类问题，往往使用softmax，表示概率。
        return F.log_softmax(output)
    
def preprocess_and_tokenize(word2idx, text):
    # 对文本进行预处理（例如分词）
    tokens = text.split()

    # 将词转换为索引
    token_indices = [word2idx.get(token, word2idx["把"]) for token in tokens]

    # 进行padding或截断
    padded_indices = token_indices[:sequence_length] + [word2idx["把"]] * (sequence_length - len(token_indices))
    
    return padded_indices
    
def predict(commands):
    # commands: list of str
    model2 = TextCNN()  # 或者是模型的构造函数
    model2.load_state_dict(torch.load('res\\dicts\\textcnn\\textcnn.pth'))
    model2.eval()  # 切换到评估模式

    commands = [preprocess_and_tokenize(word2idx, command) for command in commands]
    
    res = []
    with torch.no_grad():
        for c in commands:
            input_tensor = torch.LongTensor([c])  # 将处理后的文本转换为Tensor
            output = model2(input_tensor)
            prediction = torch.max(output, dim=1)[1].item()  # 获取预测类别
            res.append(prediction)
    
    return res

if __name__ == '__main__':
    import sys
    sys.path.append('src\preprocessing')
    import preprocessing as pp
    comments = pp.get_comment_content('./res/comments/剧情/泰坦尼克号_200条影评.txt')

    scores = predict(comments)
    for index, c in enumerate(comments):
        print(c, scores[index])

    good_comments = [c for index, c in enumerate(comments) if scores[index] > 0]
    bad_comments = [c for index, c in enumerate(comments) if scores[index] < 0]
    mid_comments = [c for index, c in enumerate(comments) if scores[index] == 0]
    print(f'正面评论数：{len(good_comments)}')
    print(f'负面评论数：{len(bad_comments)}')
    print(f'中性评论数：{len(mid_comments)}')
