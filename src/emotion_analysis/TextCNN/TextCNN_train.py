root_path="res\\dicts\\textcnn\\"
train_path="train.txt"
import pandas as pd
train_data = pd.read_csv(root_path+"train.txt",names=["label","comment"],sep="\t")
print(train_data.head())

comments_len=train_data.iloc[:,1].apply(lambda x:len(x.split()))
print(comments_len.describe())

train_data["comments_len"]=comments_len
train_data["comments_len"].describe(percentiles=[.5,.95])

from collections import Counter
words=[]
for i in range(len(train_data)):
    com=train_data["comment"][i].split()
    words=words+com
print(len(words))

Freq=30
import os
with open(os.path.join(root_path,"word_freq.txt"), 'w', encoding='utf-8') as fout:
    for word,freq in Counter(words).most_common():
        if freq>Freq:
            fout.write(word+"\n")

#初始化vocab
with open(os.path.join(root_path+"word_freq.txt"), encoding='utf-8') as fin:
    vocab = [i.strip() for i in fin]
vocab=set(vocab)
word2idx = {i:index for index, i in enumerate(vocab)}
idx2word = {index:i for index, i in enumerate(vocab)}#没有想到列表竟然可以枚举。
vocab_size = len(vocab)
print(len(vocab))

pad_id=word2idx["把"]
print(pad_id)

sequence_length = 62
#对输入数据进行预处理,主要是对句子用索引表示且对句子进行截断与padding，将填充使用”把“来。
def tokenizer():
    inputs = []
    sentence_char = [i.split() for i in train_data["comment"]]
    # 将输入文本进行padding
    for index,i in enumerate(sentence_char):
        temp=[word2idx.get(j,pad_id) for j in i]#表示如果词表中没有这个稀有词，无法获得，那么就默认返回pad_id。
        if(len(i)<sequence_length):
            #应该padding。
            for _ in range(sequence_length-len(i)):
                temp.append(pad_id)
        else:
            temp = temp[:sequence_length]
        inputs.append(temp)
    return inputs
data_input = tokenizer()

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"
Embedding_size = 50
Batch_Size = 32
Kernel = 3
Filter_num = 10#卷积核的数量。
Epoch = 60
Dropout = 0.5
Learning_rate = 1e-3

class TextCNNDataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.LongTensor(data_inputs)
        self.label = torch.LongTensor(data_targets)
        
    def __getitem__(self, index):
        return self.inputs[index], self.label[index] 
    
    def __len__(self):
        return len(self.inputs)
    
TextCNNDataSet = TextCNNDataSet(data_input, list(train_data["label"]))
train_size = int(len(data_input) * 0.9)
test_size = int(len(data_input) * 0.05)
print(train_size)
print(test_size)

val_size= len(data_input) -train_size-test_size#乘以0.75反而报错，因为那个有取整，所以导致了舍入。
train_dataset,val_dataset,test_dataset = torch.utils.data.random_split(TextCNNDataSet, [train_size,val_size, test_size])

TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True)
TestDataLoader = Data.DataLoader(test_dataset, batch_size=Batch_Size, shuffle=True)

from gensim.models import keyedvectors
w2v=keyedvectors.load_word2vec_format(os.path.join("/content/drive/MyDrive/TextCNN/wiki_word2vec_50.bin"),binary=True)

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

#使用word2vec版本的。
num_classs = 2#2分类问题。

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
    
model = TextCNN().to(device)
optimizer = optim.Adam(model.parameters(),lr=Learning_rate)

def binary_acc(pred, y):
    """
    计算模型的准确率
    :param pred: 预测值
    :param y: 实际真实值
    :return: 返回准确率
    """
    correct = torch.eq(pred, y).float()
    acc = correct.sum() / len(correct)
    return acc.item()

def train():
    avg_acc = []
    model.train()
    for index, (batch_x, batch_y) in enumerate(TrainDataLoader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        loss = F.nll_loss(pred, batch_y)
        acc = binary_acc(torch.max(pred, dim=1)[1], batch_y)
        avg_acc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_acc = np.array(avg_acc).mean()
    return avg_acc

# Training cycle
model_train_acc, model_test_acc = [], []
for epoch in range(Epoch):
    train_acc = train()
    print("epoch = {}, 训练准确率={}".format(epoch + 1, train_acc))
    model_train_acc.append(train_acc)

def evaluate():
    """
    模型评估
    :param model: 使用的模型
    :return: 返回当前训练的模型在测试集上的结果
    """
    avg_acc = []
    model.eval()  # 进入测试模式
    with torch.no_grad():
        for x_batch, y_batch in TestDataLoader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            acc = binary_acc(torch.max(pred, dim=1)[1], y_batch)
            avg_acc.append(acc)
    return np.array(avg_acc).mean()

evaluate()

import matplotlib.pyplot as plt
plt.plot(model_train_acc)
plt.ylim(ymin=0.5, ymax=0.8)
plt.title("The accuracy of textCNN model")
plt.show()
plt.savefig(root_path+"accuracy.png")

torch.save(model.state_dict(), root_path+'textcnn.pth')

