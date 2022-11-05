import torch
import numpy as np
from importlib import import_module
import argparse
import os
import pickle as pkl
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Classification based Transformer")
parser.add_argument('--model', type=str, default='TextCNN')
parser.add_argument('--dataset', type=str, default='THUCNews')
parser.add_argument('--text', type=str, help='test on a random news')
parser.add_argument('--use_word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained')
args = parser.parse_args()


UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
dataset_name = args.dataset
#ThuNews
if dataset_name == "THUCNews":
    key = {
        0: '财经', # finance
        1: '房产', # realty
        2: '股市', # stocks
        3: '教育', # education
        4: '科技', # science
        5: '社会', # society
        6: '政务', # politics
        7: '体育', # sports
        8: '游戏', # game
        9: '娱乐', # entertainment
    }

model_name = args.model # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
x = import_module('models.' + model_name)

# 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
embedding = args.embedding
if model_name == 'FastText':
    from utils_fasttext import build_dataset, build_iterator, get_time_dif
    embedding = 'random'
config = x.Config(dataset_name, embedding)
if os.path.exists(config.vocab_path):
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    config.n_vocab = len(vocab)

model = x.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location=torch.device('cuda') ))
model.eval()


def build_predict_text(text, use_word):

    if use_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    token = tokenizer(text)
    seq_len = len(token)
    pad_size = config.pad_size
    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size-len(token)))
        else:
            token = token[:pad_size]
            seq_len = pad_size

    words_line = []
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))

    ids = torch.LongTensor([words_line]).to(config.device)
    seq_len = torch.LongTensor(seq_len).to(config.device)

    return ids, seq_len


def predict(text):
    data = build_predict_text(text,args.use_word)
    with torch.no_grad():
        outputs = model(data)
        num = torch.argmax(outputs)
        pred = F.softmax(outputs, dim=1)  # softmax标准化

    return text + " 属于" + str(key[int(num)]) + "类新闻（" + str(round(float(torch.max(pred)*100), 2)) + "%）"

if __name__ == "__main__":
    if args.text is None:
        print(predict("2022年金球奖:本泽马击败马内德布劳内首次获奖"))
    else:
        print(predict(args.text))