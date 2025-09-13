"""
构造数据集
"""
import os
import json
import re
import string
from collections import OrderedDict
import torchtext.vocab
import jieba
import torch
from pandas.core.dtypes.inference import is_number
from torch.utils.data import DataLoader , Dataset
from utils_project import create_dir  , read_file

jieba.load_userdict("jieba_dict.txt")
#分词
def split_sentence(sentence):
    return jieba.lcut(sentence)

#数字判断
def is_number(token):
    re_number = re.compile("([0-9\.]+)", re.U)
    if re_number.fullmatch(token):
        return True
    return False

#符号判断
def is_punctuation(token):
    # 匹配中英文常见标点符号
    def is_punctuation(token):
        # 英文标点
        if token in string.punctuation:
            return True
        # 常见中文标点
        if token in '，。！？；：“”‘’、…—（）《》【】':
            return True
        return False

#创建停止词
"""
def create_stop_words(file_path , output_file):
    create_dir(output_file)
    stop_words = set()
    for line in read_file(file_path):
        stop_words.add(line.strip().split(" ")[-1])
    with open(output_file, "w" ,encoding="utf-8") as fout:
        for word in stop_words:
            fout.writelines(word + '\n')
        pass
"""

#加载停止词
def load_stop_words(file_path):
    stop_words = set()
    for line in read_file(file_path):
        stop_words.add(line.strip())
    return stop_words

#划分数据集
def split_data(data_file , output_file):
    create_dir(output_file)
    data_text = []
    with open(data_file , 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        text = split_sentence(item['text'])
        data_text.append(text)
    for line in read_file(output_file):
        f.write(' '.join(line) + '\n')

#构建词表
def token_vocab(file_path,out_file = None , stop_words=None ,min_freq=2 ,default_token = None ,nuk_token = "<NUK>", num_token='<NUM>' ,punct_token = "<PUNCT>"):
    create_dir(out_file)
    vocab = {}
    for line in read_file(file_path):
        tokens = line.strip().split()
        for token in tokens:
            if is_number(token):
                token = num_token
            if is_punctuation(token):
                token = punct_token
            if token in stop_words:
                continue
            vocab[token] = vocab.get(token, 0) + 1
    #去除低频词
    vocab = {token: freq for token, freq in vocab.items() if freq >= min_freq}
    vocab = OrderedDict(vocab)
    #进行torchtext的词表构建
    vocab = torchtext.vocab.vocab(vocab)
    for token , token_freq in default_token.items():
        if token not in vocab:
            vocab.insert_token(token , token_freq)
    vocab.set_default_index(vocab[nuk_token])
    torch.save(vocab, out_file)
    return vocab

# 读取数据并查询所有意图
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    domain = set()
    intent = set()
    for item in data:
        domain.add(item['domain'])
        intent.add(item['intent'])
    domain = sorted(list(domain))
    intent = list(intent)
    label_vocab = torchtext.vocab.vocab(OrderedDict({label: idx+1 for idx, label in enumerate(domain)}))
    torch.save(label_vocab, 'output_data/label_vocab.pkl')
    return domain, intent

#构造数据集
class MyDataset(Dataset):
    def __init__(self, vocab_file, data_file,label_file):
        super(MyDataset, self).__init__()
        self.token_vocab = torch.load(vocab_file, map_location=torch.device('cpu'))
        self.label_vocab = torch.load(label_file, map_location=torch.device('cpu'))
        self.records = []
        for line in read_file(data_file):
            tokens , label = line["text"] , line["domain"]
            token = split_sentence(tokens)
            token_ides = self.token_vocab(token)
            label_ides = self.label_vocab([label])[0]
            token_new = self.token_vocab.lookup_tokens(token_ides)
            self.records.append((token , token_new , token_ides , label, label_ides , len(token_ides)))
            self.pad_token_id = self.token_vocab['<PAD>']


    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        return self.records[idx]
    def collate_fn(self, records,token_len = 16):
        #records: text:List[Tuple[List[str], text_new :List[str], token_ids:[N,M], label:list[str], label_id:[N]，seq_len:[N]]
        #N表示batch_size , M表示句子长度 ，需要对句子进行padding
        #seq_len表示实际上的句子长度
        seq_len = []
        batch_token = [record[2] for record in records]
        for pad_token in batch_token:
            pad_len = token_len - len(pad_token)
            if pad_len > 0:
                seq_len.append(len(pad_token))
                pad_token.extend([self.pad_token_id] * pad_len)
            else:
                del pad_token[token_len:]
                seq_len.append(token_len)
        batch_token = torch.tensor(batch_token, dtype=torch.long)
        batch_label = torch.tensor([record[4] for record in records], dtype=torch.long)
        seq_len = torch.tensor(seq_len, dtype=torch.long)
        return batch_token, batch_label ,seq_len

if __name__ == '__main__':
    # data_path = 'train_annal.json'
    # domain, intent = read_data(data_path)
    # print(f'域的数量：{len(domain)}，意图的数量：{len(intent)}')
    # print(f'域的种类：{domain}')
    # print(f'意图的种类：{intent}')

    # split_data("train_annal.json" , "output_data/train_split.txt")
    # token_vocab("output_data/train_split.txt")
    # create_stop_words("output_data/train_split.txt" , "output_data/stop_words.txt" )

    # stop_words = load_stop_words("output_data/stop_words.txt")
    # vocab = token_vocab("output_data/train_split.txt" , out_file = 'output_data/vocab.pkl',stop_words=stop_words, min_freq=2
    #                     , default_token={"<PAD>":0 , "<NUK>":1 , '<PUN>':2 , 'NUM':3}
    #                     , nuk_token="<NUK>", num_token='<NUM>' , punct_token = "<PUN>")
    # print(f'词表大小：{len(vocab)}')
    #
    dataset = MyDataset("output_data/vocab.pkl" , "train_annal.json" , "output_data/label_vocab.pkl")
    max = 0
    # for q,w,e,r,t,y in dataset:
    #     max = len(e) if len(e) > max else max
    # print(f'最大句子长度：{max}')

    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True , collate_fn=dataset.collate_fn )
    for batch in train_dataloader:
        print(batch)
        break
    # pass
