"""
定义通用模块功能
"""
import copy
import os
import torch.nn as nn
import torch
import random
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.vocab import vocab


class CreatTokenEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size ):
        '''
        创建token的embedding层，并对输入的x进行embedding
        :param x: padding后的输入序列，形状是[N,M]，N是batch_size，M是序列的最大长度
        :param seq_len:  每个序列的实际长度，形状是[N]
        :param vocab_size: 词表大小
        :param embed_size: embedding的维度
        '''
        super(CreatTokenEmbeddingLayer, self).__init__()
        self.token_embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.output_size = embed_size
        #先对x进行embedding


    def forward(self, x ):
        x_embed = self.token_embedding_layer(x)  # 形状是[N,M,embed_size]
        # pack_padded_sequence是用来处理变长序列的，将填充的部分去掉，构建一个packed sequence对象

        return x_embed

class RNNTextClassifyModel(nn.Module):
    def __init__(self,
                 embed_size = 64,
                 hidden_size = 64,
                 num_layers = 1,
                 batch_first = True,
                 dropout = 0.1,
                 bidirectional = True,
                 rnn_output_type = 'all_mean',# "all_sum" "all_mean" "last"
                 seq_len= None,
                 ):
        super(RNNTextClassifyModel, self).__init__()

        self.rnn_lay = nn.RNN(input_size=embed_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              dropout = dropout ,
                              bidirectional = bidirectional
                              )
        self.rnn_output_type = rnn_output_type# "all_sum" "all_mean" "last"
        self.output_size = hidden_size * (2 if bidirectional else 1)
        self.seq_len = seq_len
        self.batch_first = batch_first

    def forward(self, x,seq_len ):
        """
        前向传播
        :param x:表示输入 ，形状是[N,M] N表示batch_size , M表示句子长度
        :param y: 表示标签，形状是[N]
        :return:
        """
        x_nupadding = pack_padded_sequence(x, seq_len.cpu(), batch_first=self.batch_first,
                                                enforce_sorted=False)
        y_hat,hiden_lay  = self.rnn_lay(x_nupadding)
        #pad_packed_sequence是将packed sequence对象还原成填充的tensor
        seq_unpacked, lens_unpacked = pad_packed_sequence(y_hat, batch_first=True)
        """
        output：形状为 (batch, seq_len, num_directions * hidden_size)（如果 batch_first=True），表示每个时间步的隐藏状态。
        h_n：形状为 (num_layers * num_directions, batch, hidden_size)，表示每一层最后一个时间步的隐藏状态。重点记住只有最后一个时间步
        """
        #RNN输出的特征用所有的时间步的输出，然后求和做平均
        #输出用最后一个时间步的输出，但是这样最后一个的反向是全部都是0
        if self.rnn_output_type == 'all_sum':
            y_hat = torch.sum(seq_unpacked, dim=1)
        elif self.rnn_output_type == 'all_mean':
            y_hat = torch.mean(seq_unpacked, dim=1)
        elif self.rnn_output_type == 'last':
            if self.rnn_lay.bidirectional == True:
                y_hat = torch.concat([hiden_lay[-2,:,:] ,hiden_lay[-1,:,:]], dim=1)
            else:
                y_hat = hiden_lay[-1,:,:]
        return y_hat

    #因为torch.jit.script 不支持在 nn.Module 的属性（如 self.x_nupadding）中保存 PackedSequence 这样的动态对象。
    def forward_scrip(self, x  , seq_len):
        """
        前向传播
        :param x:表示输入 ，形状是[N,M] N表示batch_size , M表示句子长度
        :param y: 表示标签，形状是[N]
        :return:
        """
        # self.x_nupadding = pack_padded_sequence(x, self.seq_len, batch_first=self.batch_first,
        #                                         enforce_sorted=False)
        y_hat,hiden_lay  = self.rnn_lay(x)
        #pad_packed_sequence是将packed sequence对象还原成填充的tensor
        # seq_unpacked, lens_unpacked = pad_packed_sequence(y_hat, batch_first=True)
        seq_unpacked, lens_unpacked = y_hat ,seq_len
        """
        output：形状为 (batch, seq_len, num_directions * hidden_size)（如果 batch_first=True），表示每个时间步的隐藏状态。
        h_n：形状为 (num_layers * num_directions, batch, hidden_size)，表示每一层最后一个时间步的隐藏状态。重点记住只有最后一个时间步
        """
        #RNN输出的特征用所有的时间步的输出，然后求和做平均
        #输出用最后一个时间步的输出，但是这样最后一个的反向是全部都是0
        if self.rnn_output_type == 'all_sum':
            y_hat = torch.sum(seq_unpacked, dim=1)
        elif self.rnn_output_type == 'all_mean':
            y_hat = torch.mean(seq_unpacked, dim=1)
        elif self.rnn_output_type == 'last':
            if self.rnn_lay.bidirectional == True:
                y_hat = torch.concat([hiden_lay[-2,:,:] ,hiden_lay[-1,:,:]], dim=1)
            else:
                y_hat = hiden_lay[-1,:,:]
        return y_hat

class LSTMTextClassifyModel(nn.Module):
    def __init__(self,
                 embed_size = 64,
                 hidden_size = 64,
                 num_layers = 1,
                 batch_first = True,
                 dropout = 0.1,
                 bidirectional = True,
                 lstm_output_type = 'all_mean',# "all_sum" "all_mean" "last"
                 ):
        super(LSTMTextClassifyModel, self).__init__()
        self.lstm_lay = nn.RNN(input_size=embed_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              dropout = dropout ,
                              bidirectional = bidirectional
                              )
        self.lstm_output_type = lstm_output_type# "all_sum" "all_mean" "last"
        self.output_size = hidden_size * (2 if bidirectional else 1)
        self.batch_first = batch_first

    def forward(self, x , seq_len ):
        """
        前向传播
        :param x:表示输入 ，形状是[N,M] N表示batch_size , M表示句子长度
        :param y: 表示标签，形状是[N]
        :return:
        """
        x_nupadding = pack_padded_sequence(x, seq_len.cpu(), batch_first=self.batch_first,
                                                enforce_sorted=False)
        y_hat, hiden_cell = self.lstm_lay(x_nupadding)
        """
        这里对hiden_cell进行拆包
        hiden_cell是[num_layer*bidirectional ,batch , E] 的形状
        cell_lay是[num_layer*bidirectional ,batch , E] 的形状
        """
        hiden_lay, cell_lay = hiden_cell
        #pad_packed_sequence是将packed sequence对象还原成填充的tensor
        seq_unpacked, lens_unpacked = pad_packed_sequence(y_hat, batch_first=True)
        """
        output：形状为 (batch, seq_len, num_directions * hidden_size)（如果 batch_first=True），表示每个时间步的隐藏状态。
        h_n：形状为 (num_layers * num_directions, batch, hidden_size)，表示每一层最后一个时间步的隐藏状态。重点记住只有最后一个时间步
        """
        #RNN输出的特征用所有的时间步的输出，然后求和做平均
        #输出用最后一个时间步的输出，但是这样最后一个的反向是全部都是0
        if self.lstm_output_type == 'all_sum':
            y_hat = torch.sum(seq_unpacked, dim=1)
        elif self.lstm_output_type == 'all_mean':
            y_hat = torch.mean(seq_unpacked, dim=1)
        elif self.lstm_output_type == 'last':
            if self.lstm_lay.bidirectional == True:
                y_hat = torch.concat([hiden_lay[-2,:,:] ,hiden_lay[-1,:,:]], dim=1)
            else:
                y_hat = hiden_lay[-1,:,:]
        output = self.fc(y_hat)#[N , E] -> [N,num_classes]
        return output

class GRUTextClassifyModel(nn.Module):
    def __init__(self,vocab_size,
                 embed_size = 64,
                 hidden_size = 64,
                 num_layers = 1,
                 batch_first = True,
                 dropout = 0.1,
                 bidirectional = True,
                 gru_output_type = 'all_mean',# "all_sum" "all_mean" "last"
                 ):
        super(GRUTextClassifyModel, self).__init__()
        self.gru_lay = nn.RNN(input_size=embed_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              dropout = dropout ,
                              bidirectional = bidirectional
                              )
        self.gru_output_type = gru_output_type# "all_sum" "all_mean" "last"
        self.output_size = hidden_size * (2 if bidirectional else 1)
        self.batch_first = batch_first

    def forward(self, x  , seq_len):
        """
        前向传播
        :param x:表示输入 ，形状是[N,M] N表示batch_size , M表示句子长度
        :param y: 表示标签，形状是[N]
        :return:
        """
        x_nupadding = pack_padded_sequence(x, seq_len.cpu(), batch_first=self.batch_first,
                                                enforce_sorted=False)
        y_hat,hiden_lay  = self.rnn_lay(x_nupadding)
        #pad_packed_sequence是将packed sequence对象还原成填充的tensor
        seq_unpacked, lens_unpacked = pad_packed_sequence(y_hat, batch_first=True)
        """
        output：形状为 (batch, seq_len, num_directions * hidden_size)（如果 batch_first=True），表示每个时间步的隐藏状态。
        h_n：形状为 (num_layers * num_directions, batch, hidden_size)，表示每一层最后一个时间步的隐藏状态。重点记住只有最后一个时间步
        """
        #RNN输出的特征用所有的时间步的输出，然后求和做平均
        #输出用最后一个时间步的输出，但是这样最后一个的反向是全部都是0
        if self.gru_output_type == 'all_sum':
            y_hat = torch.sum(seq_unpacked, dim=1)
        elif self.gru_output_type == 'all_mean':
            y_hat = torch.mean(seq_unpacked, dim=1)
        elif self.gru_output_type == 'last':
            if self.gru_lay.bidirectional == True:
                y_hat = torch.concat([hiden_lay[-2,:,:] ,hiden_lay[-1,:,:]], dim=1)
            else:
                y_hat = hiden_lay[-1,:,:]
        output = self.fc(y_hat)#[N , E] -> [N,num_classes]
        return output

class ClassifyPredictLayer(nn.Module):
    def __init__(self, hidden_size ,bidirectional, dropout,num_classes,):
        '''
        创建分类的预测层，并对输入的x进行预测
        :param x: 输入的特征，形状是[N,E]，N是batch_size，E是特征的维度
        '''
        super(ClassifyPredictLayer, self).__init__()
        self.classify_predict = nn.Sequential(
            nn.Linear(in_features=hidden_size * (2 if bidirectional else 1), out_features=256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x ,):
        output = self.classify_predict(x)
        return output

class FCModule(nn.Module):
    def __init__(self, in_feature , out_feature,dropout = 0.0 ,act = None,):
        super(FCModule, self).__init__()
        act = nn.ReLU() if act is not None else copy.deepcopy(act)#当用户不指定 act 参数时（即 act=None），默认使用 ReLU 激活函数。
        act = None if not act else act#当用户明确传递一个"假值"（如 False, 0 等）给 act 参数时，不使用任何激活函数（使用 nn.Identity()）。
        self.fc = nn.Linear(in_feature, out_feature)
        self.act = nn.Identity() if act is not None else act#自定义激活函数：当用户传递一个具体的激活函数（如 nn.Tanh(), nn.Sigmoid() 等）时，使用该激活函数。
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
    def forward(self, x ,):
        x = self.dropout(self.act(self.fc(x)))
        return x

class MLPModule(nn.Module):
    def __init__(self, in_feature , out_feature, hidden_feature, dropout = 0.0, act = None,last_act =False, ):
        super(MLPModule, self).__init__()
        if hidden_feature is None:
            hidden_feature = []

        layers = []
        for i in hidden_feature:
            layers.append(FCModule(in_feature, i, dropout = dropout, act = act))
            in_feature = i
        layers.append(FCModule(in_feature, out_feature, dropout = dropout, act = last_act))
        self.layers = nn.Sequential(*layers)
        self.output_size = out_feature

    def forward(self, x):
        x = self.layers(x)
        return x

