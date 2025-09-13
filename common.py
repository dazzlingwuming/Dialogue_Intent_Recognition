"""
定义通用模块功能
"""
import os
import torch.nn as nn
import torch
import random
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.vocab import vocab


class CreatTokenEmbeddingLayer(nn.Module):
    def __init__(self, x , seq_len , vocab_size, embed_size,batch_first, **kwargs):
        '''
        创建token的embedding层，并对输入的x进行embedding
        :param x: padding后的输入序列，形状是[N,M]，N是batch_size，M是序列的最大长度
        :param seq_len:  每个序列的实际长度，形状是[N]
        :param vocab_size: 词表大小
        :param embed_size: embedding的维度
        '''
        super(CreatTokenEmbeddingLayer, self).__init__()
        self.token_embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        #先对x进行embedding
        x_embed = self.token_embedding_layer(x)#形状是[N,M,embed_size]
        #pack_padded_sequence是用来处理变长序列的，将填充的部分去掉，构建一个packed sequence对象
        self.x_nupadding = pack_padded_sequence(x_embed, seq_len, batch_first=True,enforce_sorted=False)

    def forward(self, **kwargs):
            return self.x_nupadding

class RNNTextClassifyModel(nn.Module):
    def __init__(self,
                 embed_size = 64,
                 hidden_size = 64,
                 num_layers = 1,
                 batch_first = True,
                 dropout = 0.1,
                 bidirectional = True,
                 rnn_output_type = 'all_mean',# "all_sum" "all_mean" "last"
                 num_classes = 20
                 ):
        super(RNNTextClassifyModel, self).__init__()
        self.rnn_lay = nn.RNN(input_size=embed_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              dropout = dropout ,
                              bidirectional = bidirectional
                              )
        self.rnn_output_type = 'last'# "all_sum" "all_mean" "last"

    def forward(self, x , y , seq_len ):
        """
        前向传播
        :param x:表示输入 ，形状是[N,M] N表示batch_size , M表示句子长度
        :param y: 表示标签，形状是[N]
        :return:
        """
        y_hat,hiden_lay  = self.rnn_lay(x)
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

class LSTMTextClassifyModel(nn.Module):
    def __init__(self,vocab_size,
                 embed_size = 64,
                 hidden_size = 64,
                 num_layers = 1,
                 batch_first = True,
                 dropout = 0.1,
                 bidirectional = True,
                 lstm_output_type = 'all_mean',# "all_sum" "all_mean" "last"
                 num_classes = 20
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

    def forward(self, x , y , seq_len ):
        """
        前向传播
        :param x:表示输入 ，形状是[N,M] N表示batch_size , M表示句子长度
        :param y: 表示标签，形状是[N]
        :return:
        """
        y_hat, hiden_cell = self.lstm_lay(x)
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
                 num_classes = 20
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

    def forward(self, x , y , seq_len ):
        """
        前向传播
        :param x:表示输入 ，形状是[N,M] N表示batch_size , M表示句子长度
        :param y: 表示标签，形状是[N]
        :return:
        """
        y_hat,hiden_lay  = self.rnn_lay(x)
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
    def __init__(self, hidden_size ,bidirectional, dropout,num_classes,**kwargs):
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

    def forward(self, x ,**kwargs):
        output = self.classify_predict(x)
        return output
