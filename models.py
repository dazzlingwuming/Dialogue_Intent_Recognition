'''
定义模型
'''
from torch import nn
from torch.nn.functional import dropout
from torchtext.vocab import vocab
from unicodedata import bidirectional
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy ,scipy ,pandas ,matplotlib


class RNNTextClassifyModel(nn.Module):
    def __init__(self,vocab_size,
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
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn_lay = nn.RNN(input_size=embed_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              dropout = dropout ,
                              bidirectional = bidirectional
                              )
        self.rnn_output_type = 'last'# "all_sum" "all_mean" "last"
        self.fc = nn.Sequential(
            nn.Linear(in_features=hidden_size * (2 if bidirectional else 1),out_features= 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=256,out_features= 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=128,out_features= num_classes)
        )
    def forward(self, x , y , seq_len ):
        """
        前向传播
        :param x:表示输入 ，形状是[N,M] N表示batch_size , M表示句子长度
        :param y: 表示标签，形状是[N]
        :return:
        """
        #先对x进行embedding
        x_embed = self.embedding(x)#形状是[N,M,embed_size]
        #pack_padded_sequence是用来处理变长序列的，将填充的部分去掉，构建一个packed sequence对象
        x_nupadding = pack_padded_sequence(x_embed, seq_len, batch_first=True,enforce_sorted=False)
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
        output = self.fc(y_hat)#[N , E] -> [N,num_classes]
        return output

class LSTMTextClassifyModel(nn.Module):
    def __init__(self,vocab_size,
                 embed_size = 64,
                 hidden_size = 64,
                 num_layers = 1,
                 batch_first = True,
                 dropout = 0.1,
                 bidirectional = True,
                 rnn_output_type = 'all_mean',# "all_sum" "all_mean" "last"
                 num_classes = 20
                 ):
        super(LSTMTextClassifyModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm_lay = nn.LSTM(input_size=embed_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              dropout = dropout ,
                              bidirectional = bidirectional
                              )
        self.lstm_output_type = 'last'# "all_sum" "all_mean" "last"
        self.fc = nn.Sequential(
            nn.Linear(in_features=hidden_size * (2 if bidirectional else 1),out_features= 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=256,out_features= 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=128,out_features= num_classes)
        )

    def forward(self, x, y, seq_len):
        """
        前向传播
        :param x:表示输入 ，形状是[N,M] N表示batch_size , M表示句子长度
        :param y: 表示标签，形状是[N]
        :return:
        """
        # 先对x进行embedding
        x_embed = self.embedding(x)  # 形状是[N,M,embed_size]
        # pack_padded_sequence是用来处理变长序列的，将填充的部分去掉，构建一个packed sequence对象
        x_unpadding = pack_padded_sequence(x_embed, seq_len, batch_first=True, enforce_sorted=False)
        """
        #这里经过lstm后会返回两个值，
        # 第一个是所有时间步的输出 y_hat-> [N，T,E*bidirectional] 但是因为经过了紧凑后产生的结果是[N*T,E] ,
        # 第二个是最后一个时间步的隐藏状态和细胞状态hiden_cell，隐藏态和细胞态的维度是由hidden_size决定，且维度一致，
        """
        y_hat, hiden_cell = self.lstm_lay(x_unpadding)
        """
        这里对hiden_cell进行拆包
        hiden_cell是[num_layer*bidirectional ,batch , E] 的形状
        cell_lay是[num_layer*bidirectional ,batch , E] 的形状
        """
        hiden_lay, cell_lay = hiden_cell
        # pad_packed_sequence是将packed sequence对象还原成填充的tensor
        seq_unpacked, lens_unpacked = pad_packed_sequence(y_hat, batch_first=True)
        # LSTM输出的特征用所有的时间步的输出，然后求和做平均
        # 输出用最后一个时间步的输出，但是这样最后一个的反向是全部都是0
        if self.lstm_output_type == 'all_sum':
            y_hat = torch.sum(seq_unpacked, dim=1)
        elif self.lstm_output_type == 'all_mean':
            y_hat = torch.mean(seq_unpacked, dim=1)
        elif self.lstm_output_type == 'last':
            if self.lstm_lay.bidirectional == True:
                y_hat = torch.concat([hiden_lay[-2, :, :], hiden_lay[-1, :, :]], dim=1)
            else:
                y_hat = hiden_lay[-1, :, :]
        output = self.fc(y_hat)  # [N , E] -> [N,num_classes]
        return output

class GRUTextClassifyModel(nn.Module):
    def __init__(self,vocab_size,
                 embed_size = 64,
                 hidden_size = 64,
                 num_layers = 1,
                 batch_first = True,
                 dropout = 0.1,
                 bidirectional = True,
                 rnn_output_type = 'all_mean',# "all_sum" "all_mean" "last"
                 num_classes = 20
                 ):
        super(GRUTextClassifyModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.gru_lay = nn.GRU(input_size=embed_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=batch_first,
                              dropout = dropout ,
                              bidirectional = bidirectional
                              )
        self.gru_output_type = 'last'# "all_sum" "all_mean" "last"
        self.fc = nn.Sequential(
            nn.Linear(in_features=hidden_size * (2 if bidirectional else 1),out_features= 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=256,out_features= 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=128,out_features= num_classes)
        )
    def forward(self, x , y , seq_len ):
        """
        前向传播
        :param x:表示输入 ，形状是[N,M] N表示batch_size , M表示句子长度
        :param y: 表示标签，形状是[N]
        :return:
        """
        #先对x进行embedding
        x_embed = self.embedding(x)#形状是[N,M,embed_size]
        #pack_padded_sequence是用来处理变长序列的，将填充的部分去掉，构建一个packed sequence对象
        x_nupadding = pack_padded_sequence(x_embed, seq_len, batch_first=True,enforce_sorted=False)
        y_hat,hiden_lay  = self.gru_lay(x_nupadding)
        #pad_packed_sequence是将packed sequence对象还原成填充的tensor
        seq_unpacked, lens_unpacked = pad_packed_sequence(y_hat, batch_first=True)

        """
        y_hat：形状为 (batch, seq_len, num_directions * hidden_size)
        hiden_lay：形状为 (num_layers * num_directions, batch, hidden_size)，表示每一层最后一个时间步的隐藏状态。重点记住只有最后一个时间步
        """
        if self.gru_output_type == 'all_sum':
            y_hat = torch.sum(seq_unpacked, dim=1)
        elif self.gru_output_type == 'all_mean':
            y_hat = torch.mean(seq_unpacked, dim=1)
        elif self.gru_output_type == 'last':
            if self.gru_lay.bidirectional == True:
                y_hat = torch.concat([hiden_lay[-2, :, :], hiden_lay[-1, :, :]], dim=1)
            else:
                y_hat = hiden_lay[-1, :, :]
        output = self.fc(y_hat)  # [N , E] -> [N,num_classes]
        return output

if __name__ == '__main__':
    vocab = torch.load('data/output_data/vocab.pkl')
    label_vocab = torch.load('data/output_data/label_vocab.pkl')
    model = GRUTextClassifyModel(
        vocab_size=len(vocab),
        num_classes=len(label_vocab),)
    x = torch.tensor([[ 35,   1,   1,   1, 419,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0],
        [  1,  42,  89,  44,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0],
        [673,   1,   1,   1, 674, 675,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0],
        [ 21,   1,  39,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0]])
    y = torch.tensor([20,  4, 20, 14])
    seq_len = torch.tensor([5,4,6,3])
    out = model(x,y , seq_len)
    print(out)