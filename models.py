'''
定义模型
'''
from jinja2.utils import missing
from torch import nn
from torch.nn.functional import dropout
from torchtext.vocab import vocab
from unicodedata import bidirectional
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy ,scipy ,pandas ,matplotlib
from common import *


#为了方便模型进行调试，建立一个模型的基类，总共三层，token_embedding_layer，feteach_feature_layer，classify_predict_layer
class ModelTextClassify(nn.Module):
    def __init__(self,
                 token_embedding_layer,
                 feteach_feature_layer,
                 classify_predict_layer
                 ):
        super(ModelTextClassify, self).__init__()
        #传入common.py中定义的三个层
        self.token_embedding_layer = token_embedding_layer
        self.feteach_feature_layer = feteach_feature_layer
        self.classify_predict_layer = classify_predict_layer
        pass


    def forward(self,x,seq_len):
        """
        前向传播
        :param x: 输入的文本，形状是[N,M] N表示batch_size , M表示句子长度
        :param y: 分类的结果
        :param seq_len: 类别标签
        :return: output : 分类的结果
        """
        #1.通过embedding层将x转换成词向量
        x = self.token_embedding_layer(x)
        #2.通过特征提取层提取文本的特征
        x = self.feteach_feature_layer(x,seq_len)
        #3.通过分类预测层得到最终的分类结果
        output = self.classify_predict_layer(x)
        return output

    @staticmethod
    def bulid_model( cfg , weights = None):
        #这里需要注意cfg的每一层的第一个参数都是上一个层的输出维度
        #构建token_embedding_layer：
        m = eval(cfg['token_embedding_layer']["name"])
        #eval 在 Python 里是一个内置函数，用于“执行字符串表达式”，并返回结果。比如 eval('1+2') 会返回 3。
        args = cfg['token_embedding_layer']["args"]
        token_embedding_layer = m(*args)
        #构建feteach_feature_layer
        m = eval(cfg['feteach_feature_layer']["name"])
        args = cfg['feteach_feature_layer']["args"]
        args.insert(0, token_embedding_layer.output_size)
        feteach_feature_layer = m(*args)
        #构建classify_predict_layer
        m = eval(cfg['classify_predict_layer']["name"])
        args = cfg['classify_predict_layer']["args"]
        args.insert(0, feteach_feature_layer.output_size)
        classify_predict_layer = m(*args)
        #构建整个模型
        model = ModelTextClassify(
            token_embedding_layer,
            feteach_feature_layer,
            classify_predict_layer)
        if weights is not None:
            if isinstance(weights, str):
                weights = torch.load(weights, map_location="cpu").state_dict()
            elif isinstance(weights, ModelTextClassify):
                weights = weights.state_dict()
        #missingkey是加载预训练模型时，缺失的key
        #unexpectedkey是加载预训练模型时，多余的key
            missingkey , unexpectedkey = model.load_state_dict(weights, strict=False)
            if len(missingkey) > 0:
                print(f'加载预训练模型时，缺失的key有：{missingkey}')
            if len(unexpectedkey) > 0:
                print(f'加载预训练模型时，多余的key有：{unexpectedkey}')
        return model

class RNNTextClassifyModel1(nn.Module):
    def __init__(self,vocab_size,
                 embed_size = 64,
                 hidden_size = 64,
                 num_layers = 1,
                 batch_first = True,
                 dropout = 0.1,
                 bidirectional = True,
                 rnn_output_type = 'last',# "all_sum" "all_mean" "last"
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
        self.rnn_output_type = rnn_output_type# "all_sum" "all_mean" "last"
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

class LSTMTextClassifyModel1(nn.Module):
    def __init__(self,vocab_size,
                 embed_size = 64,
                 hidden_size = 64,
                 num_layers = 1,
                 batch_first = True,
                 dropout = 0.1,
                 bidirectional = True,
                 rnn_output_type = 'last',# "all_sum" "all_mean" "last"
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
        self.lstm_output_type = rnn_output_type# "all_sum" "all_mean" "last"
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

class GRUTextClassifyMode1l(nn.Module):
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
    from common import CreatTokenEmbeddingLayer, RNNTextClassifyModel, LSTMTextClassifyModel, GRUTextClassifyModel, \
    ClassifyPredictLayer, MLPModule

    vocab = torch.load('data/output_data/vocab.pkl')
    label_vocab = torch.load('data/output_data/label_vocab.pkl')
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
    cfg = {
        "token_embedding_layer": {
            "name": "CreatTokenEmbeddingLayer",
            #vocab_size, embed_size,
            "args": [len(vocab), 64]
        },
        "feteach_feature_layer": {
            "name": "RNNTextClassifyModel",
            # embed_size = 64,hidden_size = 64,num_layers = 1,batch_first = True,dropout = 0.1, bidirectional = True,rnn_output_type = 'all_mean',# "all_sum" "all_mean" "last"seq_len= None,
            "args": [64, 1, True, 0.1, True, "last"]
        },
        "classify_predict_layer": {
            "name": "MLPModule",
            #n_feature , out_feature, hidden_feature, dropout = 0.0, act = None,last_act =False,
            "args": [len(label_vocab), [256, 128], 0.1, "ReLU", False]
        }

    }
    model = ModelTextClassify.bulid_model(cfg , weights = None)
    # token_embedding_layer= CreatTokenEmbeddingLayer(seq_len , vocab_size=len(vocab), embed_size=64,batch_first = True)
    # feteach_feature_layer = RNNTextClassifyModel(
    #              embed_size = 64,
    #              hidden_size = 64,
    #              num_layers = 1,
    #              batch_first =True,
    #              dropout = 0.1,
    #              bidirectional = True,
    #              rnn_output_type = "last",# "all_sum" "all_mean" "last"
    #              num_classes = len(label_vocab))
    # classify_predict_layer = MLPModule(in_feature=feteach_feature_layer.output_size, out_feature=len(label_vocab), hidden_feature=[256,128], dropout=0.1, act=nn.ReLU(), last_act=False)
    # feteach_feature_layer.forward = feteach_feature_layer.forward_scrip
    # model = ModelTextClassify(
    #     token_embedding_layer,
    #     feteach_feature_layer,
    #     classify_predict_layer)
    out = model(x, seq_len = seq_len)
    print(model)
    print(out)
    print(out.shape)
    #
    # model = torch.jit.script(model)
    # model.save('data/output_data/tempt_model.pt')