'''
定义一个执行器类，用于控制模型训练和评估的流程。
'''
import os
import torch
from torch.utils.tensorboard import SummaryWriter, FileWriter
from torchtext.vocab import vocab
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from data.data_buddling import train_eval_split, MyDataset, split_sentence
from metrics import accuracy_slef
from models import ModelTextClassify
from utils_project import create_dir


class Trainactuator(object):
    def __init__(self, vocab_file , data_file ,label_vocab_file,output_dir='output_data/train' ):
        super(Trainactuator, self ).__init__()
        #加载数据
        self.vocab = torch.load(vocab_file,map_location="cpu")
        self.label_vocab = torch.load(label_vocab_file,map_location="cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_summary_dir = os.path.join(output_dir,'summary')
        self.output_dir = os.path.join(output_dir,'model')
        self.data_file = data_file
        create_dir(self.output_dir)
        create_dir(self.output_summary_dir)

    def train_loop(self,dataloder , model, opt, loss_fn,train_batch_step ,epoch ,log_interval_batch =10,writer=None):
        model.train()
        loss_viwe = []
        acc_view = []
        for batch_token, batch_label, seq_len in dataloder:
            batch_token = batch_token.to(self.device)
            batch_label = batch_label.to(self.device)
            seq_len = seq_len.to(self.device)

            #前向计算
            scores = model(batch_token, seq_len)
            loss = loss_fn(scores, batch_label)
            loss_viwe.append(loss.item())
            acc = accuracy_slef(scores, batch_label)
            acc_view.append(acc)

            #反向传播
            opt.zero_grad()
            loss.backward()
            opt.step()
            #打印日志信息
            if (train_batch_step + 1) % log_interval_batch == 0:
                writer.add_scalar('loss', loss.item(), train_batch_step + 1)
                writer.add_scalar('acc', acc_view[-1], train_batch_step + 1)
                print(f" epoch:{epoch},batch_step:{train_batch_step}, train_loss:{loss.item()} , ACC:{acc}'")
            train_batch_step += 1
        writer.add_scalar('epoch_train_loss', np.mean(loss_viwe), epoch)
        writer.add_scalar('epoch_train_acc', np.mean(acc_view), epoch)


        pass

    def eval_loop(self,dataloder ,model, loss_fn,eval_batch_step ,epoch ,log_interval_batch =10,writer=None):
        model.eval()
        loss_viwe = []
        acc_view = []
        with torch.no_grad():
            for batch_token, batch_label, seq_len in dataloder:
                batch_token = batch_token.to(self.device)
                batch_label = batch_label.to(self.device)
                seq_len = seq_len.to(self.device)
                #前向计算
                scores = model(batch_token, seq_len)
                loss = loss_fn(scores, batch_label)
                #打印日志信息
                if (eval_batch_step + 1) % log_interval_batch == 0:
                    acc =accuracy_slef(scores, batch_label)
                    acc_view.append(acc)
                    writer.add_scalar('loss', loss.item(), eval_batch_step + 1)
                    writer.add_scalar('acc', acc_view[-1], eval_batch_step + 1)
                    print(f"epoch{epoch},batch_step:{eval_batch_step}, eval_loss:{loss.item()} , ACC:{acc}")
                eval_batch_step += 1
            writer.add_scalar('epoch_train_loss', np.mean(loss_viwe), epoch)
            writer.add_scalar('epoch_train_acc', np.mean(acc_view), epoch)


    def save_model(self, model, epoch ,opt ):
        obj = {
            'model_state_dict': model,
            'optimizer_state_dict': opt.state_dict(),
            'epoch': epoch
        }
        torch.save(obj, os.path.join(self.output_dir,'model_epoch_{}.pkl'.format(epoch)))



    def train(self, cfg, all_epoch , train_ratio, batch_size, shuffle,
              lr=0.001 ,save_model_epoch=10,log_interval_batch =10,
              ckpt_path = None
              ):
        '''
        训练函数
        该函数实现了从数据集构造、模型构造、训练过程、评估过程、模型保存等功能,可以从断点处恢复训练
        :param cfg: 配置参数
        :param all_epoch:  训练总轮次
        :param train_ratio:训练集比例
        :param batch_size:批次大小
        :param shuffle:是否打乱数据
        :param lr:学习率
        :param save_model_epoch:多少轮保存一次模型
        :param log_interval_batch:多少个batch打印一次日志
        :param ckpt_path:加载断点模型路径
        :return:
        '''
        #数据集构造
        dataset = MyDataset(self.vocab, self.data_file, self.label_vocab)
        train_dataloader, eval_dataloader = train_eval_split(dataset, train_ratio=train_ratio, batch_size=batch_size,
                                                             shuffle=shuffle, collate_fn=dataset.collate_fn)

        #参数恢复
        weights = None
        if ckpt_path and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            weights = checkpoint['model_state_dict']
            start_epoch = checkpoint['epoch']+1
            all_epoch = all_epoch + start_epoch
            print(f'从{ckpt_path}恢复训练，当前epoch为{start_epoch}')
        else:
            start_epoch = 0
            print('从头开始训练')

        #模型构造
        model =ModelTextClassify.bulid_model(cfg , weights)
        model = model.to(self.device)
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay= 0.01,momentum=0.9)
        loss = nn.CrossEntropyLoss()

        train_batch_step = 0
        eval_batch_step = 0
        #训练过程
        with SummaryWriter(log_dir=self.output_summary_dir) as writer:
            for epoch in range(start_epoch ,all_epoch):
                #训练
                self.train_loop(train_dataloader , model, opt, loss, train_batch_step=train_batch_step , epoch=epoch,log_interval_batch=log_interval_batch,writer=writer)
                #评估
                self.eval_loop(eval_dataloader , model, loss, eval_batch_step=eval_batch_step ,  epoch=epoch,log_interval_batch=log_interval_batch,writer=writer)
                #保存模型
                if (epoch + 1) % save_model_epoch == 0:
                    self.save_model(model, epoch, opt)
            #最终保存模型
            self.save_model(model, all_epoch, opt)
            print('Finished Training')

class Evaluateactuator(object):
    def __init__(self, cfg ,vocab_file , data_file ,label_vocab_file,ckpt_path ,output_dir='output_data/train',eval_dir ="output_data/eval" ):
        super(Evaluateactuator, self).__init__()
        self.vocab = torch.load(vocab_file, map_location="cpu")
        self.label_vocab = torch.load(label_vocab_file, map_location="cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = os.path.join(output_dir, 'model')
        self.data_file = data_file
        create_dir(self.output_dir)
        self.eval_dir =eval_dir
        create_dir(self.eval_dir)

        #恢复模型
        weights = None
        if ckpt_path and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            weights = checkpoint['model_state_dict']
            print(f'从{ckpt_path}恢复训练!')
        else:
            raise ValueError('评估模型不存在，请检查ckpt_path路径')

        model =ModelTextClassify.bulid_model(cfg , weights)
        self.model = model.to(self.device)
        self.model.eval()
        for parameters in self.model.parameters():
            parameters.requires_grad = False
        self.loss = nn.CrossEntropyLoss()

    def  eval(self,batch_size):
        #数据集构造
        dataset = MyDataset(self.vocab, self.data_file, self.label_vocab)
        _, eval_dataloader = train_eval_split(dataset, train_ratio=0.0, batch_size=batch_size,shuffle=False, collate_fn=dataset.collate_fn)
        with open (os.path.join(self.eval_dir,'eval_result.txt'), 'w', encoding='utf-8') as f:
            i= 0
            #遍历数据
            for  batch_token, batch_label, seq_len in eval_dataloader:
                batch_token = batch_token.to(self.device)
                batch_label = batch_label.to(self.device)
                seq_len = seq_len.to(self.device)
                # 前向计算
                scores = self.model(batch_token, seq_len)
                loss = self.loss(scores, batch_label)
                acc = accuracy_slef(scores, batch_label)
                print(f'eval_loss:{loss.item()} , ACC:{acc}')
                f.writelines(f'第{i}批次的原始文本是{batch_token}和标签是{batch_label}，\n'
                             f'损失是{loss}，acc是{acc}')

class Predictactuator(object):
    def __init__(self  ,cfg ,vocab_file  ,label_vocab_file,ckpt_path ):
        '''
        预测文本对应的top k标签
        :param text: 文本（str）
        :param k:  前k个最高概率的标签
        :return eg:[('标签1',0.9),('标签2',0.8)]
        '''
        super(Predictactuator, self).__init__()
        #加载数据
        self.vocab = torch.load(vocab_file, map_location="cpu")#加载词表
        self.label_vocab = torch.load(label_vocab_file, map_location="cpu")#加载标签表
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 恢复模型
        weights = None
        if ckpt_path and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            weights = checkpoint['model_state_dict']
            print(f'检测{ckpt_path}模型')
        else:
            raise ValueError('预测模型不存在，请检查ckpt_path路径')

        model = ModelTextClassify.bulid_model(cfg, weights)
        self.model = model.to(self.device)
        self.model.eval()
        for parameters in self.model.parameters():
            parameters.requires_grad = False

    def predict(self, text :str , k=3):
            #数据集构造
            split_sentences = split_sentence(text)
            token_ides = self.vocab(split_sentences)
            seq_len = len(token_ides)
            pad_len = 16 - len(token_ides)
            # padding
            if pad_len > 0:
                token_ides.extend([self.vocab['<PAD>']] * pad_len)
            else:
                print("输入文本过长，截断为前16个词")
                del token_ides[16:]
                seq_len = 16
            batch_token = torch.tensor([token_ides], dtype=torch.long).to(self.device)
            seq_len = torch.tensor([seq_len], dtype=torch.long).to(self.device)
            # 前向计算
            scores = self.model(batch_token, seq_len)
            probs = F.softmax(scores, dim=1)#计算每一个样本的所有概率
            topk_probs, topk_indices = torch.topk(probs, k=k, dim=1)
            topk_probs = topk_probs.cpu().numpy().flatten()
            topk_indices = topk_indices.cpu().numpy().flatten()
            idx_to_label = {idx: label for label, idx in self.label_vocab.get_stoi().items()}
            topk_labels = [idx_to_label[idx] for idx in topk_indices]
            return list(zip(topk_labels, topk_probs))

if __name__ == '__main__':
    vocab = torch.load('data/output_data/vocab.pkl')
    label_vocab = torch.load('data/output_data/label_vocab.pkl')
    cfg = {
        "token_embedding_layer": {
            "name": "CreatTokenEmbeddingLayer",
            # vocab_size, embed_size,
            "args": [len(vocab), 64]
        },
        "feteach_feature_layer": {
            "name": "RNNTextClassifyModel",
            # embed_size = 64,hidden_size = 64,num_layers = 1,batch_first = True,dropout = 0.1, bidirectional = True,rnn_output_type = 'all_mean',# "all_sum" "all_mean" "last"seq_len= None,
            "args": [64, 1, True, 0.1, True, "last"]
        },
        "classify_predict_layer": {
            "name": "MLPModule",
            # n_feature , out_feature, hidden_feature, dropout = 0.0, act = None,last_act =False,
            "args": [len(label_vocab), [256, 128], 0.1, "ReLU", False]
        }

    }
    vocab_file = "data/output_data/vocab.pkl"
    label_vocab_file = "data/output_data/label_vocab.pkl"
    data_file = "data/train_annal.json"
    output_dir = 'data/output_data/train'
    ckpt_path = "D:\github\Dialogue_Intent_Recognition\data\output_data\\train\model\model_epoch_5002.pkl"
    #,cfg ,vocab_file  ,label_vocab_file,ckpt_path
    model_evaluator = Predictactuator( cfg ,vocab_file ,label_vocab_file,ckpt_path )
    consult = model_evaluator.predict(text="我要去米其林餐厅吃饭", k=3)
    print(consult)