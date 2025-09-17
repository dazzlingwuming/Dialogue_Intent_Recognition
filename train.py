'''
训练
'''
import torch

from actuator import Trainactuator


def train_model():
    vocab = torch.load('data/output_data/vocab.pkl')
    label_vocab = torch.load('data/output_data/label_vocab.pkl')

    trainer = Trainactuator(
        vocab_file =   "data/output_data/vocab.pkl",
        label_vocab_file= "data/output_data/label_vocab.pkl" ,
        data_file= "data/train_annal.json",
        output_dir='data/output_data/train'
                            )
    cfg = {
        "token_embedding_layer": {
            "name": "CreatTokenEmbeddingLayer",
            #vocab_size, embed_size,
            "args": [ len(vocab), 64]
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
    trainer.train(
        cfg = cfg,
        all_epoch = 200,
        train_ratio =0.9,
        batch_size = 512,
        shuffle = True,
        lr=0.001,
        save_model_epoch=100,
        log_interval_batch=1,
        ckpt_path="D:\github\Dialogue_Intent_Recognition\data\output_data\\train\model\model_epoch_5002.pkl"
    )
if __name__ == '__main__':
    train_model()