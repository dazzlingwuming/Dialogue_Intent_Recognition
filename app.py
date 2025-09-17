'''
使用 Flask 创建一个简单的 Web 应用程序
'''
import torch
from flask import Flask ,request,jsonify
from actuator import Predictactuator
import numpy as np

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
ckpt_path = "D:\github\Dialogue_Intent_Recognition\data\output_data\\train\model\model_epoch_5203.pkl"
# ,cfg ,vocab_file  ,label_vocab_file,ckpt_path
model_evaluator = Predictactuator(cfg, vocab_file, label_vocab_file, ckpt_path)

def model_evaluator_user(text, k=3):
    consult = model_evaluator.predict(text=text, k=k)
    return consult

app = Flask(__name__)
@app.route('/')
def index():
    return "Hello, this is a simple Flask !你是一个聊天机器人，请问有什么您？"
@app.route('/classify' , methods=['GET','POST'])
def classify():
    #获取当前的请求数据
    if request.method == 'POST':
        args = request.form
    else:
        args = request.args
    #获取参数
    text = args.get('text' , '' )
    k = int(args.get('k' , 3))
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    else:
        #结果返回
        result =model_evaluator_user(text , k)
        user_result = {
            "text": text,
            "topk": {result[0][0]:float(result[0][1]), result[1][0]:float(result[1][1])}
        }

        return jsonify({
            'status': 'success',
            'data': user_result
        })

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True
    )
