import numpy
import torch

from data.data_buddling import split_sentence

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = torch.load('data/output_data/vocab.pkl')
    ckpt_path =  "D:\github\Dialogue_Intent_Recognition\data\output_data\\train\model\model_epoch_5203.pkl"
    checkpoint = torch.load(ckpt_path, map_location=device)
    weights = checkpoint['model_state_dict']