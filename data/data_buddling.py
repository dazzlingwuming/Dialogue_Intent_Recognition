"""
构造数据集
"""
import os
import json
import torch
from torch.utils.data import DataLoader , Dataset


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
    return domain, intent

if __name__ == '__main__':
    data_path = 'train_annal.json'
    domain, intent = read_data(data_path)
    print(f'域的数量：{len(domain)}，意图的数量：{len(intent)}')
    print(f'域的种类：{domain}')
    print(f'意图的种类：{intent}')