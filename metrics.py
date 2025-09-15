'''
置信度计算
'''

import torch


def accuracy_slef(scores, target):
    '''
    计算准确率
    :param scores: 预测分数 [batch_size, n_class]
    :param target: 真实标签 [batch_size]
    :return: 准确率
    '''
    _, pred = torch.max(scores, dim=1)
    correct = (pred == target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy