# -*-coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])

class TextCNN(nn.Module):
    def __init__(self,
                 num_classes,
                 num_embeddings=-1,
                 embedding_dim=128,
                 kernel_sizes = [3, 4, 5, 6],
                 num_channels=[256, 256, 256, 256],
                 embeddings_pretrained=None):
        '''
        :param num_classes: 输出维度
        :param num_embeddings:
        :param embedding_dim: 词向量特征维度
        :param kernel_sizes: CNN层卷积核大小
        :param num_channels: CNN层卷积核通道数
        :param embeddings_pretrained: embeddings pretrained参数，默认None
        '''
        super(TextCNN, self).__init__()
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings














