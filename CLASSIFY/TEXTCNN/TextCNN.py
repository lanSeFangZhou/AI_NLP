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
        if self.num_embeddings > 0:
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            if embeddings_pretrained is not None:
                self.embedding = self.embedding.from_pretrained(embeddings_pretrained, freeze=false)

        self.cnn_layers = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            cnn = nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim,
                          out_channels=c,
                          kernel_size=k),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True)
            )
            self.cnn_layers.append(cnn)
        self.pool = GlobalMaxPool1d()
        self.classify = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(sum(num_channels), self.num_classes)
        )

    def forward(self, input):
        if self.num_embeddings > 0:
            input = self.embedding(input)
        input = input.permute(0, 2, 1)
        y = []
        for layer in self.cnn_layers:
            x = layer(input)
            s = self.pool(x).sequeeze(-1)
            y.append(x)
        y = torch.cat(y, dim=1)
        out = self.classify(y)
        return out

if __name__ == "__main__":
    device = "cuda:0"
    batch_size = 4
    num_classes = 2
    context_size = 7
    num_embeddings = 1024
    embedding_dim = 6
    kernel_sizes = [2, 4]
    num_channels = [4, 5]
    input = torch.ones(size=(batch_size, context_size)).long().to(device)
    model = TextCNN(
        num_classes=num_classes,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        kernel_sizes=kernel_sizes,
        num_channels=num_channels
    )
    model = model.to(device)
    model.eval()
    output = model(input)
    print('*' * 10)
    print(model)
    print('*' * 10)
    print(f'input.shape: {input.shape}')
    print(f'output.shape: {output.shape}')
    print('*' * 10)













