import math
import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# # NMT任务的训练阶段，这些参数分别设置为：
# src_mask = None
# tgt_mask = tgt_mask
# memory_mask = None
# src_key_padding_mask = src_key_padding_mask
# tgt_key_padding_mask = tgt_key_padding_mask
# memory_key_padding_mask = src_key_padding_mask
#
# # NMT任务的推理阶段，这些参数的设置分别为：
# src_mask = None
# tgt_mask = tgt_mask
# memory_mask = None
# src_key_padding_mask = src_key_padding_mask
# tgt_key_padding_mask = None
# memory_key_padding_mask = src_key_padding_mask

# 定义seq2seqModel:
class Seq2SeqModel(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1
                 ):
        super().__init()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout)

        self.transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self,
                src,
                tgt,
                src_mask=None,
                tgt_mask=None,
                memory_mask=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        '''
        :param src: (N, S)
        :param tgt: (N, T)
        :param src_mask: (N, S)
        :param tgt_mask: (T, T)
        :param memory_mask: (N, S)
        :param src_key_padding_mask: (N, S)
        :param tgt_key_padding_mask: (N, T)
        :param memory_key_padding_mask: (N, S)
        :return:
        '''
        src = self.pe(self.src_embedding(src).transpose(0, 1) * math.sqrt(self.d_model)) # (S, N, E)
        tgt = self.pe(self.tgt_embedding(tgt).transpose(0, 1) * math.sqrt(self.d_model)) # (T, N, E)
        transformer_output = self.transformer(src=src,
                                              tgt=tgt,
                                              src_mask=src_mask,
                                              tgt_mask=tgt_mask,
                                              memory_mask=memory_mask,
                                              src_key_padding_mask=src_key_padding_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask,
                                              memory_key_padding_mask=memory_key_padding_mask) # (T, N, E)
        logits = self.out(transformer_output) # (T, N, tgt_vocab_size)
        return logits


    # 在inference阶段，我们需要单独使用Transformer的Encoder和Decoder，因此我们还需为Seq2SeqModel定义encoder和decoder方法：
    def encoder(self, src, src_mask=None, src_key_padding_mask=None):
        '''
        :param src: (N, S)
        :param src_mask:
        :param sec_key_padding_mask:
        :return:
        '''
        src = self.pe(self.src_embedding(src).transpose(0, 1) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src, src_mask, src_key_padding_mask)
        return memory

    def decoder(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        '''
        :param tgt: (N, T)
        :param memory:
        :param tgt_mask:
        :param memory_mask:
        :param tgt_key_padding_mask:
        :param memory_key_padding_mask:
        :return:
        '''
        tgt = self.pe(self.tgt_embedding(tgt).transpose(0, 1) * math.sqrt(self.d_model))
        decoder_output = self.transformer.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        logits = self.out(decoder_output)
        return logits

# 4.1 训练和推理
def train(train_loader, model, criterion, optimizer, num_epochs):
    train_loss = []
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (encoder_inputs, decoder_targets) in enumerate(train_loader):

            encoder_inputs, decoder_targets = encoder_inputs.to(device), decoder_targets.to(device)
            bos_column = torch.tensor([tgt_vocab['<bos>']]) * decoder_targets.shape[0].reshape(-1, 1).to(device)
            decoder_inputs = torch.cat((bos_column, decoder_targets[:, :-1]), dim=1)

            tgt_mask = model.transformer.generate_square_subsequent_mask(SEQ_LEN)
            src_key_padding_mask = encoder_inputs == 1 # 因为padding_idx=1
            tgt_key_padding_mask = decoder_inputs == 1

            pred = model(encoder_inputs,
                         decoder_inputs,
                         tgt_mask=tgt_mask.to(device),
                         src_key_padding_mask=src_key_padding_mask.to(device),
                         tgt_key_padding_mask=tgt_key_padding_mask.to(device),
                         memory_key_padding_mask=src_key_padding_mask.to(device)
                         )
            loss = criterion(pred.permute(1, 2, 0), decoder_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if (batch_idx + 1) % 50 == 0:
                print(
                    f'[Epoch {epoch + 1}] [{(batch_idx + 1) * len(encoder_inputs)}/{len(train_loader.dataset)}] loss: {loss:.4f}'
                )
        print()
    return train_loss

@torch.no_grad
def translate(test_loader, model):
    translation_results = []
    model.eval()
    for src_seq, tgt_seq in test_loader:
        encoder_inputs = src_seq.to(device)
        src_key_padding_mask = encoder_inputs == 1
        memory = model.encoder(encoder_inputs, src_key_padding_mask=src_key_padding_mask)
        pred_seq = [tgt_vocab['<bos>']]
        for _ in range(SEQ_LEN):
            decoder_inputs = torch.tensor(pred_seq).reshape(1, -1).to(device) # 注意是pred_seq而不是pred_seq[-1]
            tgt_mask = model.transformer.generate_square_subsequent_mask(len(pred_seq))
            pred = model.decoder(
                decoder_inputs,
                memory,
                tgt_mask=tgt_mask.to(device),
                memory_key_padding_mask=src_key_padding_mask.to(device) # (len(pred_seq), 1, tgt_vocab_size)
            )
            next_token_idx = pred[-1].squeeze().argmax().item()  # 选取输出序列的最后一个词元
            if next_token_idx == tgt_vocab['<eos>']:
                break
            pred_seq.append(next_token_idx)
        pred_seq = tgt_vocab[pred_seq[1:]]
        assert len(pred_seq) > 0, "The predicted sequence is empty!"
        tgt_seq = tgt_seq.sequeeze().tolist()
        tgt_seq = tgt_vocab[
            tgt_seq[:tgt_seq.index(tgt_vocab['<eos>'])]] if tgt_vocab['<eos>'] in tgt_seq else tgt_vocab[tgt_seq]
        translation_results.append((' '.join(tgt_seq), ' '.join(pred_seq)))
    return translation_results

# 定义一个函数用来计算BLEU得分
def evaluate(translation_results, bleu_k_list=[2, 3, 4]):
    assert type(bleu_k_list) == list and len(bleu_k_list) > 0
    bleu_scores = {k: [] for k in sorted(bleu_k_list)}
    for bleu_k in bleu_scores.keys():
        for tgt_seq, pred_seq in translation_results:
            if len(pred_seq) >= bleu_k:
                bleu_scores[bleu_k].append(bleu(tgt_seq, pred_seq, k=bleu_k))
    for bleu_k in bleu_scores.keys():
        bleu_scores[blue_k] = np.mean(bleu_scores[bleu_k])
    return bleu_scores

# 之后开始跑分
# parameter settings
set_seed()
BATCH_SIZE = 512
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50

# Dataloader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)

# Model building
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Seq2SeqModel(len(src_vocab), len(tgt_vocab)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=1)
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

# Training phase
train_loss = train(train_loader, net, criterion, optimizer, NUM_EPOCHS)
torch.save(net.state_dict(), './prams/trans_seq2seq.pt')
plt.plot(train_loss)
plt.ylabel("train_loss")
plt.savefig('./output/loss.png')

# Evaluation
translation_results = translate(test_loader, net)
bleu_scores = evaluate(translation_results)
print(f"BLEU-2: {bleu_scores[2]} | BLEU-3: {bleu_scores[3]} | BLEU-4: {bleu_scores[4]}")












