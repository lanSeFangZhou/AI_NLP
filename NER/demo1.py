import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_boradcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec-max_score_boradcast)))

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tag2ix = tag2ix
        self.target_size = len(tag2ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.target_size)
        self.transitions = nn.Parameter(torch.randn(self.target_size, self.target_size))
        self.transitions.data[tag2ix[START_TAG], :] = -10000
        self.transitions.data[:, tag2ix[END_TAG]] = -10000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.target_size), -10000)
        init_alphas[0][self.tag2ix[START_TAG]] = 0
        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.target_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.target_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag2ix[END_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag2ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag2ix[END_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        init_vars = torch.full((1, self.target_size), -10000)
        init_vars[0][self.tag2ix[START_TAG]] = 0
        forward_var = init_vars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.target_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag2ix[END_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag2ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

if __name__ == "__main__":
    START_TAG = "<s>"
    END_TAG = "<e>"
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4

    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]
    word2ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word2ix:
                word2ix[word] = len(word2ix)

    tag2ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, END_TAG: 4}

    model = BiLSTM_CRF(len(word2ix), tag2ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word2ix)
        precheck_tags = torch.tensor([tag2ix[t] for t in training_data[0][1]], dtype=torch.long)
        print(model(precheck_sent))

    for epoch in range(300):
        for sentence, tags in training_data:
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word2ix)
            targets = torch.tensor([tag2ix[t] for t in tags], dtype=torch.long)
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word2ix)
        print(model(precheck_sent))












