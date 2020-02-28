import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from TorchCRF import CRF
from transformers import BertConfig, BertTokenizer, BertModel
from kdtools.layers import BiLSTMEncoder

class BiLSTMDoubleDenseOracleParser(nn.Module):
    def __init__(self,
        actions_no,
        relations_no,
        *args,
        **kargs
    ):
        super().__init__()
        self.bilstmencoder_sent = BiLSTMEncoder(*args, **kargs)
        self.bilstmencoder_stack = BiLSTMEncoder(*args, **kargs)

        dense_input_size = self.bilstmencoder_sent.hidden_size + self.bilstmencoder_stack.hidden_size

        self.action_dense = nn.Linear(dense_input_size, actions_no)
        self.relation_dense = nn.Linear(dense_input_size, relations_no)

    def forward(self, x):
        stack_encoded, _ = self.bilstmencoder_stack(x[0])
        sent_encoded, _ = self.bilstmencoder_sent(x[1])

        encoded = torch.cat([stack_encoded, sent_encoded], 1)

        action_out = F.softmax(self.action_dense(encoded), 1)
        relation_out = F.softmax(self.relation_dense(encoded), 1)

        return [action_out, relation_out]


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size=100, tagset_size=7 ,embedding_dim=100, hidden_dim=768):
        super(BiLSTM_CRF, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_size = 2 * hidden_dim
        #self.vocab_size = vocab_size
        #self.tag_to_ix = tag_to_ix
        self.tagset_size = tagset_size

        #self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size // 2, num_layers=1 ,bidirectional=True)

        self.hidden2tag = nn.Linear(self.hidden_size, self.tagset_size)

        self.crf = CRF(self.tagset_size)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_size // 2),
                torch.randn(2, 1, self.hidden_size // 2))

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        #embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(sentence, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        output = self.crf.forward(lstm_feats)
        return output



def testBiLSTM():
    print("Entro")
    words = torch.Tensor([3]).view(1,-1,1)
    print(words)

    dropout = "std"
    crf = True
    bigram = True
    embedd_dim = 100
    char_dim = 30
    mode = "LSTM"
    hidden_size = 256
    out_features = 128
    num_layers = 1
    p_in = 0.33
    p_out = 0.5
    p_rnn = [0.33, 0.5]
    activation = "elu"

    model = BiRecurrentConvCRF(1, mode, hidden_size, out_features, num_layers,num_labels=2, p_in=p_in, p_out=p_out, p_rnn=p_rnn, bigram=bigram, activation=activation)

    print(model)
    outputs = model(words)
    print(outputs.shape)


if __name__ == "__main__":
    testBiLSTM()
