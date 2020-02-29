import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchcrf import CRF
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
    def __init__(self, input_size=50, tagset_size=6, hidden_dim=100):
        super(BiLSTM_CRF, self).__init__()

        self.hidden_size = 2 * hidden_dim
        self.tagset_size = tagset_size

        self.lstm = nn.LSTM(input_size, self.hidden_size // 2, num_layers=1 ,bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_size, self.tagset_size)
        self.crf = CRF(self.tagset_size)

    def _get_lstm_features(self, x, hx):
        lstm_out, hidden = self.lstm(x, hx)
        return self.hidden2tag(lstm_out)

    def forward(self, x, hx = None):
        lstm_feats = self._get_lstm_features(x, hx)
        output = self.crf.decode(lstm_feats)
        output = F.one_hot(torch.tensor(output),6).type(torch.FloatTensor)
        output.requires_grad = True
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
