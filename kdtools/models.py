import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers import BertConfig, BertTokenizer, BertModel

import kdtools
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


class ChainCRF(nn.Module):
    def __init__(self, input_size, num_labels, bigram=True):
        super(ChainCRF, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels + 1
        self.pad_label_id = num_labels
        self.bigram = bigram


        # state weight tensor
        self.state_net = nn.Linear(input_size, self.num_labels)
        if bigram:
            # transition weight tensor
            self.transition_net = nn.Linear(input_size, self.num_labels * self.num_labels)
            self.register_parameter('transition_matrix', None)
        else:
            self.transition_net = None
            self.transition_matrix = Parameter(torch.Tensor(self.num_labels, self.num_labels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.state_net.bias, 0.)
        if self.bigram:
            nn.init.xavier_uniform_(self.transition_net.weight)
            nn.init.constant_(self.transition_net.bias, 0.)
        else:
            nn.init.normal_(self.transition_matrix)

    def forward(self, input, mask=None):
        batch, length, _ = input.size()

        # compute out_s by tensor dot [batch, length, model_dim] * [model_dim, num_label]
        # thus out_s should be [batch, length, num_label] --> [batch, length, num_label, 1]
        out_s = self.state_net(input).unsqueeze(2)

        if self.bigram:
            # compute out_s by tensor dot: [batch, length, model_dim] * [model_dim, num_label * num_label]
            # the output should be [batch, length, num_label,  num_label]
            out_t = self.transition_net(input).view(batch, length, self.num_labels, self.num_labels)
            output = out_t + out_s
        else:
            # [batch, length, num_label, num_label]
            output = self.transition_matrix + out_s

        if mask is not None:
            output = output * mask.unsqueeze(2).unsqueeze(3)

        return output


class BiRecurrentConvCRF(nn.Module):
    def __init__(self, word_dim, rnn_mode, hidden_size, out_features, num_layers,
                 num_labels, p_in=0.33, p_out=0.5, p_rnn=(0.5, 0.5), bigram=False, activation='elu'):
        super(BiRecurrentConvCRF, self).__init__()

        self.dropout_in = nn.Dropout2d(p=p_in)
        # standard dropout
        self.dropout_rnn_in = nn.Dropout(p=p_rnn[0])
        self.dropout_out = nn.Dropout(p_out)

        if rnn_mode == 'RNN':
            RNN = nn.RNN
        elif rnn_mode == 'LSTM' or rnn_mode == 'FastLSTM':
            RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        self.rnn = RNN(word_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn[1])

        self.fc = nn.Linear(hidden_size * 2, out_features)
        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Tanh()
        
        self.crf = ChainCRF(out_features, num_labels, bigram=bigram)
        self.readout = None
        self.criterion = None
    
    def _get_rnn_output(self, input_word, mask=None):
        output, _ = self.rnn(input_word)

        output = self.dropout_out(output)
        # [batch, length, out_features]
        output = self.dropout_out(self.activation(self.fc(output)))
        return output
    
    def forward(self, input_word, mask=None):
        # output from rnn [batch, length, hidden_size]
        output = self._get_rnn_output(input_word, mask=mask)
        # [batch, length, num_label, num_label]
        return self.crf(output, mask=mask)


class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, word_dim, rnn_mode, hidden_size, out_features, num_layers,
                 num_labels, p_in=0.33, p_out=0.5, p_rnn=(0.5, 0.5), bigram=False, activation='elu'):
        super(BERT_BiLSTM_CRF, self).__init__()
        
        model_path = "./pytorch"
        
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        config, unused_kwargs = BertConfig.from_pretrained(model_path, output_attention=True,
                                                        foo=False, return_unused_kwargs=True)
        self.bert = BertModel(config).from_pretrained(model_path)
        self.bert.eval()
        
        
        self.bert = BertModel.from_pretrained(model_path)
        self.bilstm_crf = BiRecurrentConvCRF(word_dim, rnn_mode, hidden_size, out_features, num_layers,
                 num_labels, p_in, p_out, p_rnn, bigram=False, activation='elu')
        
    
    def forward(self, input_sentence):      
        
        tokens = self.tokenizer.tokenize(input_sentence)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens])

        enc, _ = self.bert(tokens_tensor)
        output = self.bilstm_crf(enc)
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
