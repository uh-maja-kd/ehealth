import torch
import torch.nn as nn
import torch.nn.functional as F

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
        sent_encoded, _ = self.bilstmencoder_sent(x[0])
        stack_encoded, _ = self.bilstmencoder_stack(x[1])

        encoded = torch.cat([stack_encoded, sent_encoded], 1)

        action_out = F.softmax(self.action_dense(encoded), 1)
        relation_out = F.softmax(self.relation_dense(encoded), 1)

        return [action_out, relation_out]
