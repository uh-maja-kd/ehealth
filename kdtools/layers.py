import torch
import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    def __init__(self, *args, **kargs):
        super().__init__()
        self.layer = nn.LSTM(bidirectional=True, *args, **kargs)

    def forward(self, input, hx=None):
        output, hidden = self.layer(input, hx)

        hidden_size = self.layer.hidden_size

        if self.layer.batch_first:
            left2right = output[:, -1, :hidden_size]
            right2left = output[:, 0, hidden_size:]
        else:
            left2right = output[-1, :, :hidden_size]
            right2left = output[0, :, hidden_size:]

        output = torch.cat((left2right, right2left), 1)

        return output, hidden
