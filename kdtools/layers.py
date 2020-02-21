import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel

class BiLSTMEncoder(nn.Module):
    def __init__(self, *args, **kargs):
        super().__init__()
        self.layer = nn.LSTM(bidirectional=True, *args, **kargs)

        self.hidden_size = 2*self.layer.hidden_size

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


class BetoEncoder(nn.Module):
    def __init__(self, *args, **kargs):
        super().__init__()

    self.tokenizer = BertTokenizer.from_pretrained("pytorch/", do_lower_case=False)
    self.config = config, unused_kwargs = BertConfig.from_pretrained('pytorch/', output_attention=True,
                                                foo=False, return_unused_kwargs=True)

    self.beto = BertModel(config).from_pretrained('pytorch/')
    self.beto.eval()

    def forward(self, sentence):
        tokens = tokenizer.tokenize(sentence)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens])

        output = self.beto(tokens_tensor)

        return output
        

def testBetoLayer():
    model_path = '/pytorch'

    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
    config, unused_kwargs = BertConfig.from_pretrained(model_path, output_attention=True,
                                                    foo=False, return_unused_kwargs=True)
    model = BertModel(config).from_pretrained(model_path)
    model.eval()

    text = "[CLS] Para solucionar los [MASK] de Chile, el presidente debe [MASK] de inmediato. [SEP]"
    masked_indxs = (4,11)

    tokens = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([indexed_tokens])

    output = model(tokens_tensor)
    output[0].shape

if __name__ == "__main__":
    testBetoLayer()