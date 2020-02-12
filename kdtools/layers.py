import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel

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
        

        return output, hidden