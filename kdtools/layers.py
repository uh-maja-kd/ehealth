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