import torch
import torch.nn as nn
import torch.nn.functional as F

import kdtools as kd


class BasicSequenceTagger(nn.Module):
    def __init__(
        self,
        *,
        char_vocab_size,
        char_embedding_dim,
        padding_idx,
        char_repr_dim,
        word_repr_dim,
        postag_repr_dim,
        token_repr_dim,
        num_labels,
    ):
        super().__init__()

        # List[Char] -> List[Vector]
        self.char2embed = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=char_embedding_dim,
            padding_idx=padding_idx,
        )

        # List[Vector] -> List[Vector]
        self.char_bilstm = kd.BiLSTMEncoder(
            input_size=char_embedding_dim,
            hidden_size=char_repr_dim // 2,
            num_layers=1,
            batch_first=True,
        )

        # List[Token] -> List[Vector]
        self.token_bilstm = nn.LSTM(
            input_size=char_repr_dim + word_repr_dim + postag_repr_dim,
            hidden_size=token_repr_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Vector -> Category
        self.token2tag = nn.Linear(token_repr_dim, num_labels)

    def forward(self, sentence):
        # input
        chars, words, postags = sentence

        # character input
        char_embeds = self.char2embed(chars)
        char_reprs, _ = self.char_bilstm(char_embeds)

        # token representation
        tokens = torch.cat((char_reprs, words, postags), 1).unsqueeze(0)
        token_reprs, _ = self.token_bilstm(tokens)

        # output
        out = self.token2tag(token_reprs)
        return out


def test():
    torch.manual_seed(0)

    chars = torch.randint(5, (3, 4))
    print(chars)

    words = torch.rand((3, 15))
    print(words)

    postags = torch.zeros(3, 10)
    postags[torch.arange(3), torch.randint(10, (3,))] = 1
    print(postags)

    sentence = [chars, words, postags]

    model = BasicSequenceTagger(
        char_vocab_size=5,
        char_embedding_dim=15,
        padding_idx=0,
        char_repr_dim=20,
        word_repr_dim=15,
        postag_repr_dim=postags.size(1),
        token_repr_dim=30,
        num_labels=3,
    )
    print(model)

    outputs = model(sentence)
    print(outputs.shape)

    labels = torch.tensor(1).unsqueeze(0)
    print(labels.shape)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    print(loss)


def test2():
    from kdtools.datasets import LabeledSentencesDS
    from torch.utils.data import DataLoader

    dataset = LabeledSentencesDS(
        ["El gato azul salta muy alto.", "Las oraciones contienen datos"],
        lambda x: ([["O" for z in y] for y in x], "BILUOV"),
    )
    print(dataset.sentences)

    model = BasicSequenceTagger(
        char_vocab_size=len(dataset.char_vocab),
        char_embedding_dim=15,
        padding_idx=dataset.PADDING,
        char_repr_dim=20,
        word_repr_dim=dataset.vectors_len,
        postag_repr_dim=len(dataset.pos2index),
        token_repr_dim=30,
        num_labels=len("BILUOV"),
    )
    print(model)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for sentences in dataloader:
        for data in sentences:
            data = tuple(x.squeeze(0) for x in data)
            *sentence, label = data
            output = model(sentence)
            print(output)

    for *sentence, label in dataset.shallow_dataloader(shuffle=True):
        output = model(sentence)
        print(output)


if __name__ == "__main__":
    test2()
