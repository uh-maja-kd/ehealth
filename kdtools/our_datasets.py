from scripts.utils import Collection

from bmewov import BMEWOV
from loquetienequehacermederos.py import Vocab, SentenceUtils, WordUtils

import es_core_news_md

import torch
from torch.utils.data import DataLoader, Dataset


def _encode_words(sentences_words: list):
    return [
        torch.cat([Vocab.word2index[word] for word in sentence_words])
        for sentence_words in sentences_words
    ]

def _encode_labels(sentences_labels: list):
    return [
        torch.cat([BMEWOV.label2index[label] for label in sentence_labels])
        for sentence_labels in sentences_labels
    ]


class SimpleWordDataset(Dataset):
    def __init__(self, collection: Collection):
        self.sentences = collection.sentences

        self.words = [SentenceUtils.get_words(sentence.text) for sentence in self.sentences]
        self.spans = [SentenceUtils.get_spans(sentence.text) for sentence in self.sentences]
        self.entities_spans = [[k.spans for k in s.keyphrases] for s in self.sentences]


    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        elif isinstance(index, int):
            index = (index,)

        sentences_words = [self.words[idx] for idx in index]
        sentences_spans = [self.spans[idx] for idx in index]
        sentences_entities_spans = [self.entities_spans[idx] for idx in index]
        sentences_labels = [BMEWOV.encode(sentence, entities) \
            for (sentence, entities) in zip(sentences_spans, sentences_entities_spans)]

        return list(
            zip(
                self._encode_words(sentences_words),
                self._encode_labels(),
            )sentences_labels
        )

    def shallow_dataloader(self, **kargs):
        dataloader = DataLoader(self, batch_size=1, **kargs)
        for sentences in dataloader:
            assert len(sentences) == 1
            yield tuple(x.squeeze(0) for x in sentences[0])

