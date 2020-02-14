from scripts.utils import Collection

from kdtools.utils.bmewov import BMEWOV
import es_core_news_md

import torch
from torch.utils.data import DataLoader, Dataset

class SimpleWordIndexDataset(Dataset):
    def __init__(self, collection: Collection):
        self.sentences = collection.sentences

        self.words = [self._get_spans(sentence.text) for sentence in self.sentences]
        self.entities_spans = [[k.spans for k in s.keyphrases] for s in self.sentences]


    def __len__(self):
        return len(self.sentences)

    def _encode_words(self, sentence_words: list):
        return torch.tensor([1 for word in sentence_words])

    def _encode_labels(self, sentence_labels: list):
        return torch.tensor([2 for label in sentence_labels])

    def _get_spans(self, sentence):
        spans = []
        begun = False
        start = None
        punct = ".,;:()-\"\""
        for i, c in enumerate(sentence):
            if not begun and c not in " " + punct:
                begun = True
                start = i
            if begun and c in " " + punct:
                begun = False
                spans.append((start, i))
                if c in punct:
                    spans.append((i, i))

        return spans

    def __getitem__(self, index):
        sentence_words = self.words[index]
        sentence_entities_spans = self.entities_spans[index]
        sentence_labels = BMEWOV.encode(sentence_words, sentence_entities_spans)

        return (
                self._encode_words(sentence_words),
                self._encode_labels(sentence_labels)
        )
