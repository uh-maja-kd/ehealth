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

        self.labels = ["B", "M", "E", "W", "O", "V"]
        self.label2index = {label: idx for (idx, label) in enumerate(self.labels)}

    def __len__(self):
        return len(self.sentences)

    def _encode_word_sequence(self, words):
        return torch.rand(10, len(words)+1)

    def _encode_label_sequence(self, labels: list):
        return torch.tensor([self.label2index[label] for label in labels])

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
                self._encode_word_sequence(sentence_words),
                self._encode_label_sequence(sentence_labels)
        )
