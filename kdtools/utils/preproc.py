import spacy
from kdtools.utils.latin import CORPUS_CHARS as latin_chars, UNITS as units, CURRENCY as currencies
import re

class TokenizerComponent:

    def get_spans(self, sentence: str):
        spans, begun, start = [], False, None
        punct = '.,;:()""'
        for i, c in enumerate(sentence):
            if not begun and c not in " " + punct:
                begun = True
                start = i
            if begun and c in " " + punct:
                begun = False
                spans.append((start, i))
            if c in punct:
                spans.append((i, i+1))

        return spans[:-1]

class SpacyVectorsComponent:
    def __init__(self):
        self.nlp = spacy.load("es_core_news_md")

    @property
    def word_vector_size(self):
        return len(self.get_spacy_vector("hola"))

    def get_spacy_vector(self, word: str):
        return self.nlp.vocab.get_vector(word)


class EmbeddingComponent:
    def __init__(self, wv):
        self.wv = wv
        self.vocab = wv.vocab

    @property
    def word_vector_size(self):
        return len(self.wv.vectors[0])

    def map_word(self, word: str):
        tokens = ['<padding>', '<unseen>', '<notlatin>', '<unit>', '<number>']

        if word in tokens:
            return word
        if re.findall(r"[0-9]", word):
            return "<number>"
        if re.fullmatch(units,word):
            return "<unit>"
        if re.fullmatch(currencies, word):
            return "<currency>"
        if len(re.findall(latin_chars, word)) != len(word):
            return "<notlatin>"
        return word

    def get_word_index(self, word):
        word = self.map_word(word)
        return self.vocab[word].index if word in self.vocab else self.vocab["<unseen>"].index
