import spacy
from kdtools.utils.latin import CORPUS_CHARS as latin_chars, UNITS as units, CURRENCY as currencies
import re
from kdtools.utils.model_helpers import Tree

class SpacyComponent:
    def __init__(self):
        self.nlp = spacy.load("es_core_news_md")

class TokenizerComponent(SpacyComponent):
    def __init__(self):
        SpacyComponent.__init__(self)

    def get_spans(self, sentence: str):
        return [(token.idx, token.idx+len(token)) for token in self.nlp.tokenizer(sentence)]

class SpacyVectorsComponent(SpacyComponent):

    @property
    def word_vector_size(self):
        return len(self.get_spacy_vector("hola"))

    def get_spacy_vector(self, word: str):
        return self.nlp.vocab.get_vector(word)


class DependencyTreeComponent(SpacyComponent):

    def get_dependency_tree(self, sentence: str):
        tokens = list(self.nlp(sentence))
        nodes = [Tree(token.i, token.text+" "+token.dep_) for token in tokens]

        for node in nodes:
            for child in tokens[node.idx].children:
                node.add_child(nodes[child.i])

        root = list(filter(lambda x: x.dep_ == "ROOT", tokens))[0]
        return nodes[root.i]

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