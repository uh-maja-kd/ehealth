import spacy
from kdtools.utils.latin import CORPUS_CHARS as latin_chars, UNITS as units, CURRENCY as currencies
import re
from kdtools.utils.model_helpers import Tree
from string import ascii_lowercase
from functools import lru_cache
from torch.nn.functional import one_hot
import numpy as np

class SpacyComponent:
    nlp = None
    def __init__(self):
        self.nlp = SpacyComponent.nlp if SpacyComponent.nlp else spacy.load("es_core_news_md")
        SpacyComponent.nlp = self.nlp

    @lru_cache()
    def nlp_wrapper(self, sentence):
        return self.nlp(sentence)

class TokenizerComponent(SpacyComponent):
    def __init__(self):
        SpacyComponent.__init__(self)

    def get_spans(self, sentence: str):
        return [(token.idx, token.idx+len(token)) for token in self.nlp_wrapper(sentence)]

class SpacyVectorsComponent(SpacyComponent):

    @property
    def word_vector_size(self):
        return len(self.get_spacy_vector("hola"))

    def get_spacy_vector(self, word: str):
        return self.nlp.vocab.get_vector(word)


class DependencyTreeComponent(SpacyComponent):

    def get_dependency_tree(self, sentence: str):
        tokens = list(self.nlp_wrapper(sentence))
        nodes = [Tree(token.i, token.text+" "+token.dep_) for token in tokens]

        for node in nodes:
            for child in tokens[node.idx].children:
                node.add_child(nodes[child.i])

        root = list(filter(lambda x: x.dep_ == "ROOT", tokens))[0]
        return nodes[root.i]


class CharEmbeddingComponent:
    def __init__(self, sentences):
        self.abc = ['<pad>','<unk>']  + list(set(list(''.join(sentences))))
        self.int2char = dict(enumerate(self.abc))
        self.char2int = {char: index for index,char in self.int2char.items()}
        #print(self.abc)

    def encode(self, word, max_word_len, sent_len):
        #print(word)
        #print(max_word_len)
        solve = [self.char2int[char] for char in word]
        len_solve = len(solve)
        for i in range(max_word_len - len_solve):
            solve += [self.char2int['<pad>']]

        return solve
        
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

class PostagComponent(SpacyComponent):
    def __init__(self):
        super(SpacyComponent).__init__()

        self.postags = [
            "ADJ",
            "ADP",
            "ADV",
            "AUX",
            "CONJ",
            "CCONJ",
            "DET",
            "INTJ",
            "NOUN",
            "NUM",
            "PART",
            "PRON",
            "PROPN",
            "PUNCT",
            "SCONJ",
            "SYM",
            "VERB",
            "X",
            "SPACE"
        ]
        self.postag2index = {postag: idx for (idx, postag) in enumerate(self.postags)}

    def get_sentence_postags(self, sentence: str):
        tokens = list(self.nlp_wrapper(sentence))

        return [self.postag2index[token.pos_] for token in tokens]


class RelationComponent(SpacyComponent):
    def __init__(self):
        super().__init__()
    
        self.relations = [
            "subject",
            "target",
            "in-place",
            "in-time",
            "in-context",
            "arg",
            "domain",
            "has-property",
            "part-of",
            "is-a",
            "same-as",
            "causes",
            "entails"
        ]
        self.relation2index = {relation: self.relations.index(relation) for relation in self.relations}
    
    def get_sentence_relations(self, sentence):
        pass 

class DependencyComponent(SpacyComponent):
    def __init__(self):
        super(SpacyComponent).__init__()

        self.dependencies = [
            "acl",
            "advcl",
            "advmod",
            "amod",
            "appos",
            "aux",
            "case",
            "cc",
            "ccomp",
            "clf",
            "compound",
            "conj",
            "cop",
            "csubj",
            "dep",
            "det",
            "discourse",
            "dislocated",
            "expl",
            "fixed",
            "flat",
            "goeswith",
            "iobj",
            "list",
            "mark",
            "nmod",
            "nsubj",
            "nummod",
            "obj",
            "obl",
            "orphan",
            "parataxis",
            "punct",
            "reparandum",
            "ROOT",
            "vocative",
            "xcomp",
            "other",
            'expl:pass'
        ]
        self.dep2index = {dep: idx for (idx, dep) in enumerate(self.dependencies)}

    def get_sentence_dependencies(self, sentence: str):
        tokens = list(self.nlp_wrapper(sentence))
        return [self.dep2index[token.dep_ if token.dep_ != '' else 'other'] for token in tokens]

class PositionComponent:
    def __init__(self, max_sent_len):
        self.max_sent_len = max_sent_len
        self.no_positions = 2*max_sent_len+1

    def get_position_encoding(self, no_words, index):
        return [i - index + self.max_sent_len for i in range(no_words)]

class EntityComponent:
    def __init__(self):

        self.tags = [
            "Concept",
            "Action", 
            "Reference", 
            "Predicate",
            "<None>"
        ]

        self.tag2index = {tag: idx for (idx, tag) in enumerate(self.tags)}
    
    def get_tag_encoding(self, sequence):
        return [self.tag2index[tag] for tag in sequence]


