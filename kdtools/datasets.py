from numpy import argmax
from scripts.utils import Collection, Sentence
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from operator import add
from functools import reduce
from kdtools.utils.bmewov import BMEWOV
from kdtools.utils.preproc import get_spans, get_spacy_vector

class Node:
    def __init__(self):
        self.parent = 0
        self.children = []
        self.dep = "NONE"

class RelationsDependencyParseActionsDataset(Dataset):

    def __init__(self, collection: Collection):
        self.actions = ["IGNORE", "LEFT", "RIGHT", "REDUCE", "SHIFT"]
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
        self.actions_funcs = {
            "IGNORE": {"can": self._can_do_ignore, "do":self._ignore},
            "SHIFT": {"can": self._can_do_shift, "do":self._shift},
            "REDUCE": {"can": self._can_do_reduce, "do":self._reduce},
            "LEFT": {"can": self._can_do_leftarc, "do":self._leftarc},
            "RIGHT": {"can": self._can_do_rightarc, "do":self._rightarc}
        }
        self.action2index = { action: self.actions.index(action) for action in self.actions}
        self.relation2index = {relation: self.relations.index(relation) for relation in self.relations}

        #log util info
        self.count = [0,0]

        self.dataxsentence = self._get_data(collection)
        self.flatdata = reduce(add, self.dataxsentence)

        self.word_vector_size = len(get_spacy_vector("hola"))
        self.sentences = collection.sentences

    def _can_do_ignore(self, state, tree):
        _,t,_,_ = state
        j = t[-1]
        return "none" if tree[j].parent == 0 and len(tree[j].children) == 0 else None

    def _can_do_shift(self, state, tree):
        _,t,_,_ = state
        return "none" if len(t)>0 else None

    def _can_do_reduce(self, state, tree):
        o,_,h,_ = state
        if not o:
            return None
        i = o[-1]

        current_children = [x for x in range(len(tree)) if h[x] == i]
        tree_children = tree[i].children
        return "none" if h[i] == tree[i].parent and current_children == tree_children else None

    def _can_do_leftarc(self, state, tree):
        o,t,h,_ = state
        if not o:
            return None
        i = o[-1]
        j = t[-1]
        return tree[i].dep if h[i] == 0 and tree[i].parent == j else None

    def _can_do_rightarc(self, state, tree):
        o,t,h,_ = state
        if not o:
            return None
        i = o[-1]
        j = t[-1]
        return tree[j].dep if h[j] == 0 and tree[j].parent == i else None

    def _ignore(self, state, rel):
        o,t,h,d = state
        t.pop()

    def _reduce(self, state, rel):
        o,t,h,d = state
        o.pop()

    def _shift(self, state, rel):
        o,t,h,d = state
        o.append(t.pop())

    def _leftarc(self, state, rel):
        o,t,h,d = state
        i = o.pop()
        j = t[-1]
        h[i] = j
        d[i] = rel

    def _rightarc(self, state, rel):
        o,t,h,d = state
        i = o[-1]
        j = t.pop()
        h[j] = i
        d[j] = rel
        o.append(j)

    def _build_tree(self, sentence: Sentence):
        spans = get_spans(sentence.text)
        id2pos = {kp.id: spans.index(kp.spans[0]) + 1 for kp in sentence.keyphrases}
        sent_len = len(spans)

        headed_indexes = []
        rels = []

        for rel in sentence.relations:
            orig = rel.origin
            dest = rel.destination
            rels.append((id2pos[orig],id2pos[dest],rel.label))
            headed_indexes.append(id2pos[dest])

        not_kp_pos = [x for x in range(1,sent_len+1) if x not in headed_indexes]

        for i in not_kp_pos:
            rels.append((0,i,"NONE"))

        rels.sort()

        nodes = [Node() for _ in range(sent_len+1)]
        for i,j,r in rels:
            self.count[0]+=1
            if nodes[j].parent == 0:
                nodes[j].parent = i
                nodes[i].children.append(j)
                nodes[j].dep = r
            else:
                self.count[1]+=1

        return nodes

    def _get_actions(self, sentence: Sentence):
        tree = self._build_tree(sentence)
        sent_size = len(tree)-1
        state = ([], [i for i in range(sent_size, 0, -1)], [0] * (sent_size + 1), [0] * (sent_size + 1))

        actions = []

        while state[1]:
            deps = [self.actions_funcs[action]["can"](state, tree) for action in self.actions]
            cans = [dep is not None for dep in deps]

            if any(cans):
                idx = argmax(cans)
                action = self.actions[idx]
                dep = deps[idx]
                action_do = self.actions_funcs[action]["do"]

                action_do(state, dep)

                actions.append((action, dep))
            else:
                return False, actions
        return True, actions


    def _get_data(self, collection: Collection):
        data = []
        for sentence in tqdm(collection.sentences):
            try:
                ok, sent_data = self._get_sentence_data(sentence)
                if ok:
                    data.append(sent_data)
            except:
                pass
                # print(sentence)
        return data

    def _get_sentence_data(self, sentence: Sentence):
        samples = []
        ok, actions = self._get_actions(sentence)
        if ok:
            sent_size = len(get_spans(sentence.text))
            if actions:
                state = ([],[i for i in range(sent_size,0,-1)],[0]*(sent_size+1), ["NONE"]*(sent_size+1))
                for name, dep in actions:
                    samples.append(
                        (
                            (self._copy_state(state), sentence),
                            (name, dep)
                        )
                    )
                    action_do = self.actions_funcs[name]["do"]
                    action_do(state, dep)

            return True, samples
        return False, []

    def _copy_state(self, state):
        o,t,h,d = state
        return (o.copy(), t.copy(), h.copy(), d.copy())

    def encode_word_sequence(self, words):
        return torch.tensor([get_spacy_vector(word) for word in words])

    def __len__(self):
        return len(self.flatdata)

    def __getitem__(self, index: int):
        inp, out = self.flatdata[index]
        state, sentence = inp
        action, rel = out
        words = [sentence.text[start:end] for (start, end) in get_spans(sentence.text)]
        o,t,h,d = state

        return (
                self.encode_word_sequence(["."] + [words[i - 1] for i in o]),
                self.encode_word_sequence([words[i - 1] for i in t]),
                torch.LongTensor([self.action2index[action]]),
                torch.LongTensor([self.relation2index[rel]]) if rel != "none" else None
        )

    @property
    def evaluation(self):
        ret = []
        for sentence in self.sentences:
            spans = get_spans(sentence.text)
            sent_size = len(spans)
            ret.append(
                (
                    spans,
                    sentence,
                    ([], [i for i in range(sent_size, 0, -1)], [0]*(sent_size + 1), ["NONE"]*(sent_size + 1))
                )
            )
        return ret

    def get_actions_weights(self):
        count = {action:0 for action in self.actions}
        for data in self:
            *X, y_act, y_rel = data
            count[self.actions[y_act.item()]] += 1
        count = torch.tensor(list(count.values()), dtype = torch.float)
        return count.max()/count

    def get_relations_weights(self):
        count = {relation:0 for relation in self.relations}
        for data in self:
            *X, y_act, y_rel = data
            if y_rel is not None:
                count[self.relations[y_rel.item()]] += 1
        count = torch.tensor(list(count.values()), dtype = torch.float)
        return count.max()/count

class SimpleWordIndexDataset(Dataset):
    def __init__(self, collection: Collection, entity_criteria = lambda x: x):
        self.sentences = collection.sentences
        self.words_spans = [get_spans(sentence.text) for sentence in self.sentences]
        self.words = [[sentence.text[start:end] for (start,end) in spans] for (sentence, spans) in zip(self.sentences, self.words_spans)]
        self.entities_spans = [[k.spans for k in filter(entity_criteria, s.keyphrases)] for s in self.sentences]

        self.labels = ["B", "M", "E", "W", "O", "V"]
        self.label2index = {label: idx for (idx, label) in enumerate(self.labels)}

        self.word_vector_size = len(get_spacy_vector("hola"))

    @property
    def evaluation(self):
        return [(self.words_spans[idx], self.sentences[idx], self[idx][0]) for idx in range(len(self))]

    def __len__(self):
        return len(self.sentences)

    def _encode_word_sequence(self, words):
        return torch.tensor([get_spacy_vector(word) for word in words])

    def _encode_label_sequence(self, labels: list):
        return torch.tensor([self.label2index[label] for label in labels], dtype = torch.long)

    def __getitem__(self, index):
        sentence_words_spans = self.words_spans[index]
        sentence_words = self.words[index]
        sentence_entities_spans = self.entities_spans[index]
        sentence_labels = BMEWOV.encode(sentence_words_spans, sentence_entities_spans)

        return (
                self._encode_word_sequence(sentence_words),
                self._encode_label_sequence(sentence_labels)
        )
