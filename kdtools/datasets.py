from numpy import argmax
from scripts.utils import Collection, Sentence, Relation
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot
from tqdm import tqdm
from operator import add
from functools import reduce
from kdtools.utils.bmewov import BMEWOV
from kdtools.utils.preproc import *
from kdtools.utils.model_helpers import Tree
from itertools import product

class Node:
    def __init__(self):
        self.parent = 0
        self.children = []
        self.dep = "NONE"

class RelationsDependencyParseActionsDataset(Dataset, TokenizerComponent, SpacyVectorsComponent):

    def __init__(self, collection: Collection):
        TokenizerComponent.__init__(self)
        SpacyVectorsComponent.__init__(self)

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
        spans = self.get_spans(sentence.text)
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
            except Exception as e:
                pass
                # print(sentence)
                # print([kp.spans for kp in sentence.keyphrases])
                # print(e)
        return data

    def _get_sentence_data(self, sentence: Sentence):
        samples = []
        ok, actions = self._get_actions(sentence)
        if ok:
            sent_size = len(self.get_spans(sentence.text))
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
        return (torch.tensor([self.get_spacy_vector(word) for word in words]),)

    def __len__(self):
        return len(self.flatdata)

    def __getitem__(self, index: int):
        inp, out = self.flatdata[index]
        state, sentence = inp
        action, rel = out
        words = [sentence.text[start:end] for (start, end) in self.get_spans(sentence.text)]
        o,t,h,d = state

        return (
                *self.encode_word_sequence(["<padding>"] + [words[i - 1] for i in o]),
                *self.encode_word_sequence([words[i - 1] for i in t]),
                torch.LongTensor([self.action2index[action]]),
                torch.LongTensor([self.relation2index[rel]]) if rel != "none" else None
        )

    @property
    def evaluation(self):
        ret = []
        for sentence in self.sentences:
            spans = self.get_spans(sentence.text)
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
        count = torch.tensor(list(count.values()), dtype=torch.float)
        print(count)
        return count.min()/count

    def get_relations_weights(self):
        count = {relation:0 for relation in self.relations}
        for data in self:
            *X, y_act, y_rel = data
            if y_rel is not None:
                count[self.relations[y_rel.item()]] += 1
        count = torch.tensor(list(count.values()), dtype=torch.float)
        print(count)
        return count.min()/count

class RelationsEmbeddingDataset(RelationsDependencyParseActionsDataset, EmbeddingComponent):
    def __init__(self, collection: Collection, wv):
        RelationsDependencyParseActionsDataset.__init__(self, collection)
        EmbeddingComponent.__init__(self, wv)

    def encode_word_sequence(self, words):
        return (torch.tensor([self.get_word_index(word) for word in words], dtype=torch.long),)

class SimpleWordIndexDataset(Dataset, TokenizerComponent):

    def __init__(self, collection: Collection, entity_criteria = lambda x: x):
        TokenizerComponent.__init__(self)

        self.sentences = collection.sentences
        self.words_spans = [self.get_spans(sentence.text) for sentence in self.sentences]
        self.words = [[sentence.text[start:end] for (start,end) in spans] for (sentence, spans) in zip(self.sentences, self.words_spans)]
        self.entities_spans = [[k.spans for k in filter(entity_criteria, s.keyphrases)] for s in self.sentences]

        self.labels = ["B", "M", "E", "W", "O", "V"]
        self.label2index = {label: idx for (idx, label) in enumerate(self.labels)}

    @property
    def word_vector_size(self):
        return len(get_spacy_vector("hola"))

    @property
    def evaluation(self):
        return [(self.words_spans[idx], self.sentences[idx], self[idx][0]) for idx in range(len(self))]

    def __len__(self):
        return len(self.sentences)

    def _encode_word_sequence(self, words):
        return (torch.tensor([get_spacy_vector(word) for word in words]),)

    def _encode_label_sequence(self, labels: list):
        return torch.tensor([self.label2index[label] for label in labels], dtype = torch.long)

    def __getitem__(self, index):
        sentence_words_spans = self.words_spans[index]
        sentence_words = self.words[index]
        sentence_entities_spans = self.entities_spans[index]
        sentence_labels = BMEWOV.encode(sentence_words_spans, sentence_entities_spans)

        return (
                *self._encode_word_sequence(sentence_words),
                self._encode_label_sequence(sentence_labels)
        )

class SentenceEmbeddingDataset(SimpleWordIndexDataset, EmbeddingComponent):
    def __init__(self, collection: Collection, wv, entity_criteria = lambda x: x):
        SimpleWordIndexDataset.__init__(self, collection, entity_criteria)
        EmbeddingComponent.__init__(self, wv)

    def _encode_word_sequence(self, words):
        return (torch.tensor([self.get_word_index(word) for word in words], dtype=torch.long),)

class EntitiesPairsDataset(Dataset, TokenizerComponent, EmbeddingComponent):

    def __init__(self, collection: Collection, wv):
        TokenizerComponent.__init__(self)
        EmbeddingComponent.__init__(self, wv)

        self.collection = collection
        self.dataxsentence = self._get_data(collection)
        self.relations_data = reduce(add, self.dataxsentence)

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

    def _get_data(self, collection: Collection):
        return [self._get_sentence_data(sentence) for sentence in tqdm(collection.sentences)]

    def _get_sentence_data(self, sentence: Sentence):
        data = []
        for relation in sentence.relations:
            try:
                data.append(self._get_relation_data(relation))
            except Exception as e:
                pass
                # print(e)
                # print(sentence)
        return data

    def _get_relation_data(self, relation: Relation):
        sentence = relation.sentence

        sentence_spans = self.get_spans(sentence.text)
        sentence_words = [sentence.text[start:end] for (start, end) in sentence_spans]

        origin_spans = sentence.find_keyphrase(relation.origin).spans
        origin_idxs = [sentence_spans.index(span) for span in origin_spans]

        destination_spans = sentence.find_keyphrase(relation.destination).spans
        destination_idxs = [sentence_spans.index(span) for span in destination_spans]

        return (sentence_words, origin_idxs, destination_idxs, relation.label)

    def _encode_word_sequence(self, words):
        return (torch.tensor([self.get_word_index(word) for word in words], dtype=torch.long),)

    def __len__(self):
        return len(self.relations_data)

    def __getitem__(self, idx):
        sentence, origin, destination, label = self.relations_data[idx]

        mask_origin = torch.sum(one_hot(torch.tensor(origin), len(sentence)), dim = 0)
        mask_destination = torch.sum(one_hot(torch.tensor(destination), len(sentence)), dim = 0)

        return (
            *self._encode_word_sequence(sentence),
            mask_origin,
            mask_destination,
            torch.LongTensor([self.relation2index[label]])
        )

class JointModelDataset(
    Dataset,
    TokenizerComponent,
    EmbeddingComponent,
    CharEmbeddingComponent,
    PostagComponent,
    PositionComponent,
    DependencyComponent,
    DependencyTreeComponent,
    EntityTagsComponent,
    RelationComponent,
    BMEWOVLabelsComponent,
    ShufflerComponent):

    def __init__(self, collection: Collection, wv):
        print("Loading nlp...")
        TokenizerComponent.__init__(self)
        EmbeddingComponent.__init__(self, wv)
        CharEmbeddingComponent.__init__(self, [sentence.text for sentence in collection.sentences])
        PostagComponent.__init__(self)
        PositionComponent.__init__(self, max([len(self.get_spans(sentence.text)) for sentence in collection.sentences]))
        DependencyComponent.__init__(self)
        DependencyTreeComponent.__init__(self)
        EntityTagsComponent.__init__(self)
        RelationComponent.__init__(self)
        BMEWOVLabelsComponent.__init__(self)
        ShufflerComponent.__init__(self)

        self.dataxsentence = self._get_sentences_data(collection)
        self.data = self.get_data()

    def _get_sentences_data(self, collection):
        data = []
        print("Collecting data per sentence...")
        for sentence in tqdm(collection.sentences):
            spans = self.get_spans(sentence.text)
            words = [sentence.text[beg:end] for (beg, end) in spans]

            word_embedding_data = self._get_word_embedding_data(words)
            char_embedding_data = self._get_char_embedding_data(words)
            postag_data = self._get_postag_data(sentence.text)
            dependency_data = self._get_dependency_data(sentence.text)
            dep_tree, dependencytree_data = self._get_dependencytree_data(sentence.text)
            head_words = self._get_head_words(sentence, spans, dep_tree)

            data.append((
                sentence,
                spans,
                head_words,
                word_embedding_data,
                char_embedding_data,
                postag_data,
                dependency_data,
                dependencytree_data
            ))

        return data

    def _get_head_words(self, sentence, spans, dep_tree):
        head_words = [[] for _ in range(len(spans))]
        for kp in sentence.keyphrases:
            try:
                kp_indices = [spans.index(span) for span in kp.spans]
                head_word_index = self._get_entity_head(kp_indices, dep_tree)
                head_words[head_word_index].append(kp)
            except Exception as e:
                # print(e)
                # print(sentence)
                # print(kp)
                # print(spans)
                pass
        return head_words


    def _get_word_embedding_data(self, words):
        return torch.tensor([self.get_word_index(word) for word in words], dtype=torch.long)

    def _get_char_embedding_data(self, words):
        max_word_len = max([len(word) for word in words])
        chars_indices = [self.encode_chars_indices(word, max_word_len, len(words)) for word in words]
        return one_hot(torch.tensor(chars_indices, dtype=torch.long), len(self.abc)).type(dtype = torch.float32)

    def _get_postag_data(self, sentence):
        return torch.tensor(self.get_sentence_postags(sentence), dtype=torch.long)

    def _get_dependency_data(self ,sentence):
        return torch.tensor(self.get_sentence_dependencies(sentence), dtype=torch.long)

    def _find_node(self, tree, idx):
        if tree.idx == idx:
            return tree

        else:
            for child in tree.children:
                node = self._find_node(child, idx)
                if node:
                    return node
            return False

    def _get_dependencytree_data(self, sentence):
        dep_tree = self.get_dependency_tree(sentence)
        sent_len = len(self.get_spans(sentence))
        return dep_tree, [self._find_node(dep_tree, i) for i in range(sent_len)]

    def _get_entity_head(self, entity_words : list , dependency_tree : Tree):

        if dependency_tree.idx in entity_words:
            return dependency_tree.idx

        for child in dependency_tree.children:
            ans = self._get_entity_head(entity_words, child)
            if ans is not None:
                return ans

        return None

    def _get_false_data(self, sent_len):
        token_label = torch.tensor(self.get_tag_encoding(['<None>']), dtype=torch.long)
        sentence_label = torch.tensor([self.label2index['O'] for _ in range(sent_len)], dtype=torch.long)
        relation_matrix = torch.tensor([[0 for _ in range(sent_len)] for _ in range(len(self.relations))], dtype=torch.float32)

        return (token_label, sentence_label, relation_matrix)

    def get_data(self):
        data = []
        print("Creating dataset...")
        for sent_data in tqdm(self.dataxsentence):
            (
                sentence,
                spans,
                head_words,
                word_embedding_data,
                char_embedding_data,
                postag_data,
                dependency_data,
                dependencytree_data
            ) = sent_data

            sent_len = len(spans)
            total_relations = len(self.relations)

            for idx in range(sent_len):
                if len(head_words[idx]) > 0:
                    token_label = torch.tensor(self.get_tag_encoding([head_words[idx][0].label]), dtype=torch.long)

                    entities_spans = [kp.spans for kp in head_words[idx]]
                    sentence_labels = BMEWOV.encode(spans, entities_spans)
                    sentence_labels = torch.tensor([self.label2index[label] for label in sentence_labels], dtype = torch.long)

                    words = [sentence.text[start:end] for (start,end) in spans]

                    relation_matrix = [[0 for _ in range(sent_len)] for _ in range(len(self.relations))]

                    for dest_idx in range(sent_len):
                        for orig, dest in product(head_words[idx], head_words[dest_idx]):
                            relations = sentence.find_relations(orig.id, dest.id)
                            for relation in relations:
                                relation_matrix[self.relation2index[relation.label]][dest_idx] = 1

                    relation_matrix = torch.tensor(relation_matrix, dtype=torch.long)

                    data.append((
                        word_embedding_data,
                        char_embedding_data,
                        postag_data,
                        dependency_data,
                        dependencytree_data,
                        token_label,
                        sentence_labels,
                        relation_matrix))

                else:
                    token_label, sentence_labels, relation_matrix = self._get_false_data(sent_len)

                    data.append((
                        word_embedding_data,
                        char_embedding_data,
                        postag_data,
                        dependency_data,
                        dependencytree_data,
                        token_label,
                        sentence_labels,
                        relation_matrix)
                    )

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
