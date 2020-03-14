import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from itertools import product

from scripts.submit import Algorithm
from scripts.utils import Collection, Keyphrase, Relation
from python_json_config import ConfigBuilder

from kdtools.datasets import (
    SimpleWordIndexDataset,
    RelationsDependencyParseActionsDataset,
    RelationsEmbeddingDataset,
    SentenceEmbeddingDataset
)
from kdtools.models import (
    BiLSTMDoubleDenseOracleParser,
    EmbeddingAttentionBiLSTM_CRF,
    EmbeddingBiLSTM_CRF,
    BERT_TreeLSTM_BiLSTM_CNN_JointModel
)
from kdtools.utils.bmewov import BMEWOV

from gensim.models.word2vec import Word2VecKeyedVectors

from numpy.random import random


class BiLSTMCRF_RelationsParsing(Algorithm):
    def __init__(self):
        self.models_taskA = {}
        self.model_taskB = None

    def run(self, collection: Collection, *args, taskA: bool, taskB: bool, **kargs):
        if taskA:
            self.run_taskA(collection)

        if taskB:
            self.run_taskB(collection)

    def run_taskA(self, collection: Collection):
        print("Running taskA")

        idx = 0
        for entity_type, model in self.models_taskA.items():
            print(entity_type)
            dataset = SentenceEmbeddingDataset(collection, model.wv)

            model.eval()
            for spans, sentence, X in tqdm(dataset.evaluation):
                X = X.view(1,-1)
                output = [dataset.labels[idx] for idx in model(X)[1]]
                kps = [[spans[idx] for idx in span_list] for span_list in BMEWOV.decode(output)]
                for kp_spans in kps:
                    idx += 1
                    sentence.keyphrases.append(Keyphrase(sentence, entity_type, idx, kp_spans))

    def run_taskB(self, collection: Collection):
        print("Running taskB")
        model = self.model_taskB
        dataset = RelationsEmbeddingDataset(collection, model.wv)

        model.eval()
        it = 0
        for spans, sentence, state in tqdm(dataset.evaluation):
            it += 1
            words = [sentence.text[start:end] for (start, end) in spans]
            while state[1]:
                o, t, h, d = state

                X = (
                    *dataset.encode_word_sequence(["<padding>"]+[words[i - 1] for i in o]).view(1,-1),
                    *dataset.encode_word_sequence([words[i - 1] for i in t]).view(1,-1)
                )
                output_act, output_rel = model(X)
                action = dataset.actions[torch.argmax(output_act)]
                relation = dataset.relations[torch.argmax(output_rel)]

                try:
                    if action in ["LEFT", "RIGHT"]:
                        if action == "LEFT":
                            origidx = t[-1]
                            destidx = o[-1]
                        else:
                            origidx = o[-1]
                            destidx = t[-1]

                        origins = [kp.id for kp in sentence.keyphrases if spans[origidx-1] in kp.spans]
                        destinations = [kp.id for kp in sentence.keyphrases if spans[destidx-1] in kp.spans]

                        for origin, destination in product(origins, destinations):
                            sentence.relations.append(Relation(sentence, origin, destination, relation))

                    dataset.actions_funcs[action]["do"](state, relation)
                except Exception as e:
                    # print(e)
                    print(it)
                    state[1].clear()


    def train(self, collection: Collection):
        builder = ConfigBuilder()

        model_taskB_config = builder.parse_config('./configs/config_BiLSTM-Double-Dense-Oracle-Parser.json')
        model_taskA_config = builder.parse_config('./configs/config_BiLSTM-CRF.json')
        train_taskB_config = builder.parse_config('./configs/config_Train_TaskB.json')
        train_taskA_config = builder.parse_config('./configs/config_Train_TaskA.json')

        self.models_taskA = {
            entity_type: self.train_taskA(
                collection,
                model_taskA_config,
                train_taskA_config.__getattr__(entity_type),
                entity_type
            )
            for entity_type in ["Concept", "Action", "Reference", "Predicate"]
        }
        # self.model_taskB = self.train_taskB(collection, model_taskB_config, train_taskB_config)

    def train_taskA(self, collection, model_config, train_config, entity_type):
        print(f"Training taskA-{entity_type} model.")  #this should be a log

        wv = Word2VecKeyedVectors.load(model_config.embedding_path)
        dataset = SentenceEmbeddingDataset(collection, wv, lambda x: x.label == entity_type)

        model = EmbeddingBiLSTM_CRF(
            len(dataset.labels),
            model_config.hidden_dim,
            wv
        )

        optimizer = optim.SGD(
            model.parameters(),
            lr=train_config.optimizer.lr,
            weight_decay=train_config.optimizer.weight_decay
        )

        for epoch in range(train_config.epochs):
            running_loss = 0
            for data in tqdm(dataset):
                X, y = data
                X = X.view(1,-1)
                model.zero_grad()
                loss = model.neg_log_likelihood(X, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            correct = 0
            total = 0
            with torch.no_grad():
                for data in tqdm(dataset):
                    X, y = data
                    X = X.view(1,-1)
                    _, predicted = model(X)
                    correct += sum(torch.tensor(predicted) == y).item()
                    total += len(predicted)

            print(f"[{epoch + 1}] loss: {running_loss / len(dataset) :0.3}")
            print(f"[{epoch + 1}] accuracy: {correct / total}")

        return model

    def train_taskB(self, collection: Collection, model_config, train_config):
        wv = Word2VecKeyedVectors.load(model_config.embedding_path)
        dataset = RelationsEmbeddingDataset(collection, wv)

        model = BiLSTMDoubleDenseOracleParser(
            dataset.word_vector_size,
            model_config.lstm_hidden_size,
            model_config.dropout_ratio,
            model_config.hidden_dense_size,
            wv,
            len(dataset.actions),
            len(dataset.relations)
        )

        optimizer = optim.Adam(
            model.parameters()
        )

        criterion_act = CrossEntropyLoss(weight = dataset.get_actions_weights())
        criterion_rel = CrossEntropyLoss(weight = dataset.get_relations_weights())

        for epoch in range(train_config.epochs):
            correct = 0
            correct_act = 0
            correct_rel = 0
            total_act = 0
            total_rel = 0
            total = 0
            running_loss_act = 0.0
            running_loss_rel = 0.0

            for data in tqdm(dataset):
                *X, y_act, y_rel = data
                X = [x.view(1, -1) for x in X]

                optimizer.zero_grad()

                # forward + backward + optimize
                model.train()
                output_act, output_rel = model(X)

                if y_rel is not None:
                    loss_rel = criterion_rel(output_rel, y_rel)
                    loss_rel.backward(retain_graph=True)
                    running_loss_rel += loss_rel.item()

                loss_act = criterion_act(output_act, y_act)
                loss_act.backward()
                running_loss_act += loss_act.item()

                optimizer.step()

                #collecting data for metrics
                model.eval()
                output_act, output_rel = model(X)
                predicted_act = torch.argmax(output_act, -1)
                predicted_rel = torch.argmax(output_rel, -1)

                total_act += 1
                total_rel += int(y_rel is not None)
                total += 1
                correct_act += int(predicted_act == y_act)
                correct_rel += int(predicted_rel == y_rel) if y_rel is not None else 0
                correct += int(predicted_act == y_act and (predicted_rel == y_rel if y_rel is not None else True))

            print(f"[{epoch + 1}] loss_act: {running_loss_act / total_act}")
            print(f"[{epoch + 1}] loss_rel: {running_loss_rel / total_rel}")
            print(f"[{epoch + 1}] accuracy_act: {correct_act / total_act}")
            print(f"[{epoch + 1}] accuracy_rel: {correct_rel / total_rel}")
            print(f"[{epoch + 1}] accuracy: {correct / total}")

        return model


class JointModel(Algorithm):

    def __init__(self):
        self.model = None

    def run(self, collection: Collection):
        pass

    def train(self, collection: Collection):
        model_config = builder.parse_config('./configs/config_JointModel.json')
        train_config = builder.parse_config('./configs/config_Train_JointModel.json')

        wv = Word2VecKeyedVectors.load(model_config.embedding_path)
        dataset = NoEstaHechoDataset(collection, wv)

        self.model = BERT_TreeLSTM_BiLSTM_CNN_JointModel(
            dataset.embedding_size,
            wv,
            dataset.bert_size,
            dataset.no_postags,
            model_config.postag_size,
            dataset.no_dependencies,
            model_config.dependency_size,
            dataset.no_positions,
            model_config.position_size,
            dataset.no_chars,
            model_config.charencoding_size,
            model_config.tree_lstm_hidden_size,
            model_config.bilstm_hidden_size,
            model_config.local_cnn_channels,
            model_config.local_cnn_window_size,
            model_config.global_cnn_channels,
            model_config.global_cnn_window_size,
            model_config.dropout_chance,
            dataset.no_entity_types,
            dataset.no_entity_tags,
            dataset.no_relations
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config.optimizer.lr,
        )

        ent_type_criterion = CrossEntropyLoss()
        rels_criterion = MSELoss()

        for epoch in train_config.epochs:
            #log variables
            running_loss_ent_type = 0
            running_ent_loss_ent_tag = 0
            running_loss_rels = 0
            correct_ent_type = 0
            correct_ent_tags = 0
            total_tags = 0
            true_positive_rels = 0
            true_negative_rels = 0
            false_positive_rels = 0


            for data in dataset:
                * X, y_ent_type, y_ent_tag, y_rels = data

                optimizer.zero_grad()
                model.train()

                out_ent_type, out_ent_tag, out_rels = model(X)

                loss_ent_type = ent_type_criterion(out_ent_type, y_ent_type)
                running_loss_ent_type += loss_ent_type.item()
                loss_ent_type.backward(retain_graph=True)

                loss_ent_tag = model.entities_crf_decoder.neg_log_likelihood(out_ent_tag, y_ent_tag)
                running_ent_loss_ent_tag += loss_ent_tag.item()
                loss_ent_tag.backward(retain_graph=True)

                loss_rels = rels_criterion(out_rels, y_rels)
                running_loss_rels += loss_rels.item()
                loss_rels.backward()

                optimizer.step()

                model.eval()
                #include diagnostics code

            print(f"[{epoch + 1}] ent_type_loss: {loss_ent_type / len(dataset) :0.3}")
            print(f"[{epoch + 1}] ent_tag_loss: {loss_ent_tag / len(dataset) :0.3}")
            print(f"[{epoch + 1}] rels_loss: {running_loss_rels / len(dataset) :0.3}")
            # print(f"[{epoch + 1}] ent_type_acc: {correct_ent_type / len(dataset) :0.3}")
            # print(f"[{epoch + 1}] ent_tag_acc: {correct_ent_tags / total_tags :0.3}")
            # print(f"[{epoch + 1}] rels_precision: {true_positive_rels / (true_positive_rels+true_negative_rels) :0.3}")
            # print(f"[{epoch + 1}] rels_recovery: {true_positive_rels / (true_positive_rels+false_positive_rels) :0.3}")





if __name__ == "__main__":
    from pathlib import Path

    algorithm = UHMajaModel()

    training = Collection().load(Path("data/training/scenario.txt"))
    algorithm.train(training)
