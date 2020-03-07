import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from itertools import product

from scripts.submit import Algorithm
from scripts.utils import Collection, Keyphrase, Relation
from python_json_config import ConfigBuilder

from kdtools.datasets import SimpleWordIndexDataset, RelationsDependencyParseActionsDataset, RelationsDatasetEmbedding
from kdtools.models import BiLSTMDoubleDenseOracleParser, BiLSTM_CRF
from kdtools.utils.bmewov import BMEWOV

from gensim.models.word2vec import Word2VecKeyedVectors

from numpy.random import random


class UHMajaModel(Algorithm):
    def __init__(self):
        self.models_taskA = {
            "Concept": BiLSTM_CRF(50, 6, 100),
            "Action": BiLSTM_CRF(50, 6, 100),
            "Predicate": BiLSTM_CRF(50, 6, 100),
            "Reference": BiLSTM_CRF(50, 6, 100)
        }
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
            model.eval()
            dataset = SimpleWordIndexDataset(collection)
            for spans, sentence, X in tqdm(dataset.evaluation):
                output = [dataset.labels[idx] for idx in model(X)[1]]
                kps = [[spans[idx] for idx in span_list] for span_list in BMEWOV.decode(output)]
                for kp_spans in kps:
                    idx += 1
                    sentence.keyphrases.append(Keyphrase(sentence, entity_type, idx, kp_spans))


    def run_taskB(self, collection: Collection):
        print("Running taskB")
        model = self.model_taskB
        dataset = RelationsDatasetEmbedding(collection, model.wv)

        model.eval()
        it = 0
        for spans, sentence, state in tqdm(dataset.evaluation):
            it += 1
            words = [sentence.text[start:end] for (start, end) in spans]
            while state[1]:
                o, t, h, d = state

                X = (
                    dataset.encode_word_sequence(["<padding>"]+[words[i - 1] for i in o]).view(1,-1),
                    dataset.encode_word_sequence([words[i - 1] for i in t]).view(1,-1)
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

        # self.models_taskA = {
        #     entity_type: self.train_taskA(
        #         collection,
        #         model_taskA_config,
        #         train_taskA_config.__getattr__(entity_type),
        #         entity_type
        #     )
        #     for entity_type in ["Concept", "Action", "Reference", "Predicate"]
        # }
        self.model_taskB = self.train_taskB(collection, model_taskB_config, train_taskB_config)

    def train_taskA(self, collection, model_config, train_config, entity_type):
        print(f"Training taskA-{entity_type} model.")  #this should be a log
        dataset = SimpleWordIndexDataset(collection, lambda x: x.label == entity_type)
        model = BiLSTM_CRF(
            dataset.word_vector_size,
            model_config.tagset_size,
            model_config.hidden_dim
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
                    _, predicted = model(X)
                    correct += sum(torch.tensor(predicted) == y).item()
                    total += len(predicted)

            print(f"[{epoch + 1}] loss: {running_loss / len(dataset) :0.3}")
            print(f"[{epoch + 1}] accuracy: {correct / total}")

        return model

    def train_taskB(self, collection: Collection, model_config, train_config):
        wv = Word2VecKeyedVectors.load(model_config.embedding_path)
        dataset = RelationsDatasetEmbedding(collection, wv)
        model = BiLSTMDoubleDenseOracleParser(
            dataset.word_vector_size,
            model_config.lstm_hidden_size,
            model_config.dropout_ratio,
            model_config.hidden_dense_size,
            wv,
            len(dataset.actions),
            len(dataset.relations)
        )
        return model
        optimizer = optim.Adam(
            model.parameters()
        )
        criterion_act = CrossEntropyLoss()
        criterion_rel = CrossEntropyLoss()

        for epoch in range(train_config.epochs):
            correct = 0
            correct_act = 0
            correct_rel = 0
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

                total += 1
                is_act = predicted_act == y_act
                is_rel = predicted_rel == y_rel
                correct_act += int(is_act)
                correct_rel += int(is_rel)
                correct += int(is_act and is_rel)

            print(f"[{epoch + 1}] loss_act: {running_loss_act / total}")
            print(f"[{epoch + 1}] loss_rel: {running_loss_rel / total}")
            print(f"[{epoch + 1}] accuracy_act: {correct_act / total}")
            print(f"[{epoch + 1}] accuracy_rel: {correct_rel / total}")
            print(f"[{epoch + 1}] accuracy: {correct / total}")

        return model

if __name__ == "__main__":
    from pathlib import Path

    algorithm = UHMajaModel()

    training = Collection().load(Path("data/training/scenario.txt"))
    algorithm.train(training)
