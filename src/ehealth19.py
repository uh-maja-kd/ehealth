import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from scripts.submit import Algorithm
from scripts.utils import Collection
from python_json_config import ConfigBuilder



from kdtools.datasets import RelationsDependencyParseActionsDataset, SimpleWordIndexDataset
from kdtools.models import BiLSTMDoubleDenseOracleParser, BiLSTM_CRF


class UHMajaModel(Algorithm):
    def __init__(self):
        self.model_taskA = None
        self.model_taskB = None

    def run(self, collection: Collection, *args, taskA: bool, taskB: bool, **kargs):
        return super().run(collection, *args, taskA=taskA, taskB=taskB, **kargs)

    def train(self, collection: Collection):
        builder = ConfigBuilder()
        
        model_taskB_config = builder.parse_config('./configs/config_BiLSTM-Double-Dense-Oracle-Parser.json')
        model_taskA_config = builder.parse_config('./configs/config_BiLSTM-CRF.json')
        train_taskB_config = builder.parse_config('./configs/config_Train_TaskB.json')
        train_taskA_config = builder.parse_config('./configs/config_Train_TaskA.json')

        self.model_taskA = self.train_taskA(collection, model_taskA_config, train_taskA_config)
        #self.model_taskB = self.train_taskB(collection, model_taskB_config, train_taskB_config)

    def train_taskA(self, collection, model_config, train_config):
        dataset = SimpleWordIndexDataset(collection, lambda x : x.label == 'Concept')
        model = BiLSTM_CRF()

        optimizer = optim.SGD(model.parameters(), lr= train_config.optimizer.lr, momentum=train_config.optimizer.momentum)
        criterion = CrossEntropyLoss()

        for epoch in range(train_config.epochs):
            correct = 0
            total = 0
            running_loss = 0.0

            for data in tqdm(dataset):
                X, y = data
                X, y = X.view(1, -1 , dataset.word_vector_size), y.view(1, -1)
                optimizer.zero_grad()

                # forward + backward + optimizes
                output = model(X)

                loss = criterion(output.view(1,model.tagset_size, -1), y)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                predicted = torch.argmax(output, -1)
                
                total += len(predicted)
                correct += sum(predicted.view(-1)==y.view(-1)).item()

            print(f"[{epoch + 1}] loss_act: {running_loss / len(dataset) :0.3}")
            print(f"[{epoch + 1}] accuracy: {correct / total}")

        return model

    def train_taskB(self, collection: Collection, model_config, train_config):
        dataset = RelationsDependencyParseActionsDataset(collection)
        model = BiLSTMDoubleDenseOracleParser(5, 14, 10, 50, batch_first=model_config.batch_first)

        optimizer = optim.SGD(model.parameters(), lr= train_config.optimizer.lr, momentum=train_config.optimizer.momentum)
        criterion_act = CrossEntropyLoss()
        criterion_rel = CrossEntropyLoss()

        for epoch in range(train_config.epochs):
            correct = 0
            total = 0
            running_loss_act = 0.0
            running_loss_rel = 0.0

            for data in tqdm(dataset):
                *X, y_act, y_rel = data
                X = [x.view(1,-1,10) for x in X]
                optimizer.zero_grad()

                # forward + backward + optimize
                output_act, output_rel = model(X)

                loss_act = criterion_act(output_act, y_act)
                loss_act.backward(retain_graph=train_config.loss.retain_graph)

                loss_rel = criterion_rel(output_rel, y_rel)
                loss_rel.backward()

                optimizer.step()

                running_loss_act += loss_act.item()
                running_loss_rel += loss_rel.item()

                predicted_act = torch.argmax(output_act, -1)
                predicted_rel = torch.argmax(output_rel, -1)
                total += 1
                correct += int(predicted_act == y_act and predicted_rel == y_rel)

            print(f"[{epoch + 1}] loss_act: {running_loss_act / len(dataset) :0.3}")
            print(f"[{epoch + 1}] loss_rel: {running_loss_rel / len(dataset) :0.3}")
            print(f"[{epoch + 1}] accuracy: {correct / total}")

        return model


if __name__ == "__main__":
    from pathlib import Path

    algorithm = UHMajaModel()

    training = Collection().load(Path("D:/TESIS/ehealth/data/training/scenario.txt"))
    algorithm.train(training)
