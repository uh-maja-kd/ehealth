import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from scripts.submit import Algorithm
from scripts.utils import Collection

from kdtools.datasets import RelationsDependencyParseActionsDataset
from kdtools.models import BiLSTMDoubleDenseOracleParser


class UHMajaModel(Algorithm):
    def __init__(self):
        self.model_taskA = None
        self.model_taskB = None

    def run(self, collection: Collection, *args, taskA: bool, taskB: bool, **kargs):
        return super().run(collection, *args, taskA=taskA, taskB=taskB, **kargs)

    def train(self, collection: Collection, n_epochs=100):
        self.model_taskA = self.train_taskA(collection, n_epochs)
        self.model_taskB = self.train_taskB(collection, n_epochs)

    def train_taskA(self, collection, n_epochs=100):
        pass

    def train_taskB(self, collection: Collection, n_epochs=100):
        dataset = RelationsDependencyParseActionsDataset(collection)
        model = BiLSTMDoubleDenseOracleParser(5, 14, 10, 50, batch_first=True)

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion_act = CrossEntropyLoss()
        criterion_rel = CrossEntropyLoss()

        for epoch in range(n_epochs):
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
                loss_act.backward(retain_graph = True)

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

    training = Collection().load(Path("data/training/scenario.txt"))
    algorithm.train(training)
