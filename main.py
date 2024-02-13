import math
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Size
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torch.optim import SGD
from torchvision.transforms import ToTensor
import os


# using different matplotlib backend
matplotlib.use('QtAgg', force=True)


class NeuralNetwork(nn.Module):

    def __init__(self, trainingData : DataLoader):
        super(NeuralNetwork, self).__init__()
        self.trainingData: DataLoader = trainingData
        self.layer1 = nn.Linear(784, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 10)
        self.lossFunction = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.parameters(), lr=0.01)
        self.numEpochs = 10

    def forward(self, inputX: torch.Tensor) -> torch.Tensor:
        inputX = inputX.view(inputX.size(0), -1)
        inputX = nn.functional.relu(self.layer1(inputX))
        inputX = nn.functional.relu(self.layer2(inputX))
        inputX = self.layer3(inputX)
        return inputX

    def train(self, mode: bool = True) -> str:
        for epoch in range(self.numEpochs):
            for batch_id, (image, targets) in enumerate(self.trainingData):
                outputs = self.forward(image)
                loss = self.lossFunction(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(f"Epoch: {epoch + 1}/{self.numEpochs} Step: {math.ceil((self.trainingData.batch_size * (batch_id + 1)) / self.trainingData.batch_size)}")

        print("Done Training!")

    def test(self, inputData: torch.Tensor) -> int:
        outputs = self.forward(inputData)
        _, predictionLabel = torch.max(torch.softmax(outputs, dim=1), dim=1)
        return predictionLabel.item()


def main():
    transform = Compose([
        # range: [-1, 1] in image data
        ToTensor(),
        Normalize((0.5,), (0.5,), inplace=False)

    ])
    # get the MNIST hand written pictures to train and test on
    trainingDataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    testingDataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    trainingLoad = DataLoader(trainingDataset, batch_size=32, shuffle=True)

    # only test one data item at a time
    testingLoad = DataLoader(testingDataset, batch_size=1, shuffle=True)

    model = NeuralNetwork(trainingLoad)
    if not os.path.exists("./models/digitRecognizer.pth"):
        model.train()
        torch.save(model.state_dict(), './models/digitRecognizer.pth')
    else:
        model.load_state_dict(torch.load("./models/digitRecognizer.pth"))

    count = 0
    num_images = 6
    fig, axis = plt.subplots(1, num_images, figsize=(num_images, 1))
    # set main window title
    mainWindowManager = plt.get_current_fig_manager()
    mainWindowManager.window.setWindowTitle("Number Recognition")
    for i, (images, actualLabels) in enumerate(testingLoad):
        # only test the first 6 samples inside the testing data
        if i == num_images:
            break
        prediction = model.test(images)
        if actualLabels.item() == prediction:
            count = count + 1

        axis[i].imshow(images[0].numpy()[0], cmap="gray")
        axis[i].axis("off")
        axis[i].set_title(prediction)

    print(f'accuracy of the model for this run is: {(count / num_images) * 100} %')
    plt.show()

if __name__ == '__main__':
    main()
