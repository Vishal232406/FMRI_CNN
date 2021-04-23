import warnings
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from numpy import vstack
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import pyplot as plt



from dataloader import RandomFmriDataset
from cnn_model import CNN_model

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

if __name__ == '__main__':
    learning_rate = 1e-3
    batch_size = 6
    num_epochs = 20
    compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = RandomFmriDataset(transform=compose)
    labels = dataset.labels
    model = CNN_model()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    label_array = []
    for i in labels:
        for j in i:
            label_array.append(j)

    data_and_labels = []
    for g in range(len(dataset)):
        data_and_labels.append([dataset[g], label_array[g]])


    def train_val_dataset(dataset, val_split=0.25):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
        datasets = {'train': Subset(dataset, train_idx), 'val': Subset(dataset, val_idx)}
        return datasets

    datasets = train_val_dataset(data_and_labels)
    dataloaders = {x: DataLoader(datasets[x], 6, shuffle=True, num_workers=4) for x in ['train', 'val']}
    x_train, y_train = next(iter(dataloaders['train']))
    x_val, y_val = next(iter(dataloaders['val']))

    def train_model(train_dl, model):
        criterion = torch.nn.BCELoss(size_average=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_array = []
        loss_values = []
        acc_values = []
        for epoch in range(num_epochs):
            for ind, (data, img_label) in enumerate(train_dl):
                inputs = data.permute(0, 4, 1, 2, 3),

                inputsV, labelsV = Variable(inputs[0]), Variable(img_label)
                inputsV = inputsV.to(torch.float32)
                labelsV = labelsV.to(torch.float32)
                y_pred = model(inputsV)

                loss = criterion(y_pred.squeeze(), labelsV)
                running_loss = abs(loss.item())
                loss_array.append(running_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_loss = np.average(loss_array)
            loss_values.append(avg_loss)
            acc = evaluate_model(dataloaders['val'], model)
            acc = acc * 100
            acc_values.append(acc)
            print(f'Epoch [{epoch + 1}/{num_epochs}], \n Loss: {abs(avg_loss):.4f}, Accuracy: {acc:.4f}')
        plot(acc_values, loss_values)

    def evaluate_model(test_dl, model):
        predictions, actuals = list(), list()
        accuracy_array = []
        for ind, (data, img_label) in enumerate(test_dl):
            inputs = data.permute(0, 4, 1, 2, 3),

            inputsV, labelsV = Variable(inputs[0]), Variable(img_label)
            inputsV = inputsV.to(torch.float32)
            labelsV = labelsV.to(torch.float32)

            yhat = model(inputsV)

            actual = labelsV.numpy()
            actual = actual.reshape((len(actual), 1))

            yhat = yhat.detach().numpy()
            yhat = yhat.round()

            predictions.append(yhat)
            actuals.append(actual)
            predictions, actuals = vstack(predictions), vstack(actuals)

            acc = accuracy_score(actuals, predictions)
            accuracy_array.append(acc)

            return acc


    def plot(x, y):
        plt.figure(figsize=(16, 5))
        plt.xlabel('EPOCHS')
        plt.ylabel('LOSS/ACC')

        plt.plot(x, 'r', label='ACCURACY')
        plt.plot(y, 'b', label='LOSS')
        plt.legend()
        plt.show()

    train_model(dataloaders['train'], model)
    evaluate_model(dataloaders['val'], model)
