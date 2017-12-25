# -*- coding: utf-8 -*-
import torch
import model
import numpy as np
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from pre_process import transform_train, transform_test
from skorch.net import NeuralNetClassifier



batch_size = 256
use_cuda = torch.cuda.is_available()

train_dataset = torchvision.datasets.CIFAR10(root="../data/", train=True, download=False, transform=transform_train)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataset = torchvision.datasets.CIFAR10(root="../data", train=False, download=False, transform=transform_test)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

x_train = train_dataset.train_data
y_train = train_dataset.train_labels

mean = np.mean(x_train)
std = np.std(x_train)
x_train = (x_train - mean) / std

x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
y_train = np.array(y_train).astype(np.int64)

net = NeuralNetClassifier(
    module=model.DNN,
    module__input_dim=3072,
    module__hidden_dim=4096,
    module__output_dim=10,
    max_epochs=10,
    batch_size=256,
    optimizer=optim.SGD,
    optimizer__lr=0.001,
    optimizer__momentum=0.9,
    use_cuda=use_cuda,
    train_split=None
)

net.fit()
y_predict = net.predict(x_train)
print(y_predict)
