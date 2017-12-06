import torchvision
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pre_process import transform_train, transform_test
from model import LogisticRegression

use_cuda = torch.cuda.is_available()
num_epochs = 10
batch_size = 32


train_dataset = torchvision.datasets.CIFAR10(root="./data/", train=True, download=False, transform=transform_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

net = LogisticRegression(3072, 10)
if use_cuda:
    net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        images = Variable(images.view(-1, 32*32*3))
        labels = Variable(labels)
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

    correct = 0
    total = 0
    for images, labels in test_dataloader:
        images = Variable(images.view(-1, 32*32*3))
        labels = Variable(labels)
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()

    print('Accuracy : %d %%' % (100 * correct / total))
