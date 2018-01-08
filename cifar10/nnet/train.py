import torchvision
import torch
import model
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pre_process import transform_train, transform_test


use_cuda = torch.cuda.is_available()
num_epochs = 50
batch_size = 100

train_dataset = torchvision.datasets.CIFAR10(root="../data/", train=True, download=False, transform=transform_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(root="../data", train=False, download=False, transform=transform_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# net = model.LogisticRegression(3072, 10)
# net = model.DNN(3072, 4096, 10)
# net = model.ResNet18()
net = model.VGG('VGG11')
if use_cuda:
    net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)
scheduler = ReduceLROnPlateau(optimizer, verbose=True, min_lr=1e-5)
best_test = []

for epoch in range(num_epochs):
    correct = 0
    total = 0
    total_loss = 0.
    for i, (images, labels) in enumerate(train_dataloader):
        images = Variable(images)
        labels = Variable(labels)
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        total_loss += loss.data[0]

        if (i + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %f %%'
                  % (epoch + 1, num_epochs, i + 1,
                     len(train_dataset) // batch_size, total_loss/(i+1),
                     100*round(correct / total, 4)))

    correct = 0
    total = 0
    total_loss = 0.
    for images, labels in test_dataloader:
        images = Variable(images)
        labels = Variable(labels)
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        loss = criterion(outputs, labels)
        total_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
    best_test.append(100 * correct / total)
    print('Test Loss %.4f Accuracy : %.4f %%' % (total_loss/len(test_dataloader), 100 * correct / total))

    scheduler.step(total_loss/len(test_dataloader))
print("Best Test Acc: {}".format(max(best_test)))
