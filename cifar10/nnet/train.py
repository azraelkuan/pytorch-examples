import torchvision
import torch
import model
import argparse
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pre_process import transform_train, transform_test

# whether use gpu
use_cuda = torch.cuda.is_available()

# default parameters
DATA_ROOT = '../data/'
num_epochs = 50
batch_size = 128

model_names = {
    'dnn': model.DNN(3072, 4096, 10),
    'resnet18': model.ResNet18(),
    'resnet34': model.ResNet34(),
    'resnet50': model.ResNet50()
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='dnn', help="the type of model")
    parser.add_argument('--lr', type=float, default=0.1, help='the initial learning rate')
    parser.add_argument('--batch_size', type=int, default=num_epochs, help='the batch size')
    parser.add_argument('--num_epochs', type=int, default=batch_size, help='the epoch')
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # get train data and test data
    train_dataset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=transform_train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=transform_test)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    net = model_names[args.model_type]
    if use_cuda:
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True, min_lr=1e-4, factor=0.1)
    test_results = []
    current_test_acc = 0

    for epoch in range(args.num_epochs):
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
                      % (epoch + 1, args.num_epochs, i + 1,
                         len(train_dataset) // args.batch_size, total_loss/(i+1),
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
        test_results.append(100 * correct / total)
        if correct / total > current_test_acc:
            current_test_acc = correct / total
            torch.save({'model': net, 'acc': current_test_acc}, 'model/{}.pkl'.format(args.model_type))

        print('Test Loss %.4f Accuracy : %.2f %%' % (total_loss/len(test_dataloader), 100 * correct / total))

        scheduler.step(total_loss/len(test_dataloader))
    print("Best Test Acc: {}".format(max(test_results)))


if __name__ == '__main__':
    main()
