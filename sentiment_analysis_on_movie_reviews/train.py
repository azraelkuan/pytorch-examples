import torch
import argparse
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from load_data import SentimentDataSet, create_lexicon


class DnnModel(nn.ModuleList):

    def __init__(self, input_dim, output_dim, hidden_size):
        super(DnnModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_dim)
        self.nonlinear = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        out = self.bn(self.nonlinear(self.linear1(x)))
        out = self.bn(self.nonlinear(self.linear2(out)))
        out = self.bn(self.nonlinear(self.linear3(out)))
        out = self.linear4(out)
        return out


def main(train_file, test_file, dev_file):
    use_cuda = torch.cuda.is_available()
    print("use cuda: {}".format(use_cuda))

    print("prepare data...", end="")
    lex = create_lexicon(train_file, test_file, dev_file)
    train_dataset = SentimentDataSet(lex, train_file, mode="train")
    dev_dataset = SentimentDataSet(lex, dev_file, mode="train")

    # test_dataset = SentimentDataSet(lex, test_file, mode="test")
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=256, shuffle=False)
    print("Done!")

    net = DnnModel(len(lex), 5, 2048)
    if use_cuda:
        net = net.cuda()
    optimizer = optim.Adam(lr=0.001, params=net.parameters())
    cross_entropy = nn.CrossEntropyLoss()

    for epoch in range(100):
        epoch_loss = 0.
        for each_data in train_dataloader:
            input_data = Variable(each_data['input_data']).float()
            target = Variable(each_data['label'])
            if use_cuda:
                input_data, target = input_data.cuda(), target.cuda()

            optimizer.zero_grad()
            predict = net(input_data)
            loss = cross_entropy(predict, target.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]

        correct = 0.
        total = 0
        for each_data in dev_dataloader:
            input_data = Variable(each_data['input_data']).float()
            labels = Variable(each_data['label'].view(-1))
            if use_cuda:
                input_data, labels = input_data.cuda(), labels.cuda()
            predict = net(input_data)
            _, predicted = torch.max(predict.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        print("Epoch : {} \t Loss : {} \t accuracy : {}".format(epoch, epoch_loss / len(train_dataloader), correct / total))


def get_args():
    parser = argparse.ArgumentParser(description="Usage:")
    parser.add_argument("--train_file", type=str, default="data/train.tsv", help="training file")
    parser.add_argument("--test_file", type=str, default="data/test.tsv", help="test file")
    parser.add_argument("--dev_file", type=str, default="data/dev.tsv", help="dev file")
    parser.add_argument("--output_file", type=str, default="data/submission.csv", help='the kaggle submission file')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args.train_file, args.test_file, args.dev_file)
