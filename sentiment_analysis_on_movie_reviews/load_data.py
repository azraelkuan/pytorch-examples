import torch
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from torch.utils.data import Dataset


def create_lexicon(train_file, test_file, dev_file):
    lex = []
    lex += precess_file(train_file)
    lex += precess_file(test_file)
    lex += precess_file(dev_file)
    lemmatizer = WordNetLemmatizer()
    # "cats" ==> "cat"
    lex = [lemmatizer.lemmatize(word) for word in lex]
    word2count = Counter(lex)
    # compute the word count and remove some useless word
    lex = []
    for word in word2count:
        if 100 < word2count[word] < 20000:
            lex.append(word)
    return lex


def precess_file(txt):
    lex = []
    with open(txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            # "this is a word." ==> ['this', 'is', 'a', 'word', '.']
            line = line.strip().lower().split("\t")
            if len(line) > 2:
                words = word_tokenize(line[2])
                lex += words
            else:
                pass
    return lex


class SentimentDataSet(Dataset):
    def __init__(self, lex, input_file, mode):
        super(SentimentDataSet, self).__init__()
        self.mode = mode
        self.lemmatizer = WordNetLemmatizer()
        self.data = []
        self.label = []
        self.add_item(input_file, lex, mode)

    def __getitem__(self, idx):
        if self.mode == 'train':
            sample = {'input_data': torch.LongTensor(self.data[idx]), 'label': torch.LongTensor([self.label[idx]])}
        if self.mode == 'test':
            sample = {'input_data': torch.LongTensor(self.data[idx])}
        return sample

    def __len__(self):
        return len(self.data)

    def string2vector(self, lex, line, mode):
        line = line.strip().lower().split("\t")
        if len(line) > 2:
            words = word_tokenize(line[2])
            words = [self.lemmatizer.lemmatize(word) for word in words]

            feature = [0 for _ in range(len(lex))]
            for word in words:
                if word in lex:
                    feature[lex.index(word)] = 1
            if mode == "train":
                target = int(line[3])
                self.data.append(feature)
                self.label.append(target)
            if mode == "test":
                self.data.append(feature)

    def add_item(self, txt, lex, mode):
        with open(txt, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:
                self.string2vector(lex, line, mode)


"""
test dataset
"""
# if __name__ == '__main__':
#     lex_text = create_lexicon("data/train.tsv", 'data/test.tsv')
#     dataset = SentimentDataSet(lex_text, "data/train.tsv", "train")
#     for sample in dataset:
#         print(sample['input_data'], sample['label'])
