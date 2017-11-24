from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from torch.utils.data import Dataset


def create_lexicon(train_file, test_file):
    lex = []
    lex += precess_file(train_file)
    lex += precess_file(test_file)
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
    with open(txt, 'r', encoding='utf-8') as f:
        lex = []
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


class CommentDataSet(Dataset):
    def __init__(self, lex, train_file, test_file):
        super(CommentDataSet, self).__init__()
        self.lemmatizer = WordNetLemmatizer()
        self.data = []
        self.add_item(train_file, lex, mode="train")
        self.add_item(test_file, lex, mode="test")

    def __getitem__(self, idx):
        return self.data[idx]

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
                target = [0 for _ in range(5)]
                target[int(line[3])] = 1
            if mode == "test":
                target = [0 for _ in range(5)]
            return feature, target
        else:
            return None

    def add_item(self, txt, lex, mode):
        with open(txt, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:
                feature, target = self.string2vector(lex, line, mode)
                self.data.append(
                    [feature, target]
                )


"""
test dataset
"""
# if __name__ == '__main__':
#     lex_text = create_lexicon("data/train.tsv", 'data/test.tsv')
#     dataset = CommentDataSet(lex_text, "data/train.tsv", 'data/test.tsv')
#     for each in dataset:
#         print(each)
