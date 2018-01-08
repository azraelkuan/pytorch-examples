# -*- coding: utf-8 -*-
import unicodedata
import re

import torch
from torch.autograd import Variable


SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10

eng_prefixes = (
    'i am ', 'i m ',
    'he is ', 'he s ',
    'she is ', 'she s ',
    'you are ', 'you re ',
    'we are ', 'we re ',
    'they are ', 'they re '
)


class Lang(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # the count of SOS and EOS

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)


def unicode_to_ascii(s):
    """
    convert a unicode string to plain ascii
    :param s: the unicode string
    :return:  the plain ascii
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    """
    lower case, trim and remove no-letter character
    :param s:
    :return:
    """
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub('([.!?])', ' \1', s)
    s = re.sub('[^a-zA-Z.!?]+', ' ', s)
    return s


def read_lang(lang1, lang2, reverse=False):
    print("Reading Lines...")

    with open('data/{}-{}.txt'.format(lang1, lang2), 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pairs = [[normalize_string(s) for s in line.split('\t')] for line in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


def filter_pair(pair, reverse):
    if reverse:
        eng_text = pair[1]
    else:
        eng_text = pair[0]

    return len(pair[0].split()) < MAX_LENGTH and len(pair[1].split()) < MAX_LENGTH and eng_text.startswith(eng_prefixes)


def filter_pairs(pairs, reverse):
    return [pair for pair in pairs if filter_pair(pair, reverse)]


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_lang(lang1, lang2, reverse)
    print("Read {} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs, reverse)
    print("Trimmed to {} sentence pairs".format(len(pairs)))
    print("Counting word...", end="")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print('Done!')
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split()]


def variable_from_sentence(lang, sentence, use_cuda):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    return result.cuda() if use_cuda else result


def variable_from_pair(input_lang, output_lang, pair, use_cuda):
    input_variable = variable_from_sentence(input_lang, pair[0], use_cuda)
    output_variable = variable_from_sentence(output_lang, pair[1], use_cuda)
    return input_variable, output_variable


"""
unit test 
"""
# output_lang, input_lang, pairs = prepare_data('eng', 'fra', reverse=True)
# print(random.choice(pairs))

