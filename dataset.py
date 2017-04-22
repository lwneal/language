import random
import numpy as np
from tokenizer import tokenize_text


PAD_TOKEN = ' '
START_TOKEN = '<BOS>'
END_TOKEN = '<EOS>'
class Dataset(object):
    def __init__(self, input_filename=None, **params):
        text = open(input_filename).read()
        text = tokenize_text(text)
        self.sentences = text.splitlines()
        self.vocab = sorted(list(set(text.split() + [PAD_TOKEN, START_TOKEN, END_TOKEN])))
        self.word_to_idx = {}
        self.idx_to_word = {}
        for i, word in enumerate(self.vocab):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word

    def indices(self, words):
        # TODO: Properly tokenize
        return [self.word_to_idx[w] for w in words.split()]

    def words(self, indices):
        return [self.idx_to_word.get(i) for i in indices]


def get_batch(dataset, **params):
    batch_size = params['batch_size']
    x, y = get_example(dataset, **params)
    X = np.zeros((batch_size,) + x.shape)
    Y = np.zeros((batch_size,) + y.shape)
    for i in range(batch_size):
        X[i] = x
        Y[i] = y
        x, y = get_example(dataset, **params)
    return X, Y


def get_example(dataset, **params):
    sentence = random.choice(dataset.sentences)
    si = dataset.indices(sentence)
    idx = np.random.randint(0, len(si))
    left, right = si[:idx], [si[idx]]
    left = left_pad(left, **params)
    left = np.array(left)
    right = np.array(right)
    return left, right


def left_pad(indices, max_words=10, **kwargs):
    res = np.zeros(max_words, dtype=int)
    if len(indices) > max_words:
        indices = indices[-max_words:]
    res[max_words - len(indices):] = indices
    return res
