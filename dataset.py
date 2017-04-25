import re
import random
import numpy as np
from tokenizer import tokenize_text


PAD_TOKEN = ' '
START_TOKEN = '<BOS>'
END_TOKEN = '<EOS>'

class Dataset(object):
    def __init__(self, input_filename=None, **params):
        if not input_filename:
            raise ValueError("No input filename supplied. See options with --help")
        text = open(input_filename).read()
        text = remove_unicode(text)
        text = text.lower()
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
        # TODO: Properly detokenize and join
        return [self.idx_to_word.get(i) for i in indices]

    def get_batch(self, **params):
        batch_size = params['batch_size']
        x, y = self.get_example(**params)
        X = np.zeros((batch_size,) + x.shape)
        Y = np.zeros((batch_size,) + y.shape)
        for i in range(batch_size):
            X[i] = x
            Y[i] = y
            x, y = self.get_example(**params)
        return X, Y

    def get_example(self, **params):
        sentence = random.choice(self.sentences)
        sentence = '{} {} {}'.format(START_TOKEN, sentence, END_TOKEN)
        si = self.indices(sentence)
        idx = np.random.randint(0, len(si))
        left, right = si[:idx], [si[idx]]
        left = left_pad(left, **params)
        left = np.array(left)
        right = np.array(right)
        return left, right

    def get_empty_batch(self, batch_size=1, max_words=12, **params):
        X = np.zeros((batch_size, max_words))
        X[:, -1] = self.word_to_idx[START_TOKEN]
        return X


def remove_unicode(text):
    return re.sub(r'[^\x00-\x7f]', r'', text)


def left_pad(indices, max_words=10, **kwargs):
    res = np.zeros(max_words, dtype=int)
    if len(indices) > max_words:
        indices = indices[-max_words:]
    res[max_words - len(indices):] = indices
    return res
