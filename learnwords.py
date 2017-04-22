# Usage: python learn.py sentences.txt model.h5
import os
import random
import numpy as np


# TODO: get from docopt
params = {
     'batch_size': 16,
     'max_words': 12,
     'rnn_type': 'LSTM',
     'rnn_size': 1024,
     'wordvec_size': 1024,
     'rnn_layers': 1,
     'model_filename': sys.argv[2],
     'input_filename': sys.argv[1],
}

class Dataset(object):
    def __init__(self, input_filename=None, **params):
        text = open(input_filename).read()
        self.vocab = sorted(list(set(text.split())))
        self.sentences = text.splitlines()

def build_model(dataset, **params):
    from keras import layers, models
    wordvec_size = params['wordvec_size']
    rnn_type = getattr(layers, params['rnn_type'])
    rnn_size = params['rnn_size']
    rnn_layers = params['rnn_layers']
    max_words = params['max_words']
    vocab_len = len(dataset.vocab)
    inp = layers.Input(shape=(max_words,), dtype='int32')
    x = layers.Embedding(vocab_len, wordvec_size, input_length=max_words, mask_zero=True)(inp)
    for _ in range(rnn_layers - 1):
        x = rnn_type(rnn_size, return_sequences=True)(x)
    x = rnn_type(rnn_size)(x)
    x = layers.Dense(vocab_len, activation='softmax')(x)
    moo = models.Model(inputs=inp, outputs=x)
    return moo


def get_example(dataset, **params):
    s = words.indices(random.choice(sentences))
    idx = np.random.randint(0, len(s))
    left, right = s[:idx], [s[idx]]
    left = util.left_pad(left, **params)
    left = np.array(left)
    right = np.array(right)
    return left, right


def left_pad(indices, max_words=10, **kwargs):
    res = np.zeros(max_words, dtype=int)
    if len(indices) > max_words:
        indices = indices[-max_words:]
    res[max_words - len(indices):] = indices
    return res


def batch(example_fn, *args, **params):
    batch_size = params['batch_size']
    x, y = example_fn(*args, **params)
    X = np.zeros((batch_size,) + x.shape)
    Y = np.zeros((batch_size,) + y.shape)
    for i in range(batch_size):
        X[i] = x
        Y[i] = y
        x, y = example_fn(**params)
    return X, Y


def validate(model, dataset, **params):
    X, _ = get_batch(get_example, dataset, **params)
    for _ in range(params['max_words']/2):
        preds = np.argmax(model.predict(X), axis=1)
        X = np.roll(X, -1, axis=1)
        X[:,-1] = preds
    for i in range(params['batch_size']):
        print words.words(X[i])


def train(model, dataset, **params):
    def batcher():
        while True:
            yield batch(get_example, dataset, **params)
    model.fit_generator(batcher(), steps_per_epoch=100)


if __name__ == '__main__':
    if os.path.exists(params['model_filename']):
        model = models.load_model(params['model_filename'])
    else:
        model = build_model(**params)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    dataset = Dataset(filename, **params)

    while True:
        train(model, dataset, **params)
        validate(model, dataset, **params)
        model.save_weights(params['model_filename'])

