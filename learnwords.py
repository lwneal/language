import os
import sys
import random
import numpy as np
from keras import layers, models

from dataset import Dataset


def train(model, dataset, **params):
    batches_per_epoch = params['batches_per_epoch']
    def batcher():
        while True:
            yield dataset.get_batch(**params)
    model.fit_generator(batcher(), steps_per_epoch=batches_per_epoch)


def demonstrate(model, dataset, **params):
    X = dataset.get_empty_batch(**params)
    batch_size, max_words = X.shape
    for i in range(max_words - 1):
        T = temperature = 1 - (float(i) / max_words)
        X = np.roll(X, -1, axis=1)
        pdf = boltzmann(model.predict(X), T)
        X[:, -1] = sample(pdf)
    for i in range(batch_size):
        print(' '.join(dataset.words(X[i])))


# Actually boltzmann(log(x)) for stability
def boltzmann(pdf, temperature=1.0, epsilon=1e-6):
    pdf = np.log(pdf) / temperature
    x = np.exp(pdf)
    sums = np.sum(x, axis=-1)[:, np.newaxis] + epsilon
    return x / sums


def sample(pdfs):
    samples = np.zeros(pdfs.shape[0])
    for i in range(len(samples)):
        samples[i] = np.argmax(np.random.multinomial(1, pdfs[i], 1))
    return samples


def build_model(dataset, **params):
    rnn_type = getattr(layers, params['rnn_type'])
    wordvec_size = params['wordvec_size']
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


def main(**params):
    dataset = Dataset(**params)

    model = build_model(dataset, **params)

    if os.path.exists(params['weights_filename']):
        model.load_weights(params['weights_filename'])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    for epoch in range(params['epochs']):
        train(model, dataset, **params)
        demonstrate(model, dataset, **params)
        model.save_weights(params['weights_filename'])

