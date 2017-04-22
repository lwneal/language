import os
import sys
import random
import numpy as np
from keras import layers, models

from dataset import Dataset


def train(model, dataset, **params):
    def batcher():
        while True:
            yield dataset.get_batch(**params)
    model.fit_generator(batcher(), steps_per_epoch=100)


def validate(model, dataset, **params):
    X, _ = dataset.get_batch(**params)
    for _ in range(params['max_words'] / 2):
        preds = np.argmax(model.predict(X), axis=1)
        X = np.roll(X, -1, axis=1)
        X[:,-1] = preds
    for i in range(params['batch_size']):
        print(' '.join(dataset.words(X[i])))


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
        validate(model, dataset, **params)
        model.save_weights(params['weights_filename'])

