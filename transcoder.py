import os
import sys
import random
import numpy as np
from keras.utils import to_categorical
from keras import layers, models

from dataset import Dataset, left_pad, right_pad


def train(model, encoder_dataset, decoder_dataset, **params):
    def get_batch():
        batch_size = params['batch_size']
        max_words = params['max_words']
        decoder_vocab_len = len(decoder_dataset.vocab)
        sentence_count = len(encoder_dataset.sentences)
        X = np.zeros((batch_size, max_words), dtype=int)
        Y = np.zeros((batch_size, max_words, decoder_vocab_len))
        for i in range(batch_size):
            j = np.random.randint(0, sentence_count)
            sent_in = encoder_dataset.sentences[j]
            sent_out = decoder_dataset.sentences[j]
            x = left_pad(encoder_dataset.indices(sent_in)[:max_words], **params)
            y = right_pad(decoder_dataset.indices(sent_out), **params)
            # TODO: Drop to_categorical and go back to sparse_categorical_crossentropy?
            X[i], Y[i] = x, to_categorical(y, decoder_vocab_len)
        return X, Y

    def gen():
        while True:
            yield get_batch()
    batches_per_epoch = params['batches_per_epoch']
    X, Y = get_batch()
    model.train_on_batch(X, Y)
    model.fit_generator(gen(), steps_per_epoch=batches_per_epoch)



def demonstrate(model, encoder_dataset, decoder_dataset, input_text=None, **params):
    max_words = params['max_words']
    X = encoder_dataset.get_empty_batch(**params)
    for i in range(params['batch_size']):
        if not input_text:
            input_text = random.choice(encoder_dataset.sentences)
        X[i] = left_pad(encoder_dataset.indices(input_text)[:max_words], **params)
    batch_size, max_words = X.shape

    preds = model.predict(X)
    Y = np.argmax(preds, axis=-1)
    for i in range(len(Y)):
        left = ' '.join(encoder_dataset.words(X[i]))
        right = ' '.join(decoder_dataset.words(Y[i]))
        print('{} --> {}'.format(left, right))


# Actually boltzmann(log(x)) for stability
def boltzmann(pdf, temperature=1.0, epsilon=1e-5):
    if temperature < epsilon:
        return pdf / (pdf.sum() + epsilon)
    pdf = np.log(pdf) / temperature
    x = np.exp(pdf)
    sums = np.sum(x, axis=-1)[:, np.newaxis] + epsilon
    return x / sums


def sample(pdfs):
    max_words, vocab_size = pdfs.shape
    samples = np.zeros(max_words)
    for i in range(len(samples)):
        samples[i] = np.random.choice(np.arange(vocab_size), p=pdfs[i])
    return samples


def build_model(encoder_dataset, decoder_dataset, **params):
    encoder = build_encoder(encoder_dataset, **params)
    decoder = build_decoder(decoder_dataset, **params)

    combined = models.Sequential()
    combined.add(encoder)
    combined.add(decoder)
    return encoder, decoder, combined


def build_encoder(dataset, **params):
    rnn_type = getattr(layers, params['rnn_type'])
    wordvec_size = params['wordvec_size']
    rnn_size = params['rnn_size']
    rnn_layers = params['rnn_layers']
    max_words = params['max_words']
    thought_vector_size = params['thought_vector_size']
    vocab_len = len(dataset.vocab)

    inp = layers.Input(shape=(max_words,), dtype='int32')
    x = layers.Embedding(vocab_len, wordvec_size, input_length=max_words, mask_zero=True)(inp)
    for _ in range(rnn_layers - 1):
        x = rnn_type(rnn_size, return_sequences=True)(x)
    x = rnn_type(rnn_size)(x)
    x = layers.Activation('relu')(x)
    encoded = layers.Dense(thought_vector_size, activation='tanh')(x)
    moo = models.Model(inputs=inp, outputs=encoded)
    return moo


def build_decoder(dataset, **params):
    rnn_type = getattr(layers, params['rnn_type'])
    wordvec_size = params['wordvec_size']
    rnn_size = params['rnn_size']
    rnn_layers = params['rnn_layers']
    max_words = params['max_words']
    thought_vector_size = params['thought_vector_size']
    vocab_len = len(dataset.vocab)

    inp = layers.Input(shape=(thought_vector_size,))
    x = layers.RepeatVector(max_words)(inp)
    for _ in range(rnn_layers - 1):
        x = rnn_type(rnn_size, return_sequences=True)(x)
    x = rnn_type(rnn_size, return_sequences=True)(x)
    word_preds = layers.TimeDistributed(layers.Dense(vocab_len, activation='softmax'))(x)
    return models.Model(inputs=inp, outputs=word_preds)


def main(**params):
    print("Loading dataset")
    # TODO: Separate datasets
    encoder_dataset = Dataset(params['encoder_input_filename'], **params)
    decoder_dataset = Dataset(params['decoder_input_filename'], **params)
    print("Dataset loaded")

    print("Building model")
    encoder, decoder, combined = build_model(encoder_dataset, decoder_dataset, **params)
    print("Model built")

    if os.path.exists(params['encoder_weights']):
        encoder.load_weights(params['encoder_weights'])
    if os.path.exists(params['decoder_weights']):
        decoder.load_weights(params['decoder_weights'])

    combined.compile(loss='categorical_crossentropy', optimizer='adam')

    if params['mode'] == 'train':
        for epoch in range(params['epochs']):
            train(combined, encoder_dataset, decoder_dataset, **params)
            demonstrate(combined, encoder_dataset, decoder_dataset, **params)
            encoder.save_weights(params['encoder_weights'])
            decoder.save_weights(params['decoder_weights'])
    elif params['mode'] == 'demo':
        print("Demonstration time!")
        params['batch_size'] = 1
        while True:
            inp = raw_input("Type a complete sentence in the input language: ")
            inp = inp.decode('utf-8').lower()
            demonstrate(combined, encoder_dataset, decoder_dataset, input_text=inp, **params)
