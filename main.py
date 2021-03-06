"""
Usage:
        main.py [options]

Options:
      --encoder-input-filename=<txt>    Input text file the encoder will read (eg. English sentences)
      --decoder-input-filename=<txt>    Input text file the decoder will try to copy (eg. German sentences)
      --encoder-weights=<name>          Filename for saved model [default: default_encoder.h5]
      --decoder-weights=<name>          Filename for saved model [default: default_decoder.h5]
      --epochs=<epochs>                 Number of epochs to train [default: 2000].
      --batches-per-epoch=<b>           Number of batches per epoch [default: 1000].
      --batch-size=<size>               Batch size for training [default: 16]
      --max-words=<words>               Number of words of context (the N in N-gram) [default: 12]
      --wordvec-size=<size>             Number of units in word embedding [default: 1024]
      --rnn-type=<type>                 One of LSTM, GRU [default: LSTM]
      --rnn-size=<size>                 Number of output units in RNN [default: 1024]
      --rnn-layers=<layers>             Number of layers of RNN to use [default: 1]
      --thought-vector-size=<size>      Size of encoder output (decoder input) vector [default: 1024]
      --tokenize=<tokenize>             If True, input text will be tokenized [default: False]
      --lowercase=<lower>               If True, lowercase all words [default: True]
      --mode=<mode>                     One of train, demo [default: train]
      --max-temperature=<temp>          Sampling temperature for log-Boltzmann distribution [default: 1.0]
"""
from docopt import docopt
from pprint import pprint


def get_params():
    args = docopt(__doc__)
    return {argname(k): argval(args[k]) for k in args}


def argname(k):
    return k.strip('<').strip('>').strip('--').replace('-', '_')


def argval(val):
    if hasattr(val, 'lower') and val.lower() in ['true', 'false']:
        return val.lower().startswith('t')
    try:
        return int(val)
    except:
        pass
    try:
        return float(val)
    except:
        pass
    if val == 'None':
        return None
    return val


if __name__ == '__main__':
    params = get_params()
    import transcoder
    transcoder.main(**params)
