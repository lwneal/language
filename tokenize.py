#!/usr/bin/env python
import sys
import nltk
import re


def main(input_fp, output_fp):
    text = input_fp.read()
    sentences = nltk.sent_tokenize(text)
    for s in sentences:
        words = nltk.word_tokenize(s)
        output_fp.write(' '.join(words) + '\n')
    output_fp.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: {} input.txt [output.txt]".format(sys.argv[0]))
        exit()
    if sys.argv[1] == '-':
        input_fp = sys.stdin
    else:
        input_fp = open(sys.argv[1])
    if len(sys.argv) > 2:
        output_fp = open(sys.argv[2], 'w')
    else:
        output_fp = sys.stdout
    main(input_fp, output_fp)
