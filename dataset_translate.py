import sys
import random
import nltk
nltk.download('punkt')

sentence_filename = 'sentences.csv'
links_filename = 'links.csv'


def get_dataset(src_language='eng', dst_language='deu'):
    sentences = open(sentence_filename).readlines()
    links = open(links_filename).readlines()
    src = get_by_language(sentences, src_language)
    dst = get_by_language(sentences, dst_language)
    matches = get_translations(src, dst, links)
    print("Loaded {} translations from {} to {}".format(len(matches), src_language, dst_language))
    print(random.choice(matches))
    return matches

def get_by_language(sentences, lang='eng'):
    res = {}
    for s in sentences:
        sentence_id, sentence_lang, text = s.split('\t')
        if sentence_lang == lang:
            res[int(sentence_id)] = text
    return res

def get_translations(src, dst, links):
    translations = []
    for l in links:
        src_id, dst_id = map(int, l.split())
        if src_id in src and dst_id in dst:
            src_sent = nltk.word_tokenize(src[src_id].strip().decode('utf-8'))
            dst_sent = nltk.word_tokenize(dst[dst_id].strip().decode('utf-8'))
            translations.append((src_sent, dst_sent))
    return translations


if __name__ == '__main__':
    src_lang = sys.argv[1]
    dst_lang = sys.argv[2]
    sentences = get_dataset(src_lang, dst_lang)
    for src, dst in sentences:
        line = (' '.join(src) + ' <EOS> ' + ' '.join(dst))
        print(line.encode('utf-8'))
