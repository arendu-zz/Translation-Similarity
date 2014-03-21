__author__ = 'arenduchintala'
import sys
from utilz import tokenize, STOP_WORDS, VEC_LEN
import gensim


def sentences(corpus_files):
    print 'making texts...'
    texts = []
    for cs in corpus_files:
        print 'reading', cs
        for doc in open(cs, 'r'):
            texts.append(set([i for i in tokenize(doc.decode('utf-8', 'ignore')) if i not in STOP_WORDS]))
            #yield [i for i in tokenize(doc.decode('utf-8', 'ignore')) if i not in STOP_WORDS]
    return texts


if __name__ == '__main__':
    save_model = sys.argv[1]
    corpus_files = sys.argv[2:]
    model_w2v = gensim.models.Word2Vec(sentences=sentences(corpus_files), workers=4, size=VEC_LEN)
    model_w2v.save(save_model + '.w2v')
    print len(model_w2v['man'])