__author__ = 'arenduchintala'

import gensim
from math import sqrt
import nltk


def cosine_similarity(v1, v2):
    vec1 = [i[1] for i in v1]
    vec2 = [i[1] for i in v2]
    mag1 = sqrt(sum([i ** 2 for i in vec1]))
    mag2 = sqrt(sum([i ** 2 for i in vec2]))
    dotsum = sum([x * y for x, y in zip(vec1, vec2)])
    return float(dotsum) / float(mag1 * mag2)


if __name__ == '__main__':
    answers = [int(i) for i in open('eval-data/dev.answers', 'r').readlines()]
    training_data = [tuple(i.split('|||')) for i in open('eval-data/hyp1-hyp2-ref').readlines()]
    lsi = gensim.models.lsimodel.LsiModel.load('data/corpus.lsi')
    dict = gensim.corpora.Dictionary.load('data/corpus.dict')
    incorrect = 0
    for idx, (hyp1, hyp2, ref) in enumerate(training_data[:len(answers)]):
        v_hyp1 = lsi[dict.doc2bow(nltk.word_tokenize(hyp1.lower()))]
        v_hyp2 = lsi[dict.doc2bow(nltk.word_tokenize(hyp2.lower()))]
        v_ref = lsi[dict.doc2bow(nltk.word_tokenize(ref.lower()))]
        cs1 = cosine_similarity(v_hyp1, v_ref)
        cs2 = cosine_similarity(v_hyp2, v_ref)
        guess = 1 if cs1 > cs2 else -1
        print idx, 'hyp1:', "%.4f" % cs1, 'hyp2:', "%.4f" % cs2, 'guess:', guess, 'ans:', answers[idx]
        incorrect += 1 if guess != answers[idx] and answers[idx] != 0 else 0