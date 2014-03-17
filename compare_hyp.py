__author__ = 'arenduchintala'

import gensim
from math import sqrt
import nltk

lsi = None
dictionary = None
tfidf = None


def cosine_similarity(v1, v2):
    vec1 = [i[1] for i in v1]
    vec2 = [i[1] for i in v2]
    mag1 = sqrt(sum([i ** 2 for i in vec1]))
    mag2 = sqrt(sum([i ** 2 for i in vec2]))
    dotsum = sum([x * y for x, y in zip(vec1, vec2)])
    return float(dotsum) / float(mag1 * mag2)


def get_vec(hyp):
    vec = lsi[tfidf[dictionary.doc2bow(nltk.word_tokenize(hyp.lower()))]]
    return vec


if __name__ == '__main__':
    answers = [int(i) for i in open('eval-data/dev.answers', 'r').readlines()]
    training_data = [tuple(i.split('|||')) for i in open('eval-data/hyp1-hyp2-ref').readlines()]
    lsi = gensim.models.lsimodel.LsiModel.load('data/corpus.lsi')
    dictionary = gensim.corpora.Dictionary.load('data/corpus.dict')
    tfidf = gensim.models.tfidfmodel.TfidfModel.load('data/corpus.tfidf')
    incorrect = 0
    correct = 0
    for idx, (hyp1, hyp2, ref) in enumerate(training_data[:len(answers)]):
        v_hyp1 = get_vec(hyp1)
        v_hyp2 = get_vec(hyp2)
        v_ref = get_vec(ref)
        cs1 = cosine_similarity(v_hyp1, v_ref)
        cs2 = cosine_similarity(v_hyp2, v_ref)
        guess = 1 if cs1 > cs2 else -1
        print idx, 'hyp1:', "%.4f" % cs1, 'hyp2:', "%.4f" % cs2, 'guess:', guess, 'ans:', answers[idx]
        incorrect += 1 if guess != answers[idx] and answers[idx] != 0 else 0
        correct += 1 if guess == answers[idx] and answers[idx] != 0 else 0
    print 'total check', incorrect + correct
    print 'incorrect', incorrect, float(incorrect) / float(incorrect + correct)
    print 'correct', correct, float(correct) / float(incorrect + correct)