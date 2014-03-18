__author__ = 'arenduchintala'

import gensim
from math import sqrt
from utilz import tokenize
import sys

lsi = None
dictionary = None
tfidf = None


def cosine_similarity(v1, v2):
    vec1 = [i[1] for i in v1]
    vec2 = [i[1] for i in v2]
    mag1 = sqrt(sum([i ** 2 for i in vec1]))
    mag2 = sqrt(sum([i ** 2 for i in vec2]))
    if mag2 == 0 or mag1 == 0:
        return 0.0
    dotsum = sum([x * y for x, y in zip(vec1, vec2)])
    return float(dotsum) / float(mag1 * mag2)


def get_vec(hyp):
    vec = lsi[dictionary.doc2bow(hyp)]
    return vec


if __name__ == '__main__':
    model_prefix = sys.argv[1]
    answers = [int(i) for i in open('eval-data/dev.answers', 'r').readlines()]
    training_data = [tuple(i.split('|||')) for i in open('eval-data/hyp1-hyp2-ref').readlines()]
    lsi = gensim.models.lsimodel.LsiModel.load(model_prefix + '.lsi')
    dictionary = gensim.corpora.Dictionary.load(model_prefix + '.dict')
    tfidf = gensim.models.tfidfmodel.TfidfModel.load(model_prefix + '.tfidf')
    incorrect = 0
    correct = 0
    for idx, (hyp1, hyp2, ref) in enumerate(training_data[:len(answers)]):
        hyp1 = tokenize(hyp1)
        hyp2 = tokenize(hyp2)
        ref = tokenize(ref)
        if answers[idx] != 0:
            t_hyp1 = [h for h in hyp1 if h not in hyp2]
            t_hyp2 = [h for h in hyp2 if h not in hyp1]
            v_hyp1 = get_vec(t_hyp1)
            v_hyp2 = get_vec(t_hyp2)
            v_ref = get_vec(ref)
            cs1 = cosine_similarity(v_hyp1, v_ref)
            cs2 = cosine_similarity(v_hyp2, v_ref)
            guess = 1 if cs1 > cs2 else -1
            incorrect += 1 if guess != answers[idx] and answers[idx] != 0 else 0
            correct += 1 if guess == answers[idx] and answers[idx] != 0 else 0
            print idx, 'hyp1:', "%.4f" % cs1, 'hyp2:', "%.4f" % cs2, 'guess:', guess, 'ans:', answers[idx]

            if guess != answers[idx] and abs(cs1 - cs2) > 0.4:
                print idx
                print hyp1, len(v_hyp1)
                print hyp2, len(v_hyp2)
                print t_hyp1
                print t_hyp2
                print ref

    print 'total check', incorrect + correct
    print 'incorrect', incorrect, float(incorrect) / float(incorrect + correct)
    print 'correct', correct, float(correct) / float(incorrect + correct)