__author__ = 'arenduchintala'
from sklearn.svm import SVC
import gensim
from math import sqrt
from utilz import tokenize
import sys
import numpy as np
import kenlm
import pdb


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
    lm = kenlm.LanguageModel('data/corpus.news.50.arpa')
    answers = [int(i) for i in open('eval-data/dev.answers', 'r').readlines()]
    training_data = [tuple(i.split('|||')) for i in open('eval-data/hyp1-hyp2-ref').readlines()]
    lsi = gensim.models.lsimodel.LsiModel.load(model_prefix + '.lsi')
    dictionary = gensim.corpora.Dictionary.load(model_prefix + '.dict')
    tfidf = gensim.models.tfidfmodel.TfidfModel.load(model_prefix + '.tfidf')
    X = []

    for idx, (hyp1_txt, hyp2_txt, ref_txt) in enumerate(training_data[:1000]):
        hyp1 = tokenize(hyp1_txt)
        hyp2 = tokenize(hyp2_txt)
        ref = tokenize(ref_txt)
        t_hyp1 = [h for h in hyp1 if h not in hyp2]
        t_hyp2 = [h for h in hyp2 if h not in hyp1]
        v_hyp1 = get_vec(t_hyp1)
        v_hyp2 = get_vec(t_hyp2)
        v_ref = get_vec(ref)
        cs1 = cosine_similarity(v_hyp1, v_ref)
        cs2 = cosine_similarity(v_hyp2, v_ref)
        csdiff = cs1 - cs2
        hyp1_lm = lm.score(hyp1_txt) / (len(hyp1) + 1)
        hyp2_lm = lm.score(hyp2_txt) / (len(hyp2) + 1)
        ref_lm = lm.score(ref_txt) / (len(ref) + 1)
        lm1 = (ref_lm / hyp1_lm)
        lm2 = (ref_lm / hyp2_lm)
        lmdiff = lm1 - lm2
        if lmdiff == 0:
            pdb.set_trace()
        print [csdiff, lmdiff], answers[idx]
        X.append([csdiff, lmdiff])
        #eprint idx, 'hyp1:', "%.4f" % cs1, 'hyp2:', "%.4f" % cs2, 'diff:', "%.4f" % abs(cs1 - cs2), 'ans:', answers[idx]

    clf = SVC()
    clf.kernel = 'poly'
    clf.fit(np.array(X), np.array(answers[:1000]))
