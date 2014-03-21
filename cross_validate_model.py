__author__ = 'arenduchintala'
from sklearn.svm import SVC
from sklearn import cross_validation
import gensim
from math import sqrt
from utilz import tokenize, STOP_WORDS, VEC_LEN
import sys
import numpy as np
import kenlm


vec_rep_model = None
dictionary = None
tfidf = None


def cosine_similarity(v1, v2):
    '''
    expects v1 and v2 in the form [(0, 0.23422), (1, 0.982343), (2, 0.00012123)...]
    '''
    vec1 = [i[1] for i in v1]
    vec2 = [i[1] for i in v2]
    mag1 = sqrt(sum([i ** 2 for i in vec1]))
    mag2 = sqrt(sum([i ** 2 for i in vec2]))
    if mag2 == 0 or mag1 == 0:
        return 0.0
    dotsum = sum([x * y for x, y in zip(vec1, vec2)])
    return float(dotsum) / float(mag1 * mag2)


def get_vec(hyp):
    hyp = [h for h in hyp if h not in STOP_WORDS]
    vec = vec_rep_model[dictionary.doc2bow(hyp)]
    if len(vec) < VEC_LEN:
        vec = [0.0] * VEC_LEN
        vec = zip(range(VEC_LEN), vec)
    return vec


def get_array(vec):
    [idx, arr] = zip(*vec)
    return [float(a) for a in arr]


if __name__ == '__main__':
    st = 5000
    sp = 10000
    model_prefix = sys.argv[1]
    model_type = sys.argv[2]  # 'lsi'  # 'lda'
    lm = kenlm.LanguageModel('data/news.small.20.tok.arpa')
    answers = [int(i) for i in open('eval-data/dev.answers', 'r').readlines()]
    weights = {}
    for label in set(answers):
        weights[label] = float(len(answers) - answers.count(label)) / len(answers)
    for label in weights:
        weights[label] *= (1.0 / min(weights.values()))

    sample_weights = [weights[i] for i in answers]
    training_data = [tuple([k.strip() for k in i.split('|||')]) for i in open('eval-data/hyp1-hyp2-ref').readlines()]
    if model_type == 'lsi':
        vec_rep_model = gensim.models.lsimodel.LsiModel.load(model_prefix + '.lsi')
    elif model_type == 'lda':
        vec_rep_model = gensim.models.LdaModel.load(model_prefix + '.lda')
    dictionary = gensim.corpora.Dictionary.load(model_prefix + '.dict')
    tfidf = gensim.models.tfidfmodel.TfidfModel.load(model_prefix + '.tfidf')
    X = []
    for idx, (hyp1_txt, hyp2_txt, ref_txt) in enumerate(training_data[:len(answers[st:sp])]):
        hyp1 = tokenize(hyp1_txt)
        hyp2 = tokenize(hyp2_txt)
        ref = tokenize(ref_txt)
        t_hyp1 = [h for h in hyp1 if h not in hyp2]
        t_hyp2 = [h for h in hyp2 if h not in hyp1]
        v_hyp1 = get_vec(hyp1)
        v_hyp2 = get_vec(hyp2)
        v_ref = get_vec(ref)
        cs1 = cosine_similarity(v_hyp1, v_ref)
        cs2 = cosine_similarity(v_hyp2, v_ref)
        csdiff = cs1 - cs2
        v_diff = [r - h for r, h in zip()]
        hyp1_lm = lm.score(' '.join(hyp1)) / (len(hyp1) + 1)
        hyp2_lm = lm.score(' '.join(hyp2)) / (len(hyp2) + 1)
        ref_lm = lm.score(' '.join(ref)) / (len(ref) + 1)
        lm1 = (ref_lm / hyp1_lm)
        lm2 = (ref_lm / hyp2_lm)
        lmdiff = lm1 - lm2
        if lmdiff == 0 or csdiff == 0:
            pass  # pdb.set_trace()
        #print idx, [csdiff, lmdiff], answers[idx], sample_weights[idx]

        train_sample = [cs1, cs2, csdiff, lm1, lm2, lmdiff] + get_array(v_hyp1) + get_array(v_hyp2) + get_array(v_ref)
        X.append(train_sample)

    clf = SVC(kernel='linear')
    print np.shape(np.array(X)), np.shape(np.array(answers[st:sp])), np.shape(np.array(sample_weights[st:sp]))
    clf.fit(np.array(X), np.array(answers[st:sp]), sample_weight=np.array(sample_weights[st:sp]))

    Z = clf.score(np.array(X), np.array(answers[st:sp]))
    print Z
    scores = cross_validation.cross_val_score(clf, np.array(X), np.array(answers[st:sp]), cv=5)
    print scores, sum(scores) / len(scores)