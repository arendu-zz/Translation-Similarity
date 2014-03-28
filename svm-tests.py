__author__ = 'arenduchintala'
from sklearn.svm import NuSVC, SVC
from sklearn import cross_validation
import gensim
from math import sqrt, log
from utilz import tokenize, STOP_WORDS, VEC_LEN
import sys
import numpy as np
import kenlm
import baseline


vec_rep_model = None
dictionary = None
tfidf = None


def kl_divergence(v1, v2):
    '''
    expects v1 and v2 in the form [(0, 0.23422), (1, 0.982343), (2, 0.00012123)...]
    computes distance of v1 from v2
    '''
    vec1 = [i[1] for i in v1]
    vec2 = [i[1] for i in v2]
    kl_vec = [i2 * log(i2 / i1) if i2 > 0 else 0.0 for i1, i2 in zip(vec1, vec2)]
    return sum(kl_vec)


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


def get_vec(hyp, tf=False):
    hyp = [h for h in hyp if h not in STOP_WORDS]
    if tf:
        vec = vec_rep_model[tfidf[dictionary.doc2bow(hyp)]]
    else:
        vec = vec_rep_model[dictionary.doc2bow(hyp)]
    if len(vec) < VEC_LEN:
        vec = [0.0] * VEC_LEN
        vec = zip(range(VEC_LEN), vec)
    return vec


def get_array_n_hot(vec):
    n = int(len(vec) / 2)
    [idx, arr] = zip(*vec)
    median = sorted(arr)[n]
    m_arr = max(arr)
    min_arr = min(arr)
    if m_arr > min_arr:
        return [1 if a > median else 0 for a in arr]
    else:
        return [0.0] * len(arr)


def get_array(vec, normalize=False):
    [idx, arr] = zip(*vec)
    if normalize:
        min_arr = min(arr)
        r = max(arr) - min(arr)
        if r == 0:
            return [float(a) for a in arr]
        else:
            return [(a - min_arr) / r for a in arr]
    else:
        return [float(a) for a in arr]


def get_meteor_vec(h1, h2, ref, alpha):
    m1 = baseline.meteor(set(h1), set(ref), alpha)
    m2 = baseline.meteor(set(h2), set(ref), alpha)
    if m1 > m2:
        return [1, 0, 0]
    elif m2 > m1:
        return [0, 1, 0]
    else:
        return [0, 0, 1]


def get_cs_vec(h1, h2, ref):
    t_hyp1 = [h for h in h1 if h not in h2]
    t_hyp2 = [h for h in h2 if h not in h1]
    v_hyp1 = get_vec(t_hyp1, use_tfifd)
    v_hyp2 = get_vec(t_hyp2, use_tfifd)
    v_ref = get_vec(ref, use_tfifd)
    cs1 = cosine_similarity(v_hyp1, v_ref)
    cs2 = cosine_similarity(v_hyp2, v_ref)
    csdiff = abs(cs1 - cs2)
    lsi_vec = []
    #lsi_vec += get_array(v_hyp1, normalize=True) + get_array(v_hyp2, normalize=True) + get_array(v_ref, normalize=True)
    nh1 = get_array_n_hot(v_hyp1)
    nh2 = get_array_n_hot(v_hyp2)
    nref = get_array_n_hot(v_ref)
    lsi_vec += [x * y for x, y in zip(nh1, nref)] + [x * y for x, y in zip(nh2, nref)]
    if cs1 > cs2:
        return [1, 0, 0, cs1, cs2, csdiff] + lsi_vec
    elif cs2 > cs1:
        return [0, 1, 0, cs1, cs2, csdiff] + lsi_vec
    else:
        return [0, 0, 1, cs1, cs2, csdiff] + lsi_vec


def get_lm_vec(h1, h2, ref):
    hyp1_lm = lm.score(' '.join(h1)) / (len(h1) + 1)
    hyp2_lm = lm.score(' '.join(h2)) / (len(h2) + 1)
    ref_lm = lm.score(' '.join(ref)) / (len(ref) + 1)
    lm1 = (ref_lm / hyp1_lm)
    lm2 = (ref_lm / hyp2_lm)
    lmdiff = abs(lm1 - lm2)
    if lm1 > lm2:
        return [1, 0, 0, lm1, lm2, lmdiff]
    elif lm2 > lm1:
        return [0, 1, 0, lm1, lm2, lmdiff]
    else:
        return [0, 0, 1, lm1, lm2, lmdiff]


if __name__ == '__main__':

    model_prefix = sys.argv[1]
    model_type = sys.argv[2]  # 'lsi'  # 'lda'
    use_tfifd = sys.argv[3].strip() == 'True'
    print 'MODEL_PREFIX:', model_prefix, 'MODEL_TYPE:', model_type, 'TFIDF:', use_tfifd
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
    st = 0
    sp =len(answers)
    X = []
    for idx, (hyp1_txt, hyp2_txt, ref_txt) in enumerate(training_data[:len(answers[st:sp])]):
        hyp1 = tokenize(hyp1_txt)
        hyp2 = tokenize(hyp2_txt)
        ref = tokenize(ref_txt)
        m_vec = get_meteor_vec(hyp1, hyp2, ref, 0.8)
        if model_type == 'lsi':
            cs_vec = get_cs_vec(hyp1, hyp2, ref)
        elif model_type == 'lda':
            cs_vec = get_cs_vec(hyp1, hyp2, ref)
            #cs1 = kl_divergence(v_hyp1, v_ref)
            #cs2 = kl_divergence(v_hyp2, v_ref)
            pass
        #csdiff = cs1 - cs2
        #v_diff = [r - h for r, h in zip()]
        lm_vec = get_lm_vec(hyp1, hyp2, ref)
        train_sample = []
        train_sample += cs_vec
        train_sample += lm_vec
        train_sample += m_vec
        X.append(train_sample)

    print 'training samples:', np.shape(np.array(X)), np.shape(np.array(answers[st:sp])), np.shape(np.array(sample_weights[st:sp]))
    print 'predicting...'
    P = []
    for idx, (hyp1_txt, hyp2_txt, ref_txt) in enumerate(training_data[sp:]):
        hyp1 = tokenize(hyp1_txt)
        hyp2 = tokenize(hyp2_txt)
        ref = tokenize(ref_txt)
        m_vec = get_meteor_vec(hyp1, hyp2, ref, 0.8)
        if model_type == 'lsi':
            cs_vec = get_cs_vec(hyp1, hyp2, ref)
        elif model_type == 'lda':
            cs_vec = get_cs_vec(hyp1, hyp2, ref)
            #cs1 = kl_divergence(v_hyp1, v_ref)
            #cs2 = kl_divergence(v_hyp2, v_ref)
            pass
        lm_vec = get_lm_vec(hyp1, hyp2, ref)
        train_sample = []
        train_sample += cs_vec
        train_sample += lm_vec
        train_sample += m_vec
        P.append(train_sample)
    print 'test samples', np.shape(np.array(P))
    print 'training classifiers...'
    for k in ['rbf', 'linear']:
        simpleclf = SVC(kernel=k, cache_size=4000)
        simpleclf.fit(np.array(X), np.array(answers[st:sp]), sample_weight=np.array(sample_weights[st:sp]))
        scores = cross_validation.cross_val_score(simpleclf, np.array(X), np.array(answers[st:sp]))
        print 'SVC:', simpleclf.kernel, 'CV(3)',sum(scores) / len(scores)
        preditions = simpleclf.predict(np.array(P))
        np.savetxt('SVC-' + simpleclf.kernel + '.3.pred', preditions, fmt='%d')

        nuclf = NuSVC(kernel=k, cache_size=4000)
        nuclf.fit(np.array(X), np.array(answers[st:sp]), sample_weight=np.array(sample_weights[st:sp]))
        scores = cross_validation.cross_val_score(nuclf, np.array(X), np.array(answers[st:sp]))
        print 'NuSVC:', nuclf.kernel, 'CV(3)',sum(scores) / len(scores)
        preditions = nuclf.predict(np.array(P))
        np.savetxt('NuSVC-' + nuclf.kernel + '.3.pred', preditions, fmt='%d')

