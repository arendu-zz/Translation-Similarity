__author__ = 'arenduchintala'
import kenlm
import utilz
import pdb

lm = kenlm.LanguageModel('data/corpus.news.50.arpa')
answers = [int(i) for i in open('eval-data/dev.answers', 'r').readlines()]
training_data = [tuple(i.split('|||')) for i in open('eval-data/hyp1-hyp2-ref').readlines()]

for idx, (hyp1, hyp2, ref) in enumerate(training_data[:len(answers)]):
    hyp1_lm = lm.score(hyp1) / (len(utilz.tokenize(hyp1)) + 1)
    hyp2_lm = lm.score(hyp2) / (len(utilz.tokenize(hyp2)) + 1)
    ref_lm = lm.score(ref) / (len(utilz.tokenize(ref)) + 1)

    print "%.4f" % (ref_lm / hyp1_lm), "%.4f" % (ref_lm / hyp2_lm), "%.4f" % hyp1_lm, "%.4f" % hyp2_lm, "%.4f" % ref_lm, answers[idx]
    if (ref_lm / hyp1_lm) > 1 or (ref_lm / hyp2_lm) > 1:
        print hyp1.decode('ascii'), utilz.tokenize(hyp1)
        print hyp2, utilz.tokenize(hyp2)
        print ref, utilz.tokenize(ref)
        exit()