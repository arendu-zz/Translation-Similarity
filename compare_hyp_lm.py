__author__ = 'arenduchintala'
import kenlm
import utilz
import pdb

lm = kenlm.LanguageModel('data/corpus.news.50.arpa')
answers = [int(i) for i in open('eval-data/dev.answers', 'r').readlines()]
training_data = [tuple(i.split('|||')) for i in open('eval-data/hyp1-hyp2-ref').readlines()]
correct = 0
incorrect = 0
for idx, (hyp1, hyp2, ref) in enumerate(training_data[:len(answers)]):
    if answers[idx] != 0:
        hyp1_lm = lm.score(hyp1)  #/ (len(utilz.tokenize(hyp1)) + 1)
        hyp2_lm = lm.score(hyp2)  #/ (len(utilz.tokenize(hyp2)) + 1)
        ref_lm = lm.score(ref)  #/ (len(utilz.tokenize(ref)) + 1)
        guess = 1 if hyp1_lm > hyp2_lm else -1
        incorrect += 1 if guess != answers[idx] and answers[idx] != 0 else 0
        correct += 1 if guess == answers[idx] and answers[idx] != 0 else 0
        if guess != answers[idx]:
            pass
            #print hyp1_lm, hyp1
            #print hyp2_lm, hyp2
            #print ref_lm, ref
            #pdb.set_trace()
print 'total check', incorrect + correct
print 'incorrect', incorrect, float(incorrect) / float(incorrect + correct)
print 'correct', correct, float(correct) / float(incorrect + correct)

