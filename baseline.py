__author__ = 'arenduchintala'
from utilz import tokenize
import os


def l(h, ref):
    return sum(1 for w in h if w in ref)


def R(h, ref):
    hnr = len(h.intersection(ref))
    r = len(ref)
    return float(hnr) / r


def P(h, ref):
    hnr = len(h.intersection(ref))
    h = len(h)
    return float(hnr) / h


def meteor(h, ref, alpha):
    prec = P(h, ref)
    recall = R(h, ref)
    if recall == 0 and prec == 0:
        return 0.0
    return prec * recall / (((1 - alpha) * recall) + (alpha * prec))


def main(training_data, alpha):
    ans = []
    for idx, (hyp1, hyp2, ref) in enumerate(training_data):
        hyp1 = set(tokenize(hyp1))
        hyp2 = set(tokenize(hyp2))
        ref = set(tokenize(ref))
        m1 = meteor(hyp1, ref, alpha)
        m2 = meteor(hyp2, ref, alpha)
        if m1 > m2:
            ans.append('1')
        elif m2 > m1:
            ans.append('-1')
        else:
            ans.append('0')
    return '\n'.join(ans)


if __name__ == '__main__':

    training_data = [tuple([k.strip() for k in i.split('|||')]) for i in open('eval-data/hyp1-hyp2-ref').readlines()]
    for a in [x * 0.1 for x in range(0, 10)]:
        ans_file = 'baseline' + str("%.2f" % a)
        writer = open(ans_file, 'w')
        ans = main(training_data, a)
        writer.write(ans)
        writer.flush()
        writer.close()
        print a, ans_file
        os.system('python compare-with-human-evaluation <' + ans_file)





