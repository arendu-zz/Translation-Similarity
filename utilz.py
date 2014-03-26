__author__ = 'arenduchintala'
import nltk
import string

STOP_WORDS = "a an and are as at be by for from has he in is it its of on that the to was were will with".split()
VEC_LEN = 100


def tokenize(s):
    #s = "string. With. Punctuation?" # Sample string
    #out = ''.join([i for i in s if (i not in string.punctuation) and (i in string.printable)])
    s = s.replace('&quot;', '')
    out = ''.join([i for i in s if i in string.printable])
    return nltk.word_tokenize(out.lower())
