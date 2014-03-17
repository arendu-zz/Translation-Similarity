__author__ = 'arenduchintala'
import nltk
import string


def tokenize(s):
    #s = "string. With. Punctuation?" # Sample string
    out = ''.join([i for i in s if i not in string.punctuation])
    return nltk.word_tokenize(out.lower())