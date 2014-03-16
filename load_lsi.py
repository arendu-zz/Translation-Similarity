__author__ = 'arenduchintala'

import gensim
import sys

text_prefix = sys.argv[1]
lsi = gensim.models.lsimodel.LsiModel.load(text_prefix + '.lsi')
dict = gensim.corpora.Dictionary.load(text_prefix + '.dict')
vec_lsi = lsi[dict.doc2bow("a man and".lower().split())]
print vec_lsi