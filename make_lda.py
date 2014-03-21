__author__ = 'arenduchintala'
import gensim
import sys, pdb
from utilz import tokenize, STOP_WORDS, VEC_LEN


save_model = sys.argv[1]
corpus_files = sys.argv[2:]
print 'making texts...'
texts = []
for cs in corpus_files:
    print 'reading', cs
    for doc in open(cs, 'r'):
        texts.append(set([i for i in tokenize(doc.decode('utf-8', 'ignore')) if i not in STOP_WORDS]))

print 'making dictionary...'
dictionary = gensim.corpora.Dictionary(texts)
dictionary.save(save_model + '.dict')
print 'making corpus...'
corpus = []
for text in texts:
    corpus.append(dictionary.doc2bow(text))
gensim.corpora.MmCorpus.serialize(save_model + '.mm', corpus)
mm = gensim.corpora.MmCorpus(save_model + '.mm')
tfidf = gensim.models.tfidfmodel.TfidfModel(corpus)
tfidf.save(save_model + '.tfidf')
print 'making lda...'
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=VEC_LEN, update_every=1, chunksize=5000, passes=1)
lda.save(save_model + '.lda')
#lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=dictionary, num_topics=VEC_LEN)
#lsi.save(save_model + '.lsi')
print 'tfidf values', tfidf[dictionary.doc2bow("a man great child and".lower().split())]
print 'dict values', dictionary.doc2bow("a man great child and".lower().split())
#print 'lsi from dict', lsi[dictionary.doc2bow("a man great child and".lower().split())]
#print 'lsi from tfidf', lsi[tfidf[dictionary.doc2bow("a man great child and".lower().split())]]
#print vec_lsi
#vec_lsi = lsi[dictionary.doc2bow("man a and".lower().split())]
#print vec_lsi
#lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=4, update_every=0, passes=20)
#vec_lda = lda[dictionary.doc2bow("a man and".lower().split())]
# print vec_lda
vec_lda = lda[dictionary.doc2bow("man a and".lower().split())]
print vec_lda
