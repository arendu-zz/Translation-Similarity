__author__ = 'arenduchintala'
import gensim
import sys

text_file = sys.argv[1]
print 'making texts...'
texts = []
for doc in open(text_file, 'r'):
    texts.append(doc.split())

#texts = [[w for w in document.split()] for document in open('data/corpus', 'r').readlines()]
print 'making dictionary...'
dictionary = gensim.corpora.Dictionary(texts)
dictionary.save(text_file + '.dict')
print 'making corpus...'
corpus = []
for text in texts:
    corpus.append(dictionary.doc2bow(text))
gensim.corpora.MmCorpus.serialize(text_file + '.mm', corpus)
mm = gensim.corpora.MmCorpus(text_file + '.mm')
tfidf = gensim.models.tfidfmodel.TfidfModel(corpus)
tfidf.save(text_file + '.tfidf')
print 'making lsi...'
lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=dictionary, num_topics=5)
lsi.save(text_file + '.lsi')
print tfidf[dictionary.doc2bow("a man and".lower().split())]
print dictionary.doc2bow("a man and".lower().split())
print 'lsi from dict', lsi[dictionary.doc2bow("a man and".lower().split())]
print 'lsi from tfidf', lsi[tfidf[dictionary.doc2bow("a man and".lower().split())]]
#print vec_lsi
#vec_lsi = lsi[dictionary.doc2bow("man a and".lower().split())]
#print vec_lsi
#lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=4, update_every=0, passes=20)
#vec_lda = lda[dictionary.doc2bow("a man and".lower().split())]
# print vec_lda
#vec_lda = lda[dictionary.doc2bow("man a and".lower().split())]
# print vec_lda
