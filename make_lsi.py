__author__ = 'arenduchintala'
import gensim
import sys
import nltk

save_model = sys.argv[1]
corpus_files = sys.argv[2:]
print 'making texts...'
texts = []
for cs in corpus_files:
    print 'reading', cs
    for doc in open(cs, 'r'):
        texts.append(nltk.word_tokenize(doc.lower().decode('utf-8', 'ignore')))
        #texts.append(doc.lower().split())

#texts = [[w for w in document.split()] for document in open('data/corpus', 'r').readlines()]
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
print 'making lsi...'
lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=dictionary, num_topics=300)
lsi.save(save_model + '.lsi')
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
