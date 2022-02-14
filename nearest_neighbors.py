import gensim
from config import *

model = gensim.models.KeyedVectors.load_word2vec_format(SAVING_DIR + '/grounded_embeddings/ZSG_GloVe_{}d'.format(ALIGNMENT_DIM), binary=True, unicode_errors='ignore')

queries=['stupid','together','afraid','pretend','people','cliff','smart']
for word in queries:
    print("*******************************\n")
    print(word)
    NN=model.most_similar(word,topn=10)
    print(NN)
    print("*******************************\n")
