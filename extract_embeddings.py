import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from utils.preprocessing import load_pkl, save_word2vec_format
from config import *
from modeling import Visual_Grounding_Model



#load the textual embeddings
print('loading the model')
embedings_matrix=load_pkl(EMBEDDINGS_DIR + '/embed.pkl')

#load the the trained alignment from model's weight
model = Visual_Grounding_Model()
model = model.build_model()

print('computing grounded embeddings')
model.load_weights(SAVING_DIR + '/model')
alignment = model.get_layer('text_map').get_weights()
keys = list(embedings_matrix.keys())
values = np.stack(np.asarray(list(embedings_matrix.values())), axis=0)
values = np.matmul(values, alignment[0])

grounded_embeddings = dict(zip(keys, values))


fname = os.path.join(SAVING_DIR,'grounded_embeddings')
save_word2vec_format(fname, grounded_embeddings, ALIGNMENT_DIM, binary=True)

print('grounded embeddings were created successfully')
