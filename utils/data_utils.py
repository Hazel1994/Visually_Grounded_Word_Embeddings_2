from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
import pickle
from config import *


class Data_Generator(Sequence):

    def __init__(self,img_features,name_tokens):

        self.batch_size = TRAINING_BATCH_SIZE
        self.img_features = img_features
        self.name_tokens=name_tokens

    def __len__(self):
        return int(np.ceil(len(self.name_tokens['names']) / float(self.batch_size)))

    def __getitem__(self, idx):


        batch_x = self.name_tokens['names'][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.name_tokens['tokens'][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_z = [self.img_features[i] for i in batch_x]

        # Count the number of tokens in all these token-sequences.
        num_tokens = [len(t) for t in batch_y]

        # Max number of tokens. using less lenghty input happens to produce slightly better results so why not
        max_tokens = np.max(num_tokens) - 1

        # Pad all the other token-sequences with zeros
        # so they all have the same length
        tokens_padded = pad_sequences(batch_y,
                                      maxlen=max_tokens,
                                      padding='post',
                                      truncating='post')

        # ensure that the data is assigned correctly.
        x_data = \
            {
            'caption': tokens_padded,
            }

        y_data = \
            {
            'lstm_out': np.asarray(batch_z),
            }

        return (x_data, y_data)

def load_pkl(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

def save_pkl(obj, filename):

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def get_embed_matrix(embed,tokenizer):
    """

    Parameters
    ----------
        the glove embedding as dict
        the trained tokenizer

    Returns
    -------
    a nd arrary containing the glove word vectors based on the
    tokenizer's vocab.
    """
    c=0
    d=len(embed[list(embed.keys())[0]])
    embedding_matrix = np.zeros((MAX_VOCAB, d))
    #tokenizer encodes the words starting from 1 since index zero might be used for masking
    #in the embedding layer.
    for word, i in list(tokenizer.word_index.items())[0:MAX_VOCAB-1]:
        if word in embed:
            embedding_matrix[i] = embed[word]
        else:
            c=c+1
            embedding_matrix[i] = embed['unknown']
    print(c, ' words were not found in Glove')

    return embedding_matrix
