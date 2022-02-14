from tensorflow.keras.layers import Input,Embedding,Dense,LSTM
from tensorflow.keras import Model
from config import *

class Visual_Grounding_Model():

    def __init__(self):
        self.num_words = MAX_VOCAB
        self.embedding_size = EMBEDDINGS_DIM
        self.alignment_size =ALIGNMENT_DIM
        self.image_vector_size=2048

    def build_model(self):

        caption = Input(shape=(None,), name='caption')

        embedding = Embedding(input_dim=self.num_words,
                              output_dim=self.embedding_size,
                              name='embedding', trainable=False, mask_zero=True)

        alignmentM = Dense(self.alignment_size, activation='linear', name='text_map', use_bias=False)

        LSTM_layer= LSTM(self.image_vector_size, return_sequences=False, use_bias=True, name='lstm_out')

        # embeddings for tokens
        tokens = embedding(caption)
        # mapping to grounded space
        vokens = alignmentM(tokens)
        # fuse perceptual knowledge with embeddings
        predicted_image = LSTM_layer(vokens)
        # build the model
        model = Model(inputs=[caption], outputs=[predicted_image])
        model.summary()
        return model