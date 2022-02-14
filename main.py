import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils.preprocessing import *
from config import *
import tensorflow as tf
from modeling import Visual_Grounding_Model

# prepare the dataset and tokenizer if not done already
setups()


tokenizer=load_pkl(os.path.join(RESOURCES,'tokenizer.pkl'))
tokenizer.num_words=MAX_VOCAB

print("loading the embedding file")
embeddings=load_pkl(EMBEDDINGS_DIR + '/embed.pkl')
embedding_matrix = get_embed_matrix(embeddings,tokenizer)
#free memory
del embeddings


train_captions=load_pkl(os.path.join(RESOURCES,'train_files_to_captions.pkl'))
valid_captions=load_pkl(os.path.join(RESOURCES,'val_files_to_captions.pkl'))

train_img_features=load_pkl(os.path.join(RESOURCES,'train_files_to_cnn_features.pkl'))
valid_img_features=load_pkl(os.path.join(RESOURCES,'val_files_to_cnn_features.pkl'))

print("running the tokenizer on the captions")
train_captions['tokens'] = tokenizer.texts_to_sequences(train_captions['captions'])
valid_captions['tokens'] = tokenizer.texts_to_sequences(valid_captions['captions'])


len_train=len(train_captions['names'])
len_valid=len(valid_captions['names'])

steps_per_epoch = int(len_train / TRAINING_BATCH_SIZE)
validation_steps = int(len_valid/ TRAINING_BATCH_SIZE)

train_gen=Data_Generator(train_img_features,train_captions)
valid_gen=Data_Generator(valid_img_features,valid_captions)

model = Visual_Grounding_Model()
model = model.build_model()

#set the embedding weights
model.get_layer('embedding').set_weights([embedding_matrix])

filepath = os.path.join(SAVING_DIR,'model')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, save_weights_only=True)

model.compile(tf.keras.optimizers.Nadam(learning_rate=lEARNING_RATE), loss=[tf.keras.losses.MeanSquaredError()])

history=model.fit(
    x=train_gen, epochs=EPOCHS, verbose=1, callbacks=[model_checkpoint,early_stopping],
    validation_data=valid_gen,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,max_queue_size=20,workers=16
)