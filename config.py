"""
BASE_DATASET_DIR:
    where you download the coco_2017 dataset this directory is expected to have the 3 directories: 1. train_images 2. val_images,  containing
    the training and validation images respectively. 3 annotations: including captions_train2017.json and captions_val2017.json

PRE_PROCESSING_BATCH_SIZE
    this batch size is used when fetching raw images for the pre-trained cnn model.
    lower this if you dont have enough memory to extract the features in the preprocessing step

TRAINING_BATCH_SIZE
    used for training the model with caption-image pairs

MAX_VOCAB
    maximum number of vocabulary to be used

lEARNING_RATE
    learning rate for training

EMBEDDINGS_DIR
    the directory under which glove embedding exist as a .txt file

EMBEDDINGS_DIM
    the dimension of textual GloVe

ALIGNMENT_DIM
    the dimension of grounded embeddings (same as the alignment dimenstion)

RESOURCES
    the directory under which the prepared dataset ( images features and ...) will be saved

SAVING_DIR
    where to save the model weight and generate grounded embeddings

EPOCHS
    number of epochs to run

EARLY_STOPPING_PATIENCE
    stop training if validation loss does not get better after this(EARLY_STOPPING_PATIENCE) number of times
"""

BASE_DATASET_DIR = '/graphics/scratch/shahmoha/coco_2017'
RESOURCES='resources'
SAVING_DIR='/graphics/scratch/shahmoha/checkpoints/final models/glove'
EMBEDDINGS_DIR='GloVe_840b_300d'

PRE_PROCESSING_BATCH_SIZE=2048
EMBEDDINGS_DIM=300
ALIGNMENT_DIM=300
MAX_VOCAB=10000

EPOCHS=20
EARLY_STOPPING_PATIENCE=5
TRAINING_BATCH_SIZE=256
lEARNING_RATE=0.001
