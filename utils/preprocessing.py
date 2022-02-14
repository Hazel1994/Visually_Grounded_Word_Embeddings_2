import tensorflow as tf

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
from smart_open import smart_open
import gensim
import json
from sklearn.utils import shuffle
from utils.data_utils import *


def save_word2vec_format(fname, vocab, vector_size, binary=True):
    """Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vector_size : int
        The number of dimensions of word vectors.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.


    """

    os.makedirs(fname, exist_ok=True)
    fname=os.path.join(fname,'ZSG_GloVe_'+str(ALIGNMENT_DIM) + 'd')
    total_vec = len(vocab)
    counter=0
    with smart_open(fname, 'wb') as fout:
        print(total_vec, vector_size)
        fout.write(gensim.utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            counter +=1
            if binary:
                row = row.astype(np.float32)
                fout.write(gensim.utils.to_utf8(''.join(word)) + b" " + row.tostring())
            else:
                fout.write(gensim.utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))
            if counter %500000==0:
                print(counter)

def save_glove_as_pickle():
    """
    Load glove vectors from a .txt file.
    """
    print("Preparing the GloVe embeddings, it might take a few minutes")
    vocab = {}
    current_idx = 0
    with open(EMBEDDINGS_DIR + '/glove.840B.300d.txt', "r", encoding="utf-8") as f:
        for _, line in enumerate(f):
            tokens = line.split()
            word = tokens[0]
            v=tokens[1:]
            if len(v)==300:
                vector = np.array(v, dtype=np.float64)
                if word not in vocab:
                    vocab[word] = vector
                    current_idx += 1
        print('Found {} number of words in the embeddings'.format(current_idx))

        save_pkl(vocab,os.path.join(EMBEDDINGS_DIR,'embed'))
        print("GloVe embeddings are saved as a pickle file at " + os.path.join(EMBEDDINGS_DIR,'embed'))

def init_inceptionv3():
    # load model
    model = InceptionV3()

    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return  model

def pre_preprocessing_coco(annotations, train=True):
    """
    input:
        get the annotations of train or validation (train=false)
    ----------
    Returns:
        tokenizer trained on training captions,
        a dict mapping from image files to their cnn features
        a dict with two list: 1 image files, 2 captions ( each caption is aligned with its file,
        since there are 5 captions per image, each image file is repeated 5 times)

    """
    all_captions = []
    all_img_file_names = []
    model_incv3 = init_inceptionv3()
    image_features = []
    image_batch = []
    train_or_val = "train_images" if train else "val_images"

    #get the caption along with their corresponing image file name
    for annot in annotations['annotations']:
        caption = 'sss ' + annot['caption'] + ' eee'
        image_id = annot['image_id']
        image_file_name = '%012d.jpg' % (image_id)
        all_img_file_names.append(image_file_name)
        all_captions.append(caption)

    #get the inique names of image files ( since we have 5 captions per image)
    unique_file_names = np.unique(all_img_file_names)

    # pretrained cnn features for all the images
    print("extracting image features using pretrained inception v3")
    for c, f_name in enumerate(unique_file_names):
        image = load_img(os.path.join(BASE_DATASET_DIR, train_or_val, f_name), target_size=(299, 299))
        image_batch.append(image)
        if (c + 1) % PRE_PROCESSING_BATCH_SIZE == 0:
            print("{} out of {} ".format( (c + 1), len(unique_file_names) ) )

            img_arrays = np.stack([img_to_array(i) for i in image_batch], axis=0)
            features = model_incv3.predict(preprocess_input(img_arrays))
            image_features.extend(features.tolist())
            image_batch = []

    if len(image_batch) > 0:
        img_arrays = np.stack([img_to_array(i) for i in image_batch], axis=0)
        features = model_incv3.predict(preprocess_input(img_arrays))
        image_features.extend(features.tolist())


    file_name_to_cnn_features = dict(zip(unique_file_names, image_features))

    if train:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True)
        tokenizer.fit_on_texts(all_captions)
        save_pkl(tokenizer, 'resources/tokenizer')
        print("tokenizer saved successfully")

    # Shuffle captions and image_file_names together
    train_captions, all_img_file_names = shuffle(all_captions,
                                                 all_img_file_names,
                                                 random_state=1)

    name_caption = {'names': all_img_file_names, 'captions': train_captions}

    #name_caption: two list containing all the file names and captions
    #file_name_features: a dict from file name to image features
    return name_caption, file_name_to_cnn_features

def prepare_coco():

    print("preparing training data")

    dataType = 'train2017'
    annotation_file = '{}/annotations/captions_{}.json'.format(BASE_DATASET_DIR, dataType)
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    name_caption, file_name_to_cnn_features= pre_preprocessing_coco(annotations)

    save_pkl(name_caption,os.path.join(RESOURCES,'train_files_to_captions'))
    save_pkl(file_name_to_cnn_features,os.path.join(RESOURCES,'train_files_to_cnn_features'))

    print("preparing validation data")
    dataType = 'val2017'
    annotation_file = '{}/annotations/captions_{}.json'.format(BASE_DATASET_DIR, dataType)
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    name_caption, file_name_to_cnn_features = pre_preprocessing_coco(annotations,train=False)

    save_pkl(name_caption, os.path.join(RESOURCES,'val_files_to_captions'))
    save_pkl(file_name_to_cnn_features, os.path.join(RESOURCES,'val_files_to_cnn_features'))

    print("done preparing the coco dataset")

def setups():
    if not os.path.exists(os.path.join(RESOURCES,'val_files_to_cnn_features.pkl')):
        prepare_coco()
    if not os.path.exists(os.path.join(EMBEDDINGS_DIR,'embed.pkl')):
        save_glove_as_pickle()