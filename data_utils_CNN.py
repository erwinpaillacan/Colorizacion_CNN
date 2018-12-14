import numpy as np
import os
import glob

from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave, imread

import tensorflow as tf
from tensorflow import data as tfdata
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

sess = None



dir=os.getcwd()
# change these to your local paths
TRAIN_IMAGE_PATH = dir+'/train'
VALIDATION_IMAGE_PATH = dir+'/validation'

IMAGE_SIZE = 256  # Global constant image size
EMBEDDING_IMAGE_SIZE = 224  # Global constant embedding size

TRAIN_RECORDS_PATH = "data/images.tfrecord"  # local path to tf record directory
VAL_RECORDS_PATH = "data/val_images.tfrecord"  # local path to tf record directory


if not os.path.exists('weights/'):
    os.makedirs('weights/')

if not os.path.exists('results/'):
    os.makedirs('results/')

if not os.path.exists('data/'):
    os.makedirs('data/')



def _float32_feature_list(floats):
    return tf.train.Feature(float_list=tf.train.FloatList(value=floats))


def _generate_records(images_path, tf_record_name, batch_size=100):
    '''
    Creates a TF Record containing the pre-processed image consisting of
    1)  L channel input
    2)  ab channels output
    3)  features extracted from MobileNet
    This step is crucial for speed during training, as the major bottleneck
    is the extraction of feature maps from MobileNet. It is slow, and inefficient.
    '''
    if os.path.exists(tf_record_name):
        print("****  Delete old TF Records first! ****")
        exit(0)

    files = glob.glob(images_path + "*/*.jpg")
    files = sorted(files)
    nb_files = len(files)

    # Use ZLIB compression to save space and create a TFRecordWriter
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(tf_record_name, options)

    size = max(EMBEDDING_IMAGE_SIZE, IMAGE_SIZE)  # keep larger size until stored in TF Record

    X_buffer = []
    for i, fn in enumerate(files):
        try:  # prevent crash due to corrupted imaged
            X = imread(fn)
            X = resize(X, (size, size, 3), mode='constant') # resize to the larger size for now
        except:
            continue

        X_buffer.append(X)

        if len(X_buffer) >= batch_size:
            X_buffer = np.array(X_buffer)
            _serialize_batch(X_buffer, writer, batch_size)  # serialize the image into the TF Record

            del X_buffer  # delete buffered images from memory
            X_buffer = []  # reset to new list

            print("Procesado %d / %d imagenes" % (i + 1, nb_files))

    if len(X_buffer) != 0:
        X_buffer = np.array(X_buffer)
        _serialize_batch(X_buffer, writer)  # serialize the remaining images in buffer

        del X_buffer  # delete buffer

    print("Procesado %d / %d imagenes" % (nb_files, nb_files))
    print("TF Record creado correctamente")

    writer.close()


def _serialize_batch(X, writer, batch_size=100):
    '''
    Processes a batch of images, and then serializes into the TFRecord
    Args:
        X: original image with no preprocessing
        writer: TFRecordWriter
        batch_size: batch size
    '''
    [X_batch], Y_batch = _process_batch(X, batch_size)  # preprocess batch

    for j, (img_l, y) in enumerate(zip(X_batch, Y_batch)):
        # resize the images to their smaller size to reduce space wastage in the record
        img_l = resize(img_l, (IMAGE_SIZE, IMAGE_SIZE, 1), mode='constant')
        y = resize(y, (IMAGE_SIZE, IMAGE_SIZE, 2), mode='constant')

        example_dict = {
            'image_l': _float32_feature_list(img_l.flatten()),
            'image_ab': _float32_feature_list(y.flatten())
        }
        example_feature = tf.train.Features(feature=example_dict)
        example = tf.train.Example(features=example_feature)
        writer.write(example.SerializeToString())


def _construct_dataset(record_path, batch_size, sess):
    def parse_record(serialized_example):
        # parse a single record
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_l': tf.FixedLenFeature([IMAGE_SIZE, IMAGE_SIZE, 1], tf.float32),
                'image_ab': tf.FixedLenFeature([IMAGE_SIZE, IMAGE_SIZE, 2], tf.float32)
            })

        l, ab= features['image_l'], features['image_ab']
        return l, ab

    dataset = tfdata.TFRecordDataset([record_path], 'ZLIB')  # create a Dataset to wrap the TFRecord
    dataset = dataset.map(parse_record, num_parallel_calls=2)  # parse the record
    dataset = dataset.repeat()  # repeat forever
    dataset = dataset.batch(batch_size)  # batch into the required batchsize
    dataset = dataset.shuffle(buffer_size=5)  # shuffle the batches
    iterator = dataset.make_initializable_iterator()  # get an iterator over the dataset

    sess.run(iterator.initializer)  # initialize the iterator
    next_batch = iterator.get_next()  # get the iterator Tensor

    return dataset, next_batch


def _process_batch(X, batchsize=100):
    '''
    Process a batch of images for training
    Args:
        X: a RGB image
    '''
    lab_batch = rgb2lab(X)  # convert to LAB colorspace
    X_batch = lab_batch[:, :, :, 0]  # extract L from LAB
    X_batch = X_batch.reshape(X_batch.shape + (1,))  # reshape into (batch, IMAGE_SIZE, IMAGE_SIZE, 1)
    X_batch = 2 * X_batch / 100 - 1.  # normalize the batch
    Y_batch = lab_batch[:, :, :, 1:] / 127  # extract AB from LAB

    return ([X_batch], Y_batch)


def generate_train_records(batch_size=100):
    _generate_records(TRAIN_IMAGE_PATH, TRAIN_RECORDS_PATH, batch_size)


def generate_validation_records(batch_size=100):
    _generate_records(VALIDATION_IMAGE_PATH, VAL_RECORDS_PATH, batch_size)


def train_generator(batch_size):
    '''
    Generator which wraps a tf.data.Dataset object to read in the
    TFRecord more conveniently.
    '''
    if not os.path.exists(TRAIN_RECORDS_PATH):
        print("\n\n", '*' * 50, "\n")
        print("Please create the TFRecord of this dataset by running `data_utils_CNN.py` script")
        exit(0)

    with tf.Session() as train_gen_session:
        dataset, next_batch = _construct_dataset(TRAIN_RECORDS_PATH, batch_size, train_gen_session)

        while True:
            try:
                l, ab = train_gen_session.run(next_batch)  # retrieve a batch of records
                yield ([l], ab)
            except:
                # if it crashes due to some reason
                iterator = dataset.make_initializable_iterator()
                train_gen_session.run(iterator.initializer)
                next_batch = iterator.get_next()

                l, ab= train_gen_session.run(next_batch)
                yield ([l], ab)


def val_batch_generator(batch_size):
    '''
    Generator which wraps a tf.data.Dataset object to read in the
    TFRecord more conveniently.
    '''
    if not os.path.exists(VAL_RECORDS_PATH):
        print("\n\n", '*' * 50, "\n")
        print("Please create the TFRecord of this dataset by running `data_utils_CNN.py` script with validation data")
        exit(0)

    with tf.Session() as val_generator_session:
        dataset, next_batch = _construct_dataset(VAL_RECORDS_PATH, batch_size, val_generator_session)

        while True:
            try:
                l, ab = val_generator_session.run(next_batch)  # retrieve a batch of records
                yield ([l], ab)
            except:
                # if it crashes due to some reason
                iterator = dataset.make_initializable_iterator()
                val_generator_session.run(iterator.initializer)
                next_batch = iterator.get_next()

                l, ab = val_generator_session.run(next_batch)
                yield ([l], ab)


def prepare_input_image_batch(X, batchsize=100):
    '''
    This is a helper function which does the same as _preprocess_batch,
    but it is meant to be used with images during testing, not training.
    Args:
        X: A grayscale image
    '''
    X_processed = X / 255.  # normalize grayscale image
    X_grayscale = gray2rgb(rgb2gray(X_processed))
    X_lab = rgb2lab(X_grayscale)[:, :, :, 0]
    X_lab = X_lab.reshape(X_lab.shape + (1,))
    X_lab = 2 * X_lab / 100 - 1.

    return X_lab


def postprocess_output(X_lab, y, image_size=None):
    '''
    This is a helper function for test time to convert and save the
    the processed image into the 'results' directory.
    Args:
        X_lab: L channel extracted from the grayscale image
        y: AB channels predicted by the colorizer network
        image_size: output image size
    '''
    y *= 127.  # scale the predictions to [-127, 127]
    X_lab = (X_lab + 1) * 50.  # scale the L channel to [0, 100]

    image_size = IMAGE_SIZE if image_size is None else image_size  # set a default image size if needed

    for i in range(len(y)):
        cur = np.zeros((image_size, image_size, 3))
        cur[:, :, 0] = X_lab[i, :, :, 0]
        cur[:, :, 1:] = y[i]
        imsave("results/img_%d.png" % (i + 1), lab2rgb(cur))

        if i % (len(y) // 20) == 0:
            print("Se ha procesado un %0.2f de las imagenes de test" % (i / float(len(y)) * 100))


if __name__ == '__main__':
    # generate the train tf record file
    generate_train_records(batch_size=200)
    #si la ram colapsa cambiar el argumento batch_size

    # generate the validation tf record file
    generate_validation_records(batch_size=100)