import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import json

from sklearn.model_selection import train_test_split
from functools import partial

print("Tensorflow version " + tf.__version__)

strategy = tf.distribute.MirroredStrategy()
print("Number of replicas:", strategy.num_replicas_in_sync)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16*strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]

samp_subm = pd.read_csv('sample_submission.csv')
print(samp_subm.head())

train_filenames, val_filenames = train_test_split(tf.io.gfile.glob(
    './tfrecords/train*.tfrec'), test_size=0.25, random_state=2020)
test_filenames = tf.io.gfile.glob('./tfrecords/test*.tfrec')

raw_dataset = tf.data.TFRecordDataset(train_filenames)
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    # print(example.features)


def number_of_files(filenames):
    """ Evaluate the number on files """

    num = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
           for filename in filenames]
    return np.sum(num)


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [*IMAGE_SIZE])
    image = tf.cast(image, tf.float32)/255.
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def read_tfrecord(example, labeled):
    tfrecord_format = {  # tfrecord 格式
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    } if labeled else {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    idnum = example['image_name']
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
    return dataset


def get_train_dataset(filenames, labeled=True, ordered=False):
    dataset = load_dataset(filenames, labeled=labeled, ordered=ordered)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2020)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


def get_val_dataset(filenames, labeled=True, ordered=False):
    dataset = load_dataset(filenames, labeled=labeled, ordered=ordered)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


def get_test_dataset(filenames, labeled=False, ordered=True):
    dataset = load_dataset(filenames, labeled=labeled, ordered=ordered)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(20, 20))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        if label_batch[n]:
            plt.title("MALIGNANT")
        else:
            plt.title("BENIGN")
        plt.axis("off")


train_dataset = get_train_dataset(train_filenames)
val_dataset = get_val_dataset(val_filenames)
test_dataset = get_test_dataset(test_filenames)

metrics = [tf.keras.metrics.AUC(name='auc', multi_label=True)]
learning_rate = 1e-3


def make_model():
    base_model = tf.keras.applications.ResNet50(weights='imagenet',
                                                include_top=False,
                                                input_shape=[*IMAGE_SIZE, 3])
    base_model.trainable = True
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=metrics
    )
    return model


with strategy.scope():
    model = make_model()

model.summary()


history = model.fit(train_dataset,
                    epochs=6,
                    validation_data=val_dataset,
                    steps_per_epoch=number_of_files(train_filenames)//BATCH_SIZE)


def to_float32(image, idnum):
    return tf.cast(image, tf.float32), idnum


test_dataset = test_dataset.map(to_float32)
test_images = test_dataset.map(lambda image, idnum: image)

preds = model.predict(test_images, verbose=1)
preds = preds.reshape(len(preds))
