# DeepLearning Group Project
# Project Done By:
# André Sousa 20240517
# Francisco Pontes 20211583
# Isabella Costa 20240685
# Jéssica Cristas 20240488
# Tiago Castilho 20240489


from google.colab import drive 
drive.mount('/content/drive')

import os
import csv
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report




# #Stratified data split for train test and validation with proportion of 70%, 15% and 15% respectively
# #EXECUTE ONCE

def split_data(base_dir, output_base):
    data = []
    for species_name in os.listdir(base_dir):
        species_path = os.path.join(base_dir, species_name)
        if os.path.isdir(species_path):
            for img_file in os.listdir(species_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(species_path, img_file)
                    data.append({'filepath': img_path, 'label': species_name})

    df = pd.DataFrame(data)

    # train test split (70 15 15)
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)


    def copy_images(df, subset):
        for _, row in df.iterrows():
            label = row['label']
            src = row['filepath']
            dst_dir = os.path.join(output_base, subset, label)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(src))
            shutil.copy2(src, dst)

    # Copiar as imagens
    copy_images(train_df, 'train')
    copy_images(val_df, 'val')
    copy_images(test_df, 'test')




normalization_layer = layers.Rescaling(1./255)
BATCH_SIZE = 32
train_dir_path = "/content/drive/MyDrive/Deep Learning/Group17_DL_Project/data_split_filtered/train"
val_dir_path = "/content/drive/MyDrive/Deep Learning/Group17_DL_Project/data_split_filtered/val"
test_dir_path = "/content/drive/MyDrive/Deep Learning/Group17_DL_Project/data_split_filtered/test"
metadata_path = "/content/drive/MyDrive/Deep Learning/Group17_DL_Project/metadata.csv"




def data_aug():
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(factor=0.2),
    ])
    return data_augmentation

def resize_with_padding(image, label):
    image = tf.image.resize_with_pad(image, 299, 299)
    return image, label

os.makedirs("tfrecords", exist_ok=True)
paths = {
    "train": "tfrecords/train.tfrecord",
    "val": "tfrecords/val.tfrecord",
    "test": "tfrecords/test.tfrecord"
}

# TF record functions

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))
def serialize_example(image, label):
    image_encoded = tf.io.encode_jpeg(tf.cast(image * 255.0, tf.uint8)).numpy()
    feature = {
        'image': _bytes_feature(image_encoded),
        'label': _int64_feature(label.numpy())
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

data_augmentation = data_aug() #We need to do it to not randomly use augmentation per each for cycle

def write_tfrecord(dataset, path, apply_augmentation=False):
    with tf.io.TFRecordWriter(path) as writer:
        for images, labels in dataset:
            for img, lbl in zip(images, labels):
                if apply_augmentation:
                    img = data_augmentation(img[None, ...], training=True)[0]
                img = normalization_layer(img)
                example = serialize_example(img, lbl)
                writer.write(example)


def caculate_class_weights(train_ds, n_classes):
    labels_list = []

    for _, labels in train_ds.unbatch():
        class_idx = tf.argmax(labels).numpy()
        labels_list.append(class_idx)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(n_classes),
        y=labels_list
    )

    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    return class_weights_dict

def evaluate_f1_scores(val_ds, model):
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(y_true, y_pred, output_dict=True)

    low_f1_classes = [int(cls) for cls, metrics in report.items() if cls.isdigit() and metrics["f1-score"] < 0.3]

    print(f"Classes with F1-Score < 0.3: {low_f1_classes}")

    return low_f1_classes, report