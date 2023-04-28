import numpy as np
import h5py
import tensorflow as tf
import os
import pandas as pd
from PIL import Image


def load_dataset(filename):
    hf = h5py.File(filename, "r")
    dataset_orig = hf["dataset"][:]
    labels_orig = hf["labels"][:]
    classes_orig = hf["classes"][:]
    
    return dataset_orig, labels_orig, classes_orig

def load_dataset_tvt(filename):
    hf = h5py.File(filename, "r")
    train = hf["train"][:]
    labels_train = hf["labels_train"][:]
    valid = hf["valid"][:]
    labels_valid = hf["labels_valid"][:]
    test = hf["test"][:]
    labels_test = hf["labels_test"][:]
    
    return train, labels_train, valid, labels_valid, test, labels_test


def save_ds_h5(filename, dataset, labels, classes):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('dataset', data=dataset)
    hf.create_dataset('labels', data=labels)
    hf.create_dataset('classes', data=classes)
    hf.close
    return

def save_ds_h5_tvt(filename, train, labels_train, valid, labels_valid, test, labels_test):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('train', data=train)
    hf.create_dataset('labels_train', data=labels_train)
    hf.create_dataset('valid', data=valid)
    hf.create_dataset('labels_valid', data=labels_valid)
    hf.create_dataset('test', data=test)
    hf.create_dataset('labels_test', data=labels_test)
    hf.close
    return

def my_patchify(img,patch_shape,steps):
    # take the first rows and divide it, then pass to the second rows and so on
    img = np.array(Image.fromarray(img).resize((1040,1300)))
    patches = []
    patch_r = patch_shape[0]; patch_c = patch_shape[1]
    step_r = steps[0]; step_c = steps[1]
    for r in range(0,np.round(((img.shape[0]-(patch_r-step_r))/step_r)).astype(int)):
        c_plus = 0
        for c in range(0,np.round(((img.shape[1]-(patch_c-step_c))/step_c)).astype(int)):
            patches.append(img[r*step_r:r*step_r+patch_r,c*step_c+c_plus:c*step_c+patch_c+c_plus])
            if step_c == 173 and c%2==1:
                c_plus += 1
    return np.array(patches)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def normalize_img(image,label):
    return (tf.cast(image, tf.float32) / 255.,label)

def create_tf_dataset(array, labels, batch_size, shuffle, normalize=False, repeat=False, cache_file=False):
    
    dataset = tf.data.Dataset.from_tensor_slices((array,labels)) # Create a Dataset object
    
    if normalize: dataset = dataset.map(normalize_img, num_parallel_calls=os.cpu_count()) # Map the normalize_img function
    if cache_file: dataset = dataset.cache(cache_file) # Cache dataset
    if shuffle: dataset = dataset.shuffle(len(array)) # Shuffle
    if repeat: dataset = dataset.repeat() # Repeat the dataset indefinitely

    dataset = dataset.batch(batch_size=batch_size) # Batch
    dataset = dataset.prefetch(buffer_size=1) # Prefetch
    
    return dataset