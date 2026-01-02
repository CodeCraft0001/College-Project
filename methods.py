import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout
from tensorflow import keras
import matplotlib.pyplot as plt
from imageio import imread
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, BatchNormalization



#declaration path of dataset for training and testing

trainpath = 'data\spiral\training'
testpath = 'data\spiral\testing'


def processimgfromdir(path):
    #processing images from dataset
    test_datagen = ImageDataGenerator(
        rescale=1./255    
    )


    test_generator = test_datagen.flow_from_directory(
        path,
        target_size = (224,224),
        batch_size = 32,
        class_mode = 'binary',
        
    )
    return test_generator



import librosa
import numpy as np

def mfcc_target1(predict):
    batch_features = []
    files = [predict]
    # print("loaded %d audio files" % len(files))
    for wav in files:
        if not wav.endswith(".wav"): continue
        # Load audio file
        wave, sr = librosa.load(predict, mono=True)
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=20)
        # Pad MFCC array with zeros
        mfcc = np.pad(mfcc, ((0, 0), (0, 10000 - len(mfcc[0]))), mode='constant', constant_values=0)
        # Convert features to 2D array
        samples = []
        for i in range(0, 20):
            row = []
            for j in range(80):
                row.append(mfcc[i][j])
            samples.append(row)
        batch_features.append(np.array(samples))
    print(np.array(batch_features).shape)
    return batch_features

