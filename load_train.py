import pandas as pd
import numpy as np
import glob
import os
import librosa
import progressbar
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from joblib import Parallel, delayed

def extract_features(files):
    # Sets the name to be the path to where the file is in my computer
    #file_name = os.path.join(os.path.abspath('voice') + '/' + str(files.file))
    file_name = files

    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    #stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    #chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # Computes a mel-scaled spectrogram.
    #mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    # Computes spectral contrast
    #contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # Computes the tonal centroid features (tonnetz)
    #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              #sr=sample_rate).T, axis=0)

    # We add also the classes of each file as a label at the end
    #label = files.label
    #print(mfccs.shape)
    #mfccs = mfccs.reshape(1,40)
    return mfccs#, chroma, mel, contrast, tonnetz#, label

def encode_labels(labels):
    classes = set(labels)

    le = preprocessing.LabelEncoder()
    le.fit(list(classes))

    #print(np.unique(le.transform(labels)))
    return to_categorical(le.transform(labels))

def define_model(classes):
    model = Sequential()

    model.add(Dense(40, input_shape=(40,), activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    return model

def split_data(files,labels):
    X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size = 0.33, random_state = 42)
    return X_train, X_test, y_train, y_test

persons = glob.glob("LibriSpeech/train-clean-100/*/*/*.flac")

def load_data(x):
    files.append(extract_features(x))
    labels.append(os.path.basename(x).split("-")[0])

    return



#print(len(persons))
files = []
labels = []

# Parallel(n_jobs=4)(delayed(load_data)(x) for x in progressbar.progressbar(persons[:500]))
for x in progressbar.progressbar(persons[:1000]):
    load_data(x)
    #print(x,os.path.basename(x).split("-")[0])
    #files.append(extract_features(x))
    #labels.append(os.path.basename(x).split("-")[0])


    #extract_features(x)
    #print(extract_features(x)[0].shape)

classes = len(set(labels))
labels = encode_labels(labels)
files = np.array(files)
labels = np.array(labels)
#print(files.shape,labels.shape)

X_train, X_test, y_train, y_test = split_data(files,labels)

model = define_model(classes=classes)
history = model.fit(X_train, y_train, batch_size=32, epochs=100,
                    validation_data=(X_test, y_test))

# Check out our train accuracy and validation accuracy over epochs.
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Set figure size.
plt.figure(figsize=(12, 8))

# Generate line plot of training, testing loss over epochs.
plt.plot(train_accuracy, label='Training Accuracy', color='#185fad')
plt.plot(val_accuracy, label='Validation Accuracy', color='orange')

# Set title
plt.title('Training and Validation Accuracy by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)
plt.xticks(range(0,100,5), range(0,100,5))

plt.legend(fontsize = 18);

plt.savefig('training.png')