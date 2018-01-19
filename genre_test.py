import numpy as np
import os, sys
import librosa
import librosa.display
import keras
import kapre
from keras.models import Sequential
from kapre.time_frequency import Melspectrogram
from kapre.time_frequency import Spectrogram
from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from kapre.augmentation import AdditiveNoise
from kapre.utils import Normalization2D
from keras import backend
from keras.utils import np_utils
import os
from os.path import isfile
from timeit import default_timer as timer
from datetime import datetime
from librosa import display
import matplotlib.pyplot as plt

now = datetime.now()

def print_version_info():
    print('%s/%s/%s' % (now.year, now.month, now.day))
    print('Keras version: {}'.format(keras.__version__))
    if keras.backend._BACKEND == 'tensorflow':
        import tensorflow
        print('Keras backend: {}: {}'.format(keras.backend._backend, tensorflow.__version__))
    else:
        import theano
        print('Keras backend: {}: {}'.format(keras.backend._backend, theano.__version__))
    print('Keras image dim ordering: {}'.format(keras.backend.image_dim_ordering()))
    print('Kapre version: {}'.format(kapre.__version__))

print_version_info()


# src = np.random.random((1, SR * 3))
# model = Sequential()
# model.add(Melspectrogram(sr=SR, n_mels=128,
#           n_dft=512, n_hop=256, input_shape=src.shape,
#           return_decibel_melgram=True,
#           trainable_kernel=True, name='melgram'))



# def genre_folder(genre):
#     path = "genres/"
#     return path

"""
the bits of script below are specific to the gtzan dataset as unzipped in its natural form~
"""
genre_list = os.listdir('genres/')

def sound_load(genre):
    folder='genres/{}/'.format(genre)
    song_list=sorted(os.listdir(folder))
    for song in song_list:
        yield folder+song
        # src, sr = librosa.load(folder+song, sr=None, mono=True)
        # yield src, sr

# print(src.shape)
# print(sr)

"""
Adjusting example code from kapre.
"""

def check_model(model):
    model.summary(line_length=80, positions=[.33, .65, .8, 1.])

    batch_input_shape = (2,) + model.input_shape[1:]
    batch_output_shape = (2,) + model.output_shape[1:]
    model.compile('sgd', 'mse')
    model.fit(np.random.uniform(size=batch_input_shape),
    np.random.uniform(size=batch_output_shape), epochs=1)

def visualise_model(model, src, sr, logam=False):
    n_ch, nsp_src = model.input_shape[1:]
    # src, _ = librosa.load('../srcs/bensound-cute.mp3', sr=SR, mono=True)
    src = src[:nsp_src]
    src_batch = src[np.newaxis, :]
    pred = model.predict(x=src_batch)
    if keras.backend.image_data_format == 'channels_first':
        result = pred[0, 0]
    else:
        result = pred[0, :, :, 0]

    if logam:
        result = librosa.logamplitude(result)
    display.specshow(result,
                     y_axis='linear', sr=sr)


"""
Testing Code
"""
# s=sound_load(genre_list[0])
# s.next()
# src, sr= librosa.load(s.next(), sr=None, mono=True)
# src = src[np.newaxis, :]
# model = Sequential()
# model.add(Melspectrogram(sr=sr, n_mels=128,
#           n_dft=512, n_hop=256, input_shape=src[np.newaxis,:].shape,
#           return_decibel_melgram=False, power_melgram=2.0,
#           trainable_kernel=True, name='melgram'))
# plt.figure(figsize=(14, 8))
# plt.subplot(2, 2, 1)
# plt.title('log-MelSpectrogram by Kapre')
# visualise_model(model, src[np.newaxis, :], sr, logam=True)
# model2 = Sequential()
# model2.add(Spectrogram(n_dft=512, n_hop=256, input_shape=src[np.newaxis,:].shape,
#           return_decibel_spectrogram=False, power_spectrogram=2.0,
#           trainable_kernel=True, name='static_stft'))
# check_model(model2)
# plt.subplot(2, 2, 2)
# plt.title('log-Spectrogram by Kapre')
# visualise_model(model2, src[np.newaxis, :], sr, logam=True)
# plt.subplot(2, 2, 3)
# display.specshow(librosa.logamplitude(np.abs(librosa.stft(src[: sr * 3], 512, 256)) ** 2, ref_power=1.0),
#                          y_axis='linear', sr=sr)
# plt.title('log-Spectrogram by Librosa')

# plt.show()
# src[np.newaxis,:].shape

"""
Model
"""

def build_model(X,Y, nb_classes, kernel_size=(2,2),nb_layers=4):
    nb_filters = 16  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling
    # convolution kernel size
    input_shape = (1,X.shape[2])
    model = Sequential()
    model.add(Melspectrogram(sr=sr, n_mels=128,
          n_dft=512, n_hop=256, input_shape=input_shape,
          return_decibel_melgram=False,
          trainable_kernel=False, name='melgram'))
    model.add(AdditiveNoise(power=0.1))
    model.add(Normalization2D(str_axis='batch')) # or 'channel', 'time', 'batch', 'data_sample'
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
    border_mode='valid', input_shape=input_shape,init="he_normal"))
    model.add(BatchNormalization(axis=1))
    model.add(ELU(alpha=0.1))

    for layer in range(nb_layers-1):
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],init="he_normal"))
        model.add(BatchNormalization(axis=1))
#        model.add(LeakyReLU(alpha=0.2))
        model.add(ELU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128,init="he_normal"))
#    model.add(Activation('tanh'))
    model.add(ELU(alpha=0.1))
    model.add(Dropout(0.1))
    model.add(Dense(nb_classes,init="he_normal"))
    model.add(Activation("softmax"))
    return model

'''
Data Munging
'''
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import copy
#  genre_list = os.listdir('genres/')

# X=[] 
# y=np.array([])
# for genre in genre_list:
#     for song in os.listdir('genres/'+genre+'/'):
#         src, sr= librosa.load('genres/'+genre+'/'+song, sr=None, mono=True)
#         X.append(src)
#         y=np.append(y,genre)
# X2=copy.copy(X)
# x_len_min=np.min([len(x) for x in X])
# X_new=np.array([x[:x_len_min] for x in X])
# X_arr=np.stack(X_new, axis=0)
# X_arr_exp=np.expand_dims(X_arr,1)
# y_dummies=pd.get_dummies(pd.Series(y), drop_first=False)
# X_train, X_test, y_train, y_test=train_test_split(X_arr_exp,y_dummies,test_size=0.2)
# X_train.shape

trained_model=build_model(X_train,y_train, len(genre_list))
optimizer=Adadelta(lr=.1)
trained_model.compile(loss='categorical_crossentropy',
          optimizer=optimizer,
          metrics=['accuracy'])
batch_size = 10
nb_epoch = 25
# check_model(trained_model)
# trained_model.get_weights()
check_model(trained_model)
# checkpointer
checkpoint_filepath = 'weights.hdf5'
checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=True)
# load_checkpoint = True
trained_model.fit(X_train, y_train.values, batch_size=batch_size, epochs=nb_epoch,
      verbose=1, callbacks=[checkpointer], validation_data=(X_test, y_test.values))
