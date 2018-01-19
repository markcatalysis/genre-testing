import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import copy
import os 
import librosa

genre_list = os.listdir('genres/')
X=[] 
y=np.array([])
for genre in genre_list:
    for song in os.listdir('genres/'+genre+'/'):
        src, sr= librosa.load('genres/'+genre+'/'+song, sr=None, mono=True)
        X.append(src)
        y =np.append(y,genre)
X2=copy.copy(X)
x_len_min=np.min([len(x) for x in X])
X_new=np.array([x[:x_len_min] for x in X])
X_arr=np.stack(X_new, axis=0)
X_arr_exp=np.expand_dims(X_arr,1)
y_dummies=pd.get_dummies(pd.Series(y), drop_first=False)
X_train, X_test, y_train, y_test=train_test_split(X_arr_exp,y_dummies,test_size=0.1)
X_train.shape

