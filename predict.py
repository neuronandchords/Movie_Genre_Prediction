import streamlit as st
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import PIL
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from tqdm import tqdm
from keras.models import model_from_json

def tell_genre(image1):
    train=pd.read_csv('train.csv')
    img=image.load_img(image1, target_size=(250,250,3))
    img=image.img_to_array(img)
    img = img/255
    #loading the model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    classes = np.array(train.columns[2:])
    proba = loaded_model.predict(img.reshape(1,250,250,3))
    top_3 = np.argsort(proba[0])[:-4:-1]
    '''
    for i in range(3):
        print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
        '''
    return classes,proba, top_3

