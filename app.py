import streamlit as st
from predict import tell_genre
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

# %matplotlib inline
st.title("Upload the image to know the top 3 genre it belongs to!")
uploaded_file=st.file_uploader("Chose an image....")
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Predicting...")
    st.write("The predicted movie genres are:")
    classes,proba,top_3= tell_genre(uploaded_file)
    for i in range(3):
        st.write("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))