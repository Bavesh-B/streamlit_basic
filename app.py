import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, Dropout, Dense, MaxPool2D

def predict_class(img, weights):
  model = Sequential()
  model.add(Conv2D(16, kernel_size = (3,3), input_shape = (28, 28, 3), activation = 'relu', padding = 'same'))
  model.add(MaxPool2D(pool_size = (2,2)))

  model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same'))
  model.add(MaxPool2D(pool_size = (2,2), padding = 'same'))

  model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same'))
  model.add(MaxPool2D(pool_size = (2,2), padding = 'same'))
  model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu', padding = 'same'))
  model.add(MaxPool2D(pool_size = (2,2), padding = 'same'))

  model.add(Flatten())
  model.add(Dense(64, activation = 'relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(7, activation='softmax'))
  model.load_weights(weights)

  # Creating array of right shape to feed into model
  data = np.ndarray(shape = (1, 28, 28, 3), dtype = np.float32)

  image = img
  size = (28, 28)
  image = ImageOps.fit(image, size, Image.ANTIALIAS)

  image_array = np.asarray(image)
  # normalizing the image array
  normalized_image_array = (image_array.astype(np.float32) / 255)

  data[0] = normalized_image_array

  y_pred = model.predict(data)
  val = np.argmax(y_pred)

  label_mapping = {
    0: 'nv',
    1: 'mel',
    2: 'bkl',
    3: 'bcc',
    4: 'akiec',
    5: 'vasc',
    6: 'df'
  }

  return y_pred, val, label_mapping[val]


uploaded_file = st.file_uploader("Choose a histopathological image", type = 'jpg')

if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img, caption = 'Uploaded file', use_column_width = True)
  st.write("")
  st.write("Classifying...")

  y_pred, val, c = predict_class(img, '/content/drive/MyDrive/Skin_Cancer.hdf5')
  st.write(f'The above image is a {c} lesion.')