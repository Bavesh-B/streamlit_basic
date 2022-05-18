import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, Dropout, Dense, MaxPool2D

from skimage.metrics import structural_similarity
import phasepack.phasecong as pc


def predict_class(img, m):
    # model = Sequential()
    # model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(
    #     28, 28, 3), activation='relu', padding='same'))
    # model.add(MaxPool2D(pool_size=(2, 2)))

    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

    # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    # model.add(Conv2D(128, kernel_size=(3, 3),
    #           activation='relu', padding='same'))
    # model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

    # model.add(Flatten())
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(7, activation='softmax'))
    # model.load_weights(weights)
    model = load_model(m)

    # Creating array of right shape to feed into model
    img2 = img.resize((28, 28))

    image_array = np.asarray(img2)
    new_one = image_array.reshape((1, 28, 28, 3))

    y_pred = model(new_one)
    # st.write(f'The predictions are {y_pred}')
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


def _assert_image_shapes_equal(org_img: np.ndarray, pred_img: np.ndarray, metric: str):
    # shape of the image should be like this (rows, cols, bands)
    # Please note that: The interpretation of a 3-dimension array read from rasterio is: (bands, rows, columns) while
    # image processing software like scikit-image, pillow and matplotlib are generally ordered: (rows, columns, bands)
    # in order efficiently swap the axis order one can use reshape_as_raster, reshape_as_image from rasterio.plot
    msg = (
        f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
        f"{str(org_img.shape)}, y_pred shape = {str(pred_img.shape)}"
    )

    assert org_img.shape == pred_img.shape, msg


def ssim(org_img, pred_img, max_p: int = 4095) -> float:
    """
    Structural Simularity Index
    """

    org_image = org_img
    size = (28, 28)
    org_image = ImageOps.fit(org_image, size, Image.Resampling.LANCZOS)

    pred_image = pred_img
    size = (28, 28)
    pred_image = ImageOps.fit(pred_image, size, Image.Resampling.LANCZOS)

    org_array = np.asarray(org_image)
    pred_array = np.asarray(pred_image)

    _assert_image_shapes_equal(org_array, pred_array, "SSIM")

    return structural_similarity(org_array, pred_array, data_range=max_p, multichannel=True)


st.title('Skin Lesion Detection from Histopathological images')

uploaded_file = st.file_uploader(
    "Choose a histopathological image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    # img2 = Image.open('test.jpg')
    st.image(img, caption='Uploaded file', use_column_width=True)

    # similarity = ssim(img, img2)
    # st.write("")
    # st.write(f'This is {similarity * 100}% histopathological image')

    # if similarity >= 0.85:
    st.write("")
    st.write("Classifying...")

    y_pred, val, c = predict_class(img, 'Skin_Cancer.sav')
    st.write(f'The above image is a {c} lesion.')

    # else:
    # st.write('Not Classifying as it is not a histopathological image.')
