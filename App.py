import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# App
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred= model.predict(img)
    result = np.argmax(pred[0])
    return result

# Streamlit 
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')
st.markdown("<h1 style='color:#4CAF50;'>Reconocimiento de Dígitos escritos a mano</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color:#2196F3;'>Dibuja el dígito en el panel y presiona <i>'Predecir'</i></h4>", unsafe_allow_html=True)

drawing_mode = "freedraw"
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

if st.button('Predecir'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.markdown(f"<h2 style='color:#FF5722;'>El Dígito es: {res}</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:#E91E63;'>Por favor dibuja en el canvas el dígito.</h3>", unsafe_allow_html=True)

st.sidebar.markdown("<h3 style='color:#795548;'>Acerca de:</h3>", unsafe_allow_html=True)
st.sidebar.text("En esta aplicación se evalua ")
st.sidebar.text("la capacidad de un RNA de reconocer") 
st.sidebar.text("dígitos escritos a mano.")
st.sidebar.text("Basado en desarrollo de Vinay Uniyal")

