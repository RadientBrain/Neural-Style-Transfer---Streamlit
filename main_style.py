from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
from PIL import Image
import streamlit as st

def load_image(image_path, image_size=(512, 256)):
    img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img
    
def export_image(tf_img):
    tf_img = tf_img*255
    tf_img = np.array(tf_img, dtype=np.uint8)
    if np.ndim(tf_img)>3:
        assert tf_img.shape[0] == 1
        img = tf_img[0]
    return Image.fromarray(img)

def st_ui():
    st.title("Image Styling")
    image_upload1 = st.sidebar.file_uploader("Load your image 1 here")
    image_upload2 = st.sidebar.file_uploader("Load your image 2 here")

    if image_upload1 is not None:
        original_image = load_image(image_upload1)
        image1 = Image.open(image_upload1)
    else:
        original_image = load_image("https://raw.githubusercontent.com/RadientBrain/Neural-Style-Transfer---Streamlit/blob/main/2.jpg")
        image1 = Image.open("https://raw.githubusercontent.com/RadientBrain/Neural-Style-Transfer---Streamlit/blob/main/2.jpg")
    if image_upload2 is not None:
        style_image = load_image(image_upload2)
        image2 = Image.open(image_upload2)
    else:
        style_image = load_image("https://raw.githubusercontent.com/RadientBrain/Neural-Style-Transfer---Streamlit/blob/main/2.jpg")
        image2 = Image.open("https://raw.githubusercontent.com/RadientBrain/Neural-Style-Transfer---Streamlit/blob/main/2.jpg")
    
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')
    st.header("Real Image")
    st.image(image1)
    st.header("Style Image")
    st.image(image2)
    
    # Load image stylization module.
    stylize_model = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    with st.spinner("Styling your image"):
        results = stylize_model(tf.constant(original_image), tf.constant(style_image))
        stylized_photo = results[0]
    
    export_image(stylized_photo).save("my_stylized_photo.png")
    st.header("Final Styled Image")
    st.image(Image.open("/content/my_stylized_photo.png"))

if __name__ == "__main__":
    st_ui()

