from urllib.request import urlopen
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import os


tf.executing_eagerly()
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(
    page_title="Neural Style Transfer", layout="wide"
)

def load_image(image_path, image_size=(512, 256)):
    """Loads and preprocesses images."""
    # Cache image file locally.
    if "http" in image_path:
        image_path = tf.keras.utils.get_file(os.path.basename(image_path)[-128:], image_path)
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
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
    image_upload1 = st.sidebar.file_uploader("Load your image 1 here",type=["jpeg", "png", "jpg"], accept_multiple_files=False, key=None, help="Upload the image whom you want to style")
    image_upload2 = st.sidebar.file_uploader("Load your image 2 here",type=["jpeg", "png", "jpg"], accept_multiple_files=False, key=None, help="Upload the image whose style you want")
    col1,col2,col3= st.columns(3)
    
    st.sidebar.title("Style Transfer")
    st.sidebar.markdown("Your personal neural style transfer")

    with st.spinner("Loading content image.."):
        if image_upload1 is not None:
            col1.header("Content Image")
            col1.image(image_upload1,use_column_width=True)
            content_image = load_image(image_upload1)
        else:
            original_image = load_image("https://raw.githubusercontent.com/RadientBrain/Neural-Style-Transfer---Streamlit/main/1.jpg")
    
    with st.spinner("Loading style image.."):
        if image_upload2 is not None:
            col2.header("Style Image")
            col2.image(image_upload2,use_column_width=True)
            style_image = load_image(image_upload2)
            style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')
        else:
            style_image = load_image("https://raw.githubusercontent.com/RadientBrain/Neural-Style-Transfer---Streamlit/main/2.jpg")
            style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')
    

    if st.sidebar.button(label="Start Styling"):
        if image_upload2 and image_upload1:
            with st.spinner('Generating Stylized image ...'):

                # Load image stylization module.
                stylize_model = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

                results = stylize_model(tf.constant(original_image), tf.constant(style_image))
                stylized_photo = results[0]
                col3.header("Final Image")
                col3.image(np.array(stylized_photo))
                st.download_button(label="Download Final Image", data=export_image(stylized_photo), file_name="stylized_image.png", mime="image/png")

        else:
            st.sidebar.markdown("Please upload images...")
            
            
if __name__ == "__main__":
    st_ui()

