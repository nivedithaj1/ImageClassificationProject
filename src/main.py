# Imports
import streamlit as st
import pandas as pd
import os

# Project Imports
from utils import get_groceries, load_image

# Set Page Title
st.set_page_config(
    page_title="Image Quality Classifier"
)


MODEL_TYPE = 'pb'  # pb / tflite
GROCERIES = get_groceries()

if MODEL_TYPE == 'pb':
    from load_pb_model import IMG_INP_SHAPE, predict, load_models
    # Load and cache the model
    models = load_models(GROCERIES)
if MODEL_TYPE == 'tflite':
    from load_tflite_model import IMG_INP_SHAPE, predict


# Set Page Heading
st.title("Image Quality Classifier")
# Drodown to select the grocery type
grocery_type = st.selectbox(
    "Select the Grocery type",
    GROCERIES,
)

selection = "Grocery type selected: {}".format(grocery_type)
st.write(selection)

uploaded_file = st.file_uploader(
    label="Choose an image", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    img = load_image(uploaded_file, IMG_INP_SHAPE)
    st.image(img)
    st.write("Image Uploaded successfully")


if uploaded_file is not None:
    col1, col2 = st.beta_columns([2, 3])
    classify_image = col1.button('Classify Image')
    if classify_image:
        if MODEL_TYPE == 'pb':
            prediction = predict(models.get(grocery_type), uploaded_file)
        else:
            model_path = os.path.join('models', grocery_type + ".tflite")
            prediction = predict(model_path, img)
        prediction = pd.DataFrame(prediction)
        col2.dataframe(prediction.style.highlight_max(axis=1, color='SeaGreen'))
