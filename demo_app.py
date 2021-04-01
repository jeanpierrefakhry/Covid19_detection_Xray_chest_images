import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from PIL import Image
import streamlit as st

PATH = "model-weights/"
WEIGHTS = "covidModel.h5"

@st.cache(allow_output_mutation= True)

def load_own_model (weights):
    return load_model(weights)

if __name__=="__main__":
    result = st.empty()
    uploaded_img = st.file_uploader(label='upload your image: ')
    if uploaded_img:
        st.image(uploaded_img, caption="Chest X-Ray Picture", width=350)
        result.info("Please wait for your results")
        model = load_own_model(PATH+WEIGHTS)
        img = Image.open(uploaded_img).convert('RGB')
        img = img_to_array(img)
        img = resize(img,(64,64))
        img = np.expand_dims(img, axis=0)
        predicted_value = model.predict(img)[0][0]
        rounding = round(predicted_value)
        if rounding == 1:
            pred = "You don't have Covid!!!!"
        else:
            pred = "You have Covid, please make sure to contact your Doctor"

        result.success(pred)
