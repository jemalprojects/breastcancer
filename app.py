import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from streamlit_option_menu import option_menu
from huggingface_hub import snapshot_download

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="♈",
    # page_icon=icon,
    layout="wide"
)
st.markdown('''
<style>
.stApp [data-testid="stToolbar"]{
    display:none;
}
</style>
''', unsafe_allow_html=True)
hide_st_text = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_text, unsafe_allow_html=True)


st.subheader("Breast Cancer Classification", divider=True)
model_dir = snapshot_download("abatejemal/breastcancer")

# Parameters
img_height = 224
img_width = 224
batch_size = 16

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Submit button
if uploaded_file:
    # Display the uploaded image
    # st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Add a button to process the image
    if st.button("Submit for Classification"):
        with st.spinner("Processing..."):
            try:
                # Load the image and resize it
                image = load_img(uploaded_file, target_size=(img_height, img_width))
                
                # Convert the image to a numpy array and expand dimensions
                image_array = img_to_array(image)
                image_array = tf.expand_dims(image_array, axis=0)  # Add batch dimension
                
                # Normalize the image
                image_array = image_array / 255.0  # Assuming model expects input in range [0, 1]
                
                # Load the saved model
                # export_dir = 'MobileNetV2_save_model/'  # Adjust path if needed
                # export_dir = '/MobileNetV2_save_model_finetuned/'  # Adjust path if needed
                # export_dir = f'{model_dir}/MobileNetV2_save_model/'
                export_dir = f'{model_dir}/MobileNetV1_finetuned'
                loaded_model = tf.keras.models.load_model(export_dir)
                
                # Make predictions
                predictions = loaded_model.predict(image_array)
                predicted_class = np.argmax(predictions, axis=1)
                score=predicted_class[0]
                score1=predictions[0][score]*100
                score1 = round(score1, 2)
                # Display predictions
                st.success("Prediction completed!")
                st.subheader("Prediction Result:")
                result=""
                if predicted_class[0]==1:
                    result="Sick"
                    st.write(f"Result: :red[{result} ({score1}%)]")
                if predicted_class[0]==0:
                    result="Healthy"
                    st.write(f"Result: :green[{result} ({score1}%)]")
                # st.write(f"Predicted Class: {result}")
            except Exception as e:
                st.error(f"Error: {e}")
