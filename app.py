import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# Parameters
img_height = 224
img_width = 224
batch_size = 16

# Streamlit app
st.title("Breast Cancer Classification")

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
                export_dir = 'MobileNetV2_save_model_finetuned/'  # Adjust path if needed
                loaded_model = tf.keras.models.load_model(export_dir)
                
                # Make predictions
                predictions = loaded_model.predict(image_array)
                predicted_class = np.argmax(predictions, axis=1)
                
                # Display predictions
                st.success("Prediction completed!")
                st.subheader("Prediction Result:")
                result=""
                if predicted_class[0]==1:
                    result="Positive"
                if predicted_class[0]==0:
                    result="Negative"
                st.write(f"Predicted Class: {result}")
            except Exception as e:
                st.error(f"Error: {e}")
