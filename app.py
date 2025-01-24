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
            ._container_gzau3_1 _viewerBadge_nim44_23{visibility: hidden;}
            ._profilePreview_gzau3_63{visibility: hidden;}
            </style>
            """
st.markdown(hide_st_text, unsafe_allow_html=True)

with st.sidebar:
    choose = option_menu("MENU", ["Home", "Make Prediction", "Retrain Model", "Update Datasets", "Learn"],
                         icons=['house', 'kanban', 'book','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )
# logo = Image.open(r'C:\Users\13525\Desktop\Insights_Bees_logo.png')
# profile = Image.open(r'C:\Users\13525\Desktop\medium_profile.png')
if choose == "Home":
    st.subheader("Home", divider=True)
elif choose == "Make Prediction":
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
                    export_dir = f'{model_dir}/MobileNetV2_save_model/'
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
                        result="Positive"
                        st.write(f"Result: :red[{result} ({score1}%)]")
                    if predicted_class[0]==0:
                        result="Negative"
                        st.write(f"Result: :green[{result} ({score1}%)]")
                    # st.write(f"Predicted Class: {result}")
                except Exception as e:
                    st.error(f"Error: {e}")
