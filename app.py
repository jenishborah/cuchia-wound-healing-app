import streamlit as st
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from PIL import Image

# Load the trained model
model = load_model('final_cuchianet_model.keras')

# Define class names (must match your dataset)
class_names = ['normal_wound', 'retonic_acid']

# --- Page Config ---
st.set_page_config(
    page_title="Cuchia Wound Healing Predictor 🐟",
    page_icon="🧬",
    layout="centered"
)

# --- Sidebar ---
st.sidebar.title("About the Project")
st.sidebar.info(
    """
    This app predicts whether a **Cuchia wound healing** image is:
    - **Normal Repaired** or
    - **Retonic Acid Repaired**.

    Model based on **MobileNet + Fine-tuning** achieving >90% accuracy.

    Developed for academic and research purposes.
    """
)
st.sidebar.write("- By Chid")

# --- Main App ---
st.title("🔬 Cuchia Wound Healing Classifier")

st.write("Upload a microscopic image below. The model will predict the wound healing type.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")
    
    # Prediction Progress Bar
    with st.spinner('Predicting...'):
        # Preprocess the image
        image = image.resize((224, 224))  # MobileNet input size
        img_array = np.array(image)

        if img_array.shape[-1] == 4:  # remove alpha channel if present
            img_array = img_array[..., :3]

        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Prediction
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    # Display result
    st.success(f"🧠 Prediction: **{predicted_class.replace('_', ' ').capitalize()}**")
    st.info(f"🔵 Confidence: **{confidence:.2f}%**")
else:
    st.info('Please upload an image to start prediction.')

