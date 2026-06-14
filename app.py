import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config for a premium and modern look
st.set_page_config(
    page_title="PhoneMatrix",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for glassmorphism, modern fonts, gradients, and custom styling
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Apply Font Family specifically to main content and widgets to preserve system icons */
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    .gradient-text, .subtitle-text, .glass-card, .result-card, .result-title, .result-value, div.stButton > button {
        font-family: 'Outfit', sans-serif !important;
    }
    .stWidget label, .stSelectbox div, .stSlider div {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Background adjustments */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161a24 100%);
    }
    
    /* Main Card styling (Glassmorphism) */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Gradient title */
    .gradient-text {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .subtitle-text {
        color: #a0aec0;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 35px;
    }
    
    /* Prediction success card styling */
    .result-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 4px 20px 0 rgba(0, 242, 254, 0.15);
        animation: fadeIn 0.8s ease-in-out;
    }
    
    .result-title {
        color: #e2e8f0;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .result-value {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* Custom button styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: #ffffff;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.6);
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
    }

    div.stButton > button:first-child:active {
        transform: translateY(1px);
    }
    
    /* Input Label modifications */
    label {
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
    }
    
    /* Animation definition */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="gradient-text">📱 PhoneMatrix</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">A premium machine learning engine to predict estimated smartphone retail prices</div>', unsafe_allow_html=True)

# Main Form Container (Glassmorphic)
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # 2-Column layout for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Memory & Storage")
        # RAM Select box (Options specified in implementation guide)
        ram_val = st.selectbox(
            "RAM Capacity (GB)",
            options=[2, 4, 6, 8, 12, 16, 24],
            index=3, # Default: 8 GB
            help="Select the smartphone RAM capacity in Gigabytes."
        )
        
        # Storage Select box (Options specified in implementation guide)
        storage_val = st.selectbox(
            "Internal Storage (GB)",
            options=[32, 64, 128, 256, 512, 1024],
            index=3, # Default: 256 GB
            help="Select the internal storage capacity in Gigabytes."
        )
        
    with col2:
        st.subheader("Battery & Camera")
        # Battery Capacity Slider (2000 to 7000 mAh)
        battery_val = st.slider(
            "Battery Capacity (mAh)",
            min_value=2000,
            max_value=7000,
            value=5000,
            step=100,
            help="Drag to select the battery capacity in milliampere-hours."
        )
        
        # Main Camera Slider (8 to 200 MP)
        camera_val = st.slider(
            "Main Camera Resolution (MP)",
            min_value=8,
            max_value=200,
            value=50,
            step=1,
            help="Drag to select the primary rear camera resolution in Megapixels."
        )
        
    st.markdown('</div>', unsafe_allow_html=True)

# Trigger Model Prediction
predict_btn = st.button("Predict Price")

if predict_btn:
    model_path = "smartphone_model.pkl"
    
    if not os.path.exists(model_path):
        st.error(f"Error: Model file '{model_path}' not found! Please run train_model.py first to train and export the model.")
    else:
        # Load pre-trained model
        model = joblib.load(model_path)
        
        # Build pandas DataFrame for prediction with EXACT matched column names
        input_data = pd.DataFrame({
            'ram_gb': [float(ram_val)],
            'storage_gb': [float(storage_val)],
            'battery_mah': [float(battery_val)],
            'main_camera_mp': [float(camera_val)]
        })
        
        try:
            # Perform inference
            predicted_raw = model.predict(input_data)[0]
            
            # Format price: round to integer and format with Indian standard commas
            predicted_price = int(round(predicted_raw))
            formatted_price = f"₹{predicted_price:,}"
            
            # Display prediction in a beautiful card
            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">Estimated Retail Price</div>
                <div class="result-value">{formatted_price}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Also invoke fallback success banner for clarity & guide adherence
            st.success(f"Successfully calculated smartphone price: {formatted_price}")
            
        except Exception as e:
            st.error(f"Inference error: {str(e)}")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; color: #718096; font-size: 0.85rem;">
    PhoneMatrix • End-Semester Machine Learning Evaluation Project
</div>
""", unsafe_allow_html=True)
