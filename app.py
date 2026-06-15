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

# Initialize Session State values for widgets
if 'ram_gb' not in st.session_state:
    st.session_state['ram_gb'] = 8
if 'storage_gb' not in st.session_state:
    st.session_state['storage_gb'] = 256
if 'battery_mah' not in st.session_state:
    st.session_state['battery_mah'] = 5000
if 'main_camera_mp' not in st.session_state:
    st.session_state['main_camera_mp'] = 50

# Custom CSS for glassmorphism, modern fonts, gradients, and custom styling
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Apply Font Family specifically to main content and widgets to preserve system icons */
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    .gradient-text, .subtitle-text, div[data-testid="stBorderContainer"], .result-card, .result-title, .result-value, div.stButton > button {
        font-family: 'Outfit', sans-serif !important;
    }
    .stWidget label, .stSelectbox div, .stSlider div {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Background adjustments */
    .stApp {
        background: linear-gradient(135deg, var(--background-color) 0%, var(--secondary-background-color) 100%);
    }
    
    /* Main Card styling (Glassmorphism applied to native container) */
    div[data-testid="stBorderContainer"] {
        background: rgba(128, 128, 128, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(128, 128, 128, 0.15) !important;
        border-radius: 16px !important;
        padding: 30px !important;
        margin-bottom: 25px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1) !important;
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
        color: var(--text-color);
        opacity: 0.75;
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
    
    /* Custom preset button styling (horizontal column buttons) */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease !important;
        background: rgba(128, 128, 128, 0.05) !important;
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
        color: var(--text-color) !important;
        width: 100% !important;
    }
    
    div[data-testid="stHorizontalBlock"] button:hover {
        background: rgba(128, 128, 128, 0.15) !important;
        border-color: rgba(128, 128, 128, 0.3) !important;
        transform: translateY(-1px);
    }
    
    /* Input Label modifications */
    label {
        font-weight: 600 !important;
        color: var(--text-color) !important;
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

# ⚡ Quick Specification Presets Section
st.markdown('<div style="text-align: center; color: var(--text-color); font-weight: 600; font-size: 1rem; margin-bottom: 12px; opacity: 0.85;">⚡ Quick Specification Presets</div>', unsafe_allow_html=True)
preset_col1, preset_col2, preset_col3 = st.columns(3)

with preset_col1:
    if st.button("🟢 Budget Profile"):
        st.session_state['ram_gb'] = 4
        st.session_state['storage_gb'] = 64
        st.session_state['battery_mah'] = 3500
        st.session_state['main_camera_mp'] = 12
        st.rerun()

with preset_col2:
    if st.button("🟡 Mid-Range Profile"):
        st.session_state['ram_gb'] = 8
        st.session_state['storage_gb'] = 128
        st.session_state['battery_mah'] = 5000
        st.session_state['main_camera_mp'] = 48
        st.rerun()

with preset_col3:
    if st.button("🔴 Flagship Profile"):
        st.session_state['ram_gb'] = 16
        st.session_state['storage_gb'] = 512
        st.session_state['battery_mah'] = 5000
        st.session_state['main_camera_mp'] = 108
        st.rerun()

st.markdown('<div style="margin-bottom: 25px;"></div>', unsafe_allow_html=True)

# Main Form Container (Glassmorphic)
with st.container(border=True):
    
    # 2-Column layout for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 Memory & Storage")
        # RAM Select box (Options: 4, 6, 8, 12, 16)
        ram_val = st.selectbox(
            "RAM Capacity (GB)",
            options=[4, 6, 8, 12, 16],
            key="ram_gb",
            help="Select the smartphone RAM capacity in Gigabytes."
        )
        
        # Storage Select box (Options: 64, 128, 256, 512, 1024)
        storage_val = st.selectbox(
            "Internal Storage (GB)",
            options=[64, 128, 256, 512, 1024],
            key="storage_gb",
            help="Select the internal storage capacity in Gigabytes."
        )
        
    with col2:
        st.subheader("🔋 Battery & Camera")
        # Battery Capacity Slider (3000 to 7000 mAh)
        battery_val = st.slider(
            "Battery Capacity (mAh)",
            min_value=3000,
            max_value=7000,
            key="battery_mah",
            help="Drag to select the battery capacity in milliampere-hours."
        )
        
        # Main Camera Slider with fixed stops (8, 12, 16, 24, 48, 50, 64, 108, 200 MP)
        camera_val = st.select_slider(
            "Main Camera Resolution (MP)",
            options=[8, 12, 16, 24, 48, 50, 64, 108, 200],
            key="main_camera_mp",
            help="Select the primary rear camera resolution in Megapixels from standard values."
        )

# Centered Predict Button Layout
st.markdown('<div style="margin-bottom: 15px;"></div>', unsafe_allow_html=True)
col_l, col_btn, col_r = st.columns([1.2, 1, 1.2])

with col_btn:
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
            
            # Market Position Analysis & Tier Badge
            market_ratio = min(max((predicted_raw - 8000.0) / 212000.0, 0.0), 1.0)
            
            # Determine Tier Label, Icon, and Color
            if predicted_price < 25000:
                tier_label = "🟢 Budget Range (Excellent Value)"
                tier_desc = "Perfect for standard daily tasks, light media, and high battery efficiency."
            elif predicted_price < 60000:
                tier_label = "🟡 Mid-Range Performance"
                tier_desc = "Optimal balance of computational power, storage, and modern camera sensors."
            elif predicted_price < 100000:
                tier_label = "🟠 High-End Premium"
                tier_desc = "Exceptional performance, multi-tasking capabilities, and flagship-level main camera."
            else:
                tier_label = "🔴 Ultra-Flagship / Luxury Tier"
                tier_desc = "Ultimate raw specs, designed for extreme workflows, professional photography, and gaming."
                
            # Render Market Tier Description Box
            st.markdown(f"""
            <div style="margin-top: 25px; padding: 20px; border-radius: 12px; background: rgba(128, 128, 128, 0.05); border: 1px solid rgba(128, 128, 128, 0.1);">
                <div style="font-weight: 600; color: var(--text-color); font-size: 1rem;">Market Tier Evaluation:</div>
                <div style="font-weight: 800; color: var(--text-color); font-size: 1.1rem; margin-top: 5px;">{tier_label}</div>
                <div style="color: var(--text-color); opacity: 0.75; font-size: 0.88rem; margin-top: 6px; line-spacing: 1.3;">{tier_desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Render Progress Bar
            st.markdown('<div style="text-align: center; color: var(--text-color); margin-top: 25px; font-weight: 600; font-size: 0.85rem; opacity: 0.85;">Market Price Spectrum (₹8,000 - ₹220,000)</div>', unsafe_allow_html=True)
            st.progress(market_ratio)
            
            # Also invoke fallback success banner for clarity & guide adherence
            st.success(f"Successfully calculated smartphone price: {formatted_price}")
            
        except Exception as e:
            st.error(f"Inference error: {str(e)}")

# Footer (General Non-Academic text)
st.markdown("""
<div style="text-align: center; margin-top: 60px; color: var(--text-color); opacity: 0.45; font-size: 0.85rem;">
    PhoneMatrix © 2026 • Intelligent Smartphone Valuation Engine • Powered by Random Forest Regression
</div>
""", unsafe_allow_html=True)
