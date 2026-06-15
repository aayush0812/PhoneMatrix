import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re

# Set page config for a premium and modern look
st.set_page_config(
    page_title="PhoneMatrix",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load raw dataset and parse specs for competitor matching (Cached for instant load)
@st.cache_data
def load_clean_dataset():
    if not os.path.exists("device_specs_structured_dataset.csv"):
        return None
    try:
        df = pd.read_csv("device_specs_structured_dataset.csv")
        
        # Local regex parser matching train_model.py
        def parse_spec_local(val_str, default_unit="GB"):
            if pd.isna(val_str):
                return None
            val_str = str(val_str).strip()
            first_part = re.split(r'[/+]', val_str)[0].strip()
            match = re.search(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?', first_part)
            if not match:
                return None
            val = float(match.group(1))
            unit = match.group(2)
            if unit:
                unit = unit.upper()
            else:
                unit = default_unit.upper()
            if unit == "MB":
                return val / 1024.0
            elif unit == "TB":
                return val * 1024.0
            elif unit == "KB":
                return val / (1024.0 * 1024.0)
            return val

        # Clean key specs for distance calculations
        df['ram_gb'] = df['ram_raw'].apply(lambda x: parse_spec_local(x, 'GB'))
        df['storage_gb'] = df['storage_raw'].apply(lambda x: parse_spec_local(x, 'GB'))
        df['battery_mah'] = df['battery_raw'].apply(lambda x: parse_spec_local(x, 'mAh'))
        df['main_camera_mp'] = df['rear_camera_raw'].apply(lambda x: parse_spec_local(x, 'MP'))
        
        # Keep rows with valid features and target
        df = df.dropna(subset=['ram_gb', 'storage_gb', 'battery_mah', 'main_camera_mp', 'price_inr'])
        return df
    except Exception as e:
        return None

# Find top 3 closest competitor matches in dataset
def get_top_matches(df, ram, storage, battery, camera, top_n=3):
    df_dist = df.copy()
    # Euclidean distance normalized by standard feature scale ranges
    df_dist['dist'] = (
        ((df_dist['ram_gb'] - ram) / 12.0) ** 2 +
        ((df_dist['storage_gb'] - storage) / 960.0) ** 2 +
        ((df_dist['battery_mah'] - battery) / 4000.0) ** 2 +
        ((df_dist['main_camera_mp'] - camera) / 192.0) ** 2
    )
    # Return sorted top records
    return df_dist.sort_values(by='dist').head(top_n)

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
    
    /* VFM Auditor Box Styling */
    .vfm-box {
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        font-weight: 600;
        text-align: center;
    }
    .vfm-hot {
        background: rgba(46, 204, 113, 0.15);
        border: 1px solid rgba(46, 204, 113, 0.4);
        color: #2ecc71;
    }
    .vfm-fair {
        background: rgba(52, 152, 219, 0.15);
        border: 1px solid rgba(52, 152, 219, 0.4);
        color: #3498db;
    }
    .vfm-over {
        background: rgba(231, 76, 60, 0.15);
        border: 1px solid rgba(231, 76, 60, 0.4);
        color: #e74c3c;
    }
    
    /* Competitor Card Styling */
    .competitor-card {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.1);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .competitor-name {
        font-weight: 700;
        color: var(--text-color);
        font-size: 0.95rem;
    }
    .competitor-price {
        font-weight: 800;
        color: #00f2fe;
        font-size: 0.95rem;
    }
    .competitor-specs {
        font-size: 0.78rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-top: 4px;
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
        st.session_state['ram_gb'] = 6
        st.session_state['storage_gb'] = 64
        st.session_state['battery_mah'] = 5000
        st.session_state['main_camera_mp'] = 48
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
        
    st.markdown('<hr style="border: 0; border-top: 1px solid rgba(128, 128, 128, 0.15); margin: 20px 0;">', unsafe_allow_html=True)
    
    # Value for Money Auditor Input
    st.subheader("🔍 Valuation Auditor (Optional)")
    store_price = st.number_input(
        "Compare with Store Price (₹)",
        min_value=0,
        value=0,
        step=1000,
        help="Optional: Enter a store price to check if the device is a good deal compared to the ML prediction."
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
            
            # 1. Value for Money (VFM) Auditor Logic
            if store_price > 0:
                vfm_ratio = predicted_price / store_price
                price_diff = abs(predicted_price - store_price)
                
                if vfm_ratio >= 1.15:
                    vfm_class = "vfm-hot"
                    vfm_text = f"🔥 Hot Deal! (Underpriced by ₹{price_diff:,} compared to spec value)"
                elif 0.90 <= vfm_ratio < 1.15:
                    vfm_class = "vfm-fair"
                    vfm_text = "⚖️ Fair Valuation (Price matches component specifications)"
                else:
                    vfm_class = "vfm-over"
                    vfm_text = f"⚠️ Overpriced! (You are paying a ₹{price_diff:,} markup above component value)"
                    
                st.markdown(f"""
                <div class="vfm-box {vfm_class}">
                    {vfm_text}
                </div>
                """, unsafe_allow_html=True)
            
            # 2. Hardware Performance Ratings Calculations
            ram_map = {4: 1.5, 6: 2.5, 8: 3.5, 12: 4.5, 16: 5.0}
            storage_map = {64: 1.5, 128: 2.5, 256: 3.5, 512: 4.5, 1024: 5.0}
            camera_map = {8: 1.5, 12: 2.0, 16: 2.5, 24: 3.0, 48: 4.0, 50: 4.2, 64: 4.5, 108: 4.8, 200: 5.0}
            
            gaming_score = (ram_map[ram_val] + storage_map[storage_val]) / 2.0
            battery_score = 2.0 + (battery_val - 3000.0) / 4000.0 * 3.0
            photo_score = camera_map[camera_val]
            
            def get_star_rating(score):
                stars = int(round(score))
                return "⭐" * stars + "⚫" * (5 - stars)
                
            st.markdown('<div style="font-weight: 600; color: var(--text-color); font-size: 1rem; margin-top: 25px; margin-bottom: 10px;">📊 Estimated Hardware Performance</div>', unsafe_allow_html=True)
            rate_col1, rate_col2, rate_col3 = st.columns(3)
            with rate_col1:
                st.markdown(f"""
                <div style="text-align: center; background: rgba(128, 128, 128, 0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(128, 128, 128, 0.1);">
                    <div style="font-size: 0.8rem; opacity: 0.7; color: var(--text-color);">Gaming Power</div>
                    <div style="font-size: 1.05rem; font-weight: bold; margin-top: 4px; color: var(--text-color);">{get_star_rating(gaming_score)}</div>
                </div>
                """, unsafe_allow_html=True)
            with rate_col2:
                st.markdown(f"""
                <div style="text-align: center; background: rgba(128, 128, 128, 0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(128, 128, 128, 0.1);">
                    <div style="font-size: 0.8rem; opacity: 0.7; color: var(--text-color);">Battery Life</div>
                    <div style="font-size: 1.05rem; font-weight: bold; margin-top: 4px; color: var(--text-color);">{get_star_rating(battery_score)}</div>
                </div>
                """, unsafe_allow_html=True)
            with rate_col3:
                st.markdown(f"""
                <div style="text-align: center; background: rgba(128, 128, 128, 0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(128, 128, 128, 0.1);">
                    <div style="font-size: 0.8rem; opacity: 0.7; color: var(--text-color);">Photography</div>
                    <div style="font-size: 1.05rem; font-weight: bold; margin-top: 4px; color: var(--text-color);">{get_star_rating(photo_score)}</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Market Position Analysis & Tier Badge
            market_ratio = min(max((float(predicted_raw) - 8000.0) / 212000.0, 0.0), 1.0)
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
            
            # 3. Similar Competitor Matches
            df_dataset = load_clean_dataset()
            if df_dataset is not None:
                matches = get_top_matches(df_dataset, float(ram_val), float(storage_val), float(battery_val), float(camera_val))
                
                st.markdown('<div style="margin-top: 30px; font-weight: 600; color: var(--text-color); font-size: 1rem; margin-bottom: 10px;">🔍 Similar Market Competitors (from Dataset)</div>', unsafe_allow_html=True)
                
                for idx, row in matches.iterrows():
                    spec_summary = f"{int(row['ram_gb'])}GB RAM • {int(row['storage_gb'])}GB Storage • {int(row['battery_mah'])}mAh Battery • {int(row['main_camera_mp'])}MP Camera"
                    formatted_actual = f"₹{int(row['price_inr']):,}"
                    
                    st.markdown(f"""
                    <div class="competitor-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="competitor-name">{row['phone_name']}</span>
                            <span class="competitor-price">{formatted_actual}</span>
                        </div>
                        <div class="competitor-specs">{spec_summary}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
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
