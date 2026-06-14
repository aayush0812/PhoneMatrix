import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def parse_spec(val_str, default_unit="GB"):
    """
    Extract the first continuous numeric value and convert it to a standard unit.
    For RAM/Storage: Convert MB to GB and TB to GB.
    For Battery: Extract mAh.
    For Camera: Extract primary sensor resolution in MP.
    """
    if pd.isna(val_str):
        return None
    
    val_str = str(val_str).strip()
    
    # Extract the first segment before any separator '/' or '+'
    first_part = re.split(r'[/+]', val_str)[0].strip()
    
    # Regular expression to extract the first numeric value and optional unit
    match = re.search(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?', first_part)
    if not match:
        return None
    
    val = float(match.group(1))
    unit = match.group(2)
    
    # Normalize unit to uppercase
    if unit:
        unit = unit.upper()
    else:
        unit = default_unit.upper()
        
    # Standardize to GB/mAh/MP units
    if unit == "MB":
        return val / 1024.0
    elif unit == "TB":
        return val * 1024.0
    elif unit == "KB":
        return val / (1024.0 * 1024.0)
        
    return val

def train_pipeline():
    # 1. Load Data
    print("Loading raw dataset...")
    df = pd.read_csv("device_specs_structured_dataset.csv")
    
    # 2. Apply text extraction (regex)
    print("Preprocessing data and applying feature extraction...")
    df['ram_gb'] = df['ram_raw'].apply(lambda x: parse_spec(x, 'GB'))
    df['storage_gb'] = df['storage_raw'].apply(lambda x: parse_spec(x, 'GB'))
    df['battery_mah'] = df['battery_raw'].apply(lambda x: parse_spec(x, 'mAh'))
    df['main_camera_mp'] = df['rear_camera_raw'].apply(lambda x: parse_spec(x, 'MP'))
    
    # Target variable mapping
    df['price'] = df['price_inr']
    
    # 3. Clean Data: select features and target, and drop missing values
    feature_cols = ['ram_gb', 'storage_gb', 'battery_mah', 'main_camera_mp']
    clean_df = df[feature_cols + ['price']].dropna()
    
    print(f"Dataset shape before cleaning: {df.shape}")
    print(f"Dataset shape after cleaning and dropping NaNs: {clean_df.shape}")
    
    # 4. Train/Test Split
    X = clean_df[feature_cols]
    y = clean_df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Model Training
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 6. Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): INR {mae:.2f}")
    print(f"R-squared (R²) Score: {r2:.4f}")
    print("--------------------------------\n")
    
    # 7. Export Model
    model_filename = "smartphone_model.pkl"
    print(f"Exporting trained model to {model_filename}...")
    joblib.dump(model, model_filename)
    print("Model successfully exported!")

if __name__ == "__main__":
    train_pipeline()
